"""
Inference Script — ClinicalTriageEnv
=====================================
MANDATORY ENV VARS (injected by the validator):
    API_BASE_URL   The API endpoint for the LLM proxy.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import time
import requests
import textwrap

# Lazy import of OpenAI to allow the script to start even if dependencies are weird
try:
    from openai import OpenAI
except ImportError:
    # This shouldn't happen if requirements.txt is installed, 
    # but we'll catch it to be safe.
    OpenAI = None

# -----------------------------------------------
# ENV SETUP — matches official sample script exactly
# -----------------------------------------------
# The validator injects HF_TOKEN; the error page references API_KEY.
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "not-set"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Server URL — detection for both local and containerized validation
SERVER_URL = os.getenv("ENV_URL") or os.getenv("OPENENV_URL") or "http://127.0.0.1:7860"
BENCHMARK = "ClinicalTriageEnv"

MAX_STEPS = 12
SUCCESS_THRESHOLD = 0.5

# -----------------------------------------------
# OPENAI CLIENT FACTORY
# -----------------------------------------------
def get_openai_client():
    if OpenAI is None:
        raise ImportError("The 'openai' Python library is not installed.")
    
    # We use a dummy key if none is provided to prevent immediate crash at module level.
    # The actual call will fail later if the key is invalid, which is easier to debug.
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

# Singleton-like client
_client = None
def get_client():
    global _client
    if _client is None:
        _client = get_openai_client()
    return _client


SYSTEM_PROMPT = textwrap.dedent("""\
You are a clinical triage AI agent deployed in a rural Indian health sub-centre.
You are triaging patients based on WHO/ICMR guidelines.

You must return a JSON object with ONE of these actions:

1. Ask the patient a question:
   {"action": "ask_patient", "question": "<your clinical question>"}

2. Order a diagnostic test:
   {"action": "request_test", "test": "<test name>"}

3. Check a vital sign:
   {"action": "request_vital", "vital": "<vital name>"}

4. Make your final assessment (do this when you have enough information):
   {"action": "make_assessment", "risk": "<LOW|MODERATE|HIGH|CRITICAL>", "condition": "<suspected_condition>", "next_step": "<referral_or_action_plan>"}

CLINICAL RULES:
- Ask about red flag symptoms early (fever pattern, blood in sputum, confusion)
- For HIGH/CRITICAL risk: always refer (PHC, district hospital, or immediate hospital)
- Return ONLY valid JSON, no extra text.""")


def call_llm(messages: list) -> str:
    """Call the LLM via the validator-provided proxy."""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM's JSON response."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try: return json.loads(text[start:end])
            except: pass
    return {"action": "ask_patient", "question": "Can you describe your symptoms?"}


def get_llm_action(observation: dict) -> dict:
    """Use the LLM to decide the next action."""
    profile = observation.get("patient_profile", {})
    history = observation.get("conversation_history", [])
    tests = observation.get("tests_ordered", [])

    prompt_lines = [
        f"Complaint: {observation.get('presenting_complaint', 'Unknown')}",
        f"Step: {observation.get('steps_taken', 0)} / {observation.get('max_steps', 8)}",
    ]
    if history: prompt_lines.append("\nHistory: " + " | ".join(history[-4:]))
    if tests: prompt_lines.append("\nTests: " + ", ".join([f"{t['name']}:{t['result']}" for t in tests]))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(prompt_lines)}
    ]

    response_text = call_llm(messages)
    return parse_llm_response(response_text)


def run_task(task_name: str):
    step_num = 0
    rewards = []
    
    try:
        # 1. Reset Environment
        reset_resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name}, timeout=10)
        reset_resp.raise_for_status()
        data = reset_resp.json()

        session_id = data["session_id"]
        observation = data["observation"]

        # START LOG
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        for i in range(1, MAX_STEPS + 1):
            step_num = i
            error_msg = "null"

            # 2. Get LLM Action
            try:
                action = get_llm_action(observation)
                # Format action for logs
                action_str = f"{action['action']}"
                if action.get("question"): action_str += f"('{action['question'][:20]}...')"
                elif action.get("test"): action_str += f"('{action['test']}')"
                elif action.get("risk"): action_str += f"('{action['risk']}')"
            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                print(f"[STEP] step={step_num} action=error reward=0.00 done=false error={error_msg}", flush=True)
                rewards.append(0.0)
                continue

            # 3. Step Environment
            step_resp = requests.post(f"{SERVER_URL}/step", json={
                "session_id": session_id,
                "action": action,
            }, timeout=10)
            
            step_data = step_resp.json()
            observation = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]

            rewards.append(reward)

            # STEP LOG
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}", flush=True)

            if done:
                break

        # 4. Final results
        final_score = rewards[-1] if rewards else 0.0
        success = final_score >= SUCCESS_THRESHOLD
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        # END LOG
        print(f"[END] success={str(success).lower()} steps={step_num} score={final_score:.2f} rewards={rewards_str}", flush=True)

    except Exception as exc:
        print(f"[END] success=false steps={step_num} score=0.00 rewards=0.00", flush=True)
        # Optional: detail logs for debugging
        # print(f"[DEBUG] Fatal error in run_task: {exc}")


def run_all():
    # Detect if a single task is requested via environment variable (common in validators)
    single_task = os.getenv("OPENENV_TASK") or os.getenv("TASK_NAME")
    if single_task:
        run_task(single_task)
    else:
        # Local baseline test
        for t in ["easy_malaria", "medium_tb", "hard_sepsis"]:
            run_task(t)
            time.sleep(0.5)


if __name__ == "__main__":
    run_all()