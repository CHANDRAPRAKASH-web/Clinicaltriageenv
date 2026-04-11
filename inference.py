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

from openai import OpenAI

# -----------------------------------------------
# ENV SETUP — matches official sample script exactly
# -----------------------------------------------
# The validator injects HF_TOKEN; the error page references API_KEY.
# Handle BOTH, exactly like the official sample does.
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

MAX_STEPS = 12
SUCCESS_THRESHOLD = 0.5
SERVER_URL = "http://127.0.0.1:7860"
BENCHMARK = "clinical_triage"

# -----------------------------------------------
# OPENAI CLIENT — single instance, created at
# module level so the proxy sees it immediately
# -----------------------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

SYSTEM_PROMPT = textwrap.dedent("""\
You are a clinical triage AI agent deployed in a rural Indian health sub-centre.
You are triaging patients based on WHO/ICMR guidelines.

At each step you will receive the patient's profile, presenting complaint,
conversation history, and tests ordered so far.

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
- Order targeted tests (RDT for malaria, sputum test for TB, blood culture for sepsis)
- Use WHO danger signs to classify risk level
- For HIGH/CRITICAL risk: always refer (PHC, district hospital, or immediate hospitalization)
- Be efficient with your steps
- Return ONLY valid JSON, no extra text.""")


# -----------------------------------------------
# LLM CALL — routes through the validator's proxy
# -----------------------------------------------
def call_llm(messages: list) -> str:
    """Call the LLM via the validator-provided proxy and return the response content."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM's JSON response, handling markdown code fences."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from mixed text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    # Fallback if LLM returns unparseable text
    return {"action": "ask_patient", "question": "Can you describe your main symptoms?"}


def build_observation_prompt(observation: dict) -> str:
    """Convert the observation dict into a readable prompt for the LLM."""
    profile = observation.get("patient_profile", {})
    history = observation.get("conversation_history", [])
    tests = observation.get("tests_ordered", [])

    parts = [
        f"Patient: {profile.get('age', '?')} year old {profile.get('gender', '?')}, from {profile.get('location', '?')}",
        f"Chief Complaint: {observation.get('presenting_complaint', 'Unknown')}",
        f"Step: {observation.get('steps_taken', 0)} / {observation.get('max_steps', 8)}",
    ]

    if history:
        parts.append("\nConversation History:")
        for entry in history:
            parts.append(f"  {entry}")

    if tests:
        parts.append("\nTests/Vitals Ordered:")
        for t in tests:
            parts.append(f"  [{t.get('type', '')}] {t.get('name', '')}: {t.get('result', 'pending')}")

    return "\n".join(parts)


def get_llm_action(observation: dict) -> dict:
    """Use the LLM to decide the next clinical action."""
    user_prompt = build_observation_prompt(observation)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # If we're near the end of allowed steps, nudge toward final assessment
    steps_taken = observation.get("steps_taken", 0)
    max_steps = observation.get("max_steps", 8)
    if steps_taken >= max_steps - 2:
        messages.append({
            "role": "user",
            "content": "You are running low on steps. Make your final assessment NOW using make_assessment.",
        })

    response_text = call_llm(messages)
    action = parse_llm_response(response_text)

    # Validate action has required fields
    if "action" not in action:
        action = {"action": "ask_patient", "question": "Please describe your symptoms in detail."}

    return action


# -----------------------------------------------
# TASK RUNNER
# -----------------------------------------------
def run_task(task_name: str):
    step = 0
    try:
        reset_resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name})
        data = reset_resp.json()

        session_id = data["session_id"]
        observation = data["observation"]

        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

        rewards = []

        for step in range(1, MAX_STEPS + 1):
            error_msg = "null"

            try:
                action = get_llm_action(observation)
            except Exception as e:
                error_msg = str(e)
                print(f"[STEP] step={step} action=error reward=0.00 done=false error={error_msg}")
                rewards.append(0.0)
                continue

            step_resp = requests.post(f"{SERVER_URL}/step", json={
                "session_id": session_id,
                "action": action,
            })

            step_data = step_resp.json()
            observation = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]

            rewards.append(reward)

            print(f"[STEP] step={step} action={action['action']} reward={reward:.2f} done={str(done).lower()} error={error_msg}")

            if done:
                break

        final_score = rewards[-1] if rewards else 0.0
        success = final_score >= SUCCESS_THRESHOLD
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(f"[END] success={str(success).lower()} steps={step} score={final_score:.2f} rewards={rewards_str}")

    except Exception as exc:
        print(f"[END] success=false steps={step} score=0.00 rewards=0.00")
        print(f"[FATAL] {exc}")


def run_all():
    for t in ["easy_malaria", "medium_tb", "hard_sepsis"]:
        run_task(t)
        time.sleep(1)


if __name__ == "__main__":
    run_all()