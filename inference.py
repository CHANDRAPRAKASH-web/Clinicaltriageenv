"""
Inference Script — ClinicalTriageEnv
=====================================
MANDATORY ENV VARS (injected by the validator):
    API_BASE_URL   The API endpoint for the LLM proxy.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

import os
import json
import time
import requests
import textwrap

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -----------------------------------------------
# ENV SETUP
# -----------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "not-set"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

SERVER_URL = os.getenv("ENV_URL") or os.getenv("OPENENV_URL") or "http://127.0.0.1:7860"
BENCHMARK = "clinicaltriageenv"

MAX_STEPS = 12
SUCCESS_THRESHOLD = 0.5

def get_client():
    if OpenAI is None:
        raise ImportError("The 'openai' Python library is not installed.")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

_client = None
def client():
    global _client
    if _client is None: _client = get_client()
    return _client

SYSTEM_PROMPT = textwrap.dedent("""\
You are a senior clinical triage AI. Your goal is to diagnose and triage patients FAST.
Guidelines: WHO/ICMR rural triage.

ACTIONS:
1. {"action": "ask_patient", "question": "..."}
2. {"action": "request_test", "test": "..."}
3. {"action": "request_vital", "vital": "..."}
4. {"action": "make_assessment", "risk": "<LOW|MODERATE|HIGH|CRITICAL>", "condition": "...", "next_step": "..."}

STRATEGY:
- Step 1-2: Ask about the most dangerous red flags (e.g., blood in cough, fever pattern, confusion).
- Step 3: Order a vital sign (temperature/BP) or a specific test (RDT/Sputum).
- Step 4-5: You MUST call 'make_assessment'. Do NOT keep asking questions once you have a suspicion.
- Efficiency is key. If you have enough info, ASSESS IMMEDIATELY.

Return ONLY valid JSON.""")

def call_llm(messages: list) -> str:
    response = client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=250,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def parse_llm_response(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try: return json.loads(text[start:end])
            except: pass
    return {"action": "ask_patient", "question": "Please describe your symptoms."}

def get_llm_action(observation: dict) -> dict:
    history = observation.get("conversation_history", [])
    tests = observation.get("tests_ordered", [])
    step = observation.get("steps_taken", 0)

    prompt = [
        f"PATIENT COMPLAINT: {observation.get('presenting_complaint')}",
        f"STEP: {step} / {MAX_STEPS}",
        "\nFULL HISTORY:"
    ]
    # Provide the full history so the agent remembers symptoms
    for h in history: prompt.append(f"  - {h}")
    
    if tests:
        prompt.append("\nTEST RESULTS:")
        for t in tests: prompt.append(f"  - {t['name']}: {t['result']}")

    # Strong nudge if we've already asked plenty of questions
    if step >= 4:
        prompt.append("\nCRITICAL: You have enough information. Do not ask more questions. Use 'make_assessment' NOW.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(prompt)}
    ]

    return parse_llm_response(call_llm(messages))

def run_task(task_name: str):
    step_num = 0
    rewards = []
    try:
        resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name}, timeout=15)
        data = resp.json()
        session_id, observation = data["session_id"], data["observation"]

        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        for i in range(1, MAX_STEPS + 1):
            step_num = i
            try:
                action = get_llm_action(observation)
                action_str = f"{action['action']}"
                if action.get("question"): action_str += f"('{action['question'][:30]}...')"
                elif action.get("test"): action_str += f"('{action['test']}')"
            except Exception as e:
                print(f"[STEP] step={step_num} action=error reward=0.00 done=false error={e}", flush=True)
                rewards.append(0.0); continue

            s_resp = requests.post(f"{SERVER_URL}/step", json={"session_id": session_id, "action": action}, timeout=15)
            s_data = s_resp.json()
            observation, reward, done = s_data["observation"], s_data["reward"], s_data["done"]
            rewards.append(reward)

            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            if done: break

        final_score = rewards[-1] if rewards else 0.0
        success = final_score >= SUCCESS_THRESHOLD
        print(f"[END] success={str(success).lower()} steps={step_num} score={final_score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

    except Exception as exc:
        print(f"[END] success=false steps={step_num} score=0.00 rewards=0.00", flush=True)

if __name__ == "__main__":
    task = os.getenv("OPENENV_TASK") or os.getenv("TASK_NAME")
    if task: run_task(task)
    else:
        for t in ["easy_malaria", "medium_tb", "hard_sepsis"]:
            run_task(t); time.sleep(1)