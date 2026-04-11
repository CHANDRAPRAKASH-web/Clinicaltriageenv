import os
import json
import time
import requests

# LITELLM / PROXY SETTINGS (Strict)
API_KEY = (os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or "not-set").strip()
API_BASE_URL = (os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1").strip()
MODEL_NAME = (os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct").strip()

SERVER_URL = "http://127.0.0.1:7860"
BENCHMARK = "clinicaltriageenv"

# IMPORT OPENAI
from openai import OpenAI

SYSTEM_PROMPT = """You are a Senior Clinical Triage AI. Be decisive and efficient.
GOAL: Diagnose the patient in 4-5 steps max.

PROTOCOL:
1. ASK: Ask about specific WHO red-flag symptoms (e.g. fever pattern, breathing difficulty).
2. TEST: Order a vital sign (request_vital) or diagnostic test (request_test) like BP, Temperature, RDT, or Sputum.
3. ASSESS: As soon as you have a suspicion, call 'make_assessment' with risk level (LOW/MODERATE/HIGH/CRITICAL).

RULES:
- Do NOT repeat questions.
- Do NOT ask more than 3 questions total.
- You MUST provide a 'make_assessment' action by step 5.
- Return ONLY valid JSON."""

def run_task(task_name: str):
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[FATAL] Client Init Error: {e}")
        return

    # 1. Reset
    try:
        resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name}, timeout=10)
        data = resp.json()
        session_id, obs = data["session_id"], data["observation"]
    except Exception as e:
        print(f"[FATAL] Connection to environment failed: {e}")
        return

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    rewards = []

    # 2. Loop
    for step in range(1, 13):
        try:
            # Build clinical context
            history = " | ".join(obs.get("conversation_history", []))
            tests = ", ".join([f"{t['name']}:{t['result']}" for t in obs.get("tests_ordered", [])])
            
            user_content = f"COMPLAINT: {obs.get('presenting_complaint')}\nHISTORY: {history}\nTESTS: {tests}\nSTEP: {step}/12\nDECISION:"

            # LLM CALL
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1
            )
            
            text = response.choices[0].message.content.strip()
            if "```" in text: text = text.split("```")[1].replace("json", "").strip()
            action = json.loads(text)

            # STEP
            s_resp = requests.post(f"{SERVER_URL}/step", json={"session_id": session_id, "action": action}, timeout=10)
            s_data = s_resp.json()
            obs, reward, done = s_data["observation"], s_data["reward"], s_data["done"]
            rewards.append(reward)

            # Log formatted action
            act_type = action.get("action", "unknown")
            log_detail = action.get("question") or action.get("test") or action.get("vital") or action.get("risk") or ""
            print(f"[STEP] step={step} action={act_type}('{log_detail[:20]}') reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            if done: break
            
        except Exception as e:
            print(f"[STEP] step={step} action=error reward=0.00 done=true error={e}", flush=True)
            break

    score = rewards[-1] if rewards else 0.0
    print(f"[END] success={str(score >= 0.5).lower()} steps={len(rewards)} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

if __name__ == "__main__":
    task = os.getenv("OPENENV_TASK") or os.getenv("TASK_NAME")
    if task: run_task(task)
    else:
        for t in ["easy_malaria", "medium_tb", "hard_sepsis"]:
            run_task(t); time.sleep(1)