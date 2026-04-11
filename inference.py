import os
import json
import time
import requests

# LITELLM PROXY SETTINGS
# We use os.environ[] directly. If the validator fails to inject these, 
# the script will CRASH, which is better than "No API calls recorded".
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# FALLBACKS (Only for local testing, validator should provide these)
if not API_KEY: API_KEY = "not-set"
if not API_BASE_URL: API_BASE_URL = "https://router.huggingface.co/v1"

# IMPORT OPENAI
from openai import OpenAI

# Initialize client EXACTLY as the 'How to fix' instructions say
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

SERVER_URL = "http://127.0.0.1:7860"
BENCHMARK = "clinicaltriageenv"

def run_task(task_name: str):
    # 1. Reset
    resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name})
    data = resp.json()
    session_id = data["session_id"]
    obs = data["observation"]

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    rewards = []

    # 2. Loop
    for step in range(1, 13):
        # Construct prompt
        history = " | ".join(obs.get("conversation_history", []))
        prompt = f"Patient: {obs.get('presenting_complaint')}\nHistory: {history}\nStep: {step}/12. Action (JSON):"
        
        # LLM CALL (Strict - no try/except here so it fails loudly)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a clinical triage AI. Return JSON: {'action': 'ask_patient', 'question': '...'} or {'action': 'make_assessment', 'risk': '...', 'condition': '...', 'next_step': '...'}. Be brief."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # Parse
        text = response.choices[0].message.content.strip()
        if "```" in text: text = text.split("```")[1].replace("json", "")
        action = json.loads(text)

        # Step
        s_resp = requests.post(f"{SERVER_URL}/step", json={"session_id": session_id, "action": action})
        s_data = s_resp.json()
        obs = s_data["observation"]
        reward = s_data["reward"]
        done = s_data["done"]
        rewards.append(reward)

        print(f"[STEP] step={step} action={action['action']} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        if done: break

    # 3. End
    score = rewards[-1] if rewards else 0.0
    print(f"[END] success={str(score >= 0.5).lower()} steps={step} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

if __name__ == "__main__":
    task = os.getenv("OPENENV_TASK") or os.getenv("TASK_NAME")
    if task:
        run_task(task)
    else:
        for t in ["easy_malaria", "medium_tb", "hard_sepsis"]:
            run_task(t)
            time.sleep(1)