import os
import json
import time
import requests

# -----------------------------------------------
# ENV SETUP
# -----------------------------------------------
# We strip() to ensure no hidden whitespaces break the URL
API_KEY = (os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or "not-set").strip()
API_BASE_URL = (os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1").strip()
MODEL_NAME = (os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct").strip()

SERVER_URL = "http://127.0.0.1:7860"
BENCHMARK = "clinicaltriageenv"

def run_task(task_name: str):
    # Import inside function to avoid module-level crashes
    from openai import OpenAI
    
    # 0. Initialize Client inside task
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
            timeout=20.0,
            max_retries=3
        )
    except Exception as e:
        print(f"[FATAL] Client Initialization failed: {e}")
        return

    # 1. Reset Environment
    try:
        resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        session_id = data["session_id"]
        obs = data["observation"]
    except Exception as e:
        print(f"[FATAL] Server Reset failed: {e}")
        return

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    rewards = []

    # 2. Main Loop
    for step in range(1, 13):
        try:
            history = " | ".join(obs.get("conversation_history", []))
            prompt = f"Patient: {obs.get('presenting_complaint')}\nHistory: {history}\nAction (JSON):"
            
            # CALL LLM
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a clinical triage AI. Return JSON: {'action': 'ask_patient', 'question': '...'} or {'action': 'make_assessment', 'risk': '...', 'condition': '...', 'next_step': '...'}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            text = response.choices[0].message.content.strip()
            if "```" in text: text = text.split("```")[1].replace("json", "")
            action = json.loads(text)

            # STEP
            s_resp = requests.post(f"{SERVER_URL}/step", json={"session_id": session_id, "action": action}, timeout=10)
            s_data = s_resp.json()
            obs = s_data["observation"]
            reward = s_data["reward"]
            done = s_data["done"]
            rewards.append(reward)

            print(f"[STEP] step={step} action={action['action']} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            if done: break
            
        except Exception as e:
            # We print as [STEP] with error so validator sees progress stopped
            print(f"[STEP] step={step} action=llm_error reward=0.00 done=true error={e}", flush=True)
            break

    # 3. Final Report
    score = rewards[-1] if rewards else 0.0
    print(f"[END] success={str(score >= 0.5).lower()} steps={len(rewards)} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

if __name__ == "__main__":
    task = os.getenv("OPENENV_TASK") or os.getenv("TASK_NAME")
    if task:
        run_task(task)
    else:
        # For baseline verification
        for t in ["easy_malaria", "medium_tb", "hard_sepsis"]:
            run_task(t)
            time.sleep(1)