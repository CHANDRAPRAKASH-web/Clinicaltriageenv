import os
import json
import time
import requests
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL = "http://127.0.0.1:7860"

# Use whichever key is provided
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "")

effective_key = HF_TOKEN if (HF_TOKEN and not HF_TOKEN.startswith("gsk_")) else OPENAI_API_KEY
if not effective_key and HF_TOKEN and HF_TOKEN.startswith("gsk_"):
    effective_key = HF_TOKEN

# Smart-detect Groq
effective_base = API_BASE_URL
if effective_key and effective_key.startswith("gsk_") and not effective_base:
    effective_base = "https://api.groq.com/openai/v1"
elif not effective_base:
    effective_base = "https://router.huggingface.co/v1"

MAX_STEPS = 7
TEMPERATURE = 0.2
MAX_TOKENS = 300
SUCCESS_THRESHOLD = 0.5

# Initialize client
print(f"📡 Connecting to API: {effective_base}")
print(f"🔑 Using Key: {effective_key[:8]}...{effective_key[-4:] if len(effective_key)>4 else ''}")

client = OpenAI(
    base_url=effective_base,
    api_key=effective_key
)

def get_llm_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Sends current observation to LLM and returns parsed JSON action."""
    
    system_prompt = (
        "You are an AI Preventive Health Screening Assistant.\n"
        "Goal: Diagnose the patient correctly.\n\n"
        "Available Actions (Output ONLY this JSON format):\n"
        "1. { \"action\": \"ask_patient\", \"question\": \"...\" }\n"
        "2. { \"action\": \"request_test\", \"test\": \"...\" }\n"
        "3. { \"action\": \"request_vital\", \"vital\": \"...\" }\n"
        "4. { \"action\": \"make_assessment\", \"risk\": \"...\", \"condition\": \"...\", \"next_step\": \"...\" }\n\n"
        "Rules:\n"
        "1. SPEED IS EVERYTHING. You have ONLY 7 STEPS.\n"
        "2. Step 1-4: Ask the most important clinical questions to the patient.\n"
        "3. Step 5-6: Order the most important tests and vitals.\n"
        "4. Step 7: YOU MUST CALL 'make_assessment' NO MATTER WHAT. If you don't, you fail.\n"
    )
    
    # Constructing a readable history for the LLM
    history_text = "\n".join(observation.get("conversation_history", []))
    active_tests = json.dumps(observation.get("tests_ordered", []), indent=2)
    
    user_prompt = (
        f"Patient Profile: {observation['patient_profile']}\n"
        f"Initial Complaint: {observation['presenting_complaint']}\n\n"
        f"History so far:\n{history_text}\n\n"
        f"Vitals & Lab Results:\n{active_tests}\n\n"
        f"Steps Taken: {observation['steps_taken']}/{MAX_STEPS}\n"
        "What is your next action? Output JSON only."
    )
    
    if int(observation['steps_taken']) >= MAX_STEPS - 1:
        user_prompt += "\n\n🚨 CRITICAL RULE: THIS IS YOUR FINAL STEP. YOU MUST OUTPUT 'make_assessment' NOW! DO NOT ASK MORE QUESTIONS. DO NOT ORDER TESTS. YOU WILL FAIL IF YOU DO NOT ASSESS."
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        res = json.loads(content)
        
        # --- NEW: Robust Auto-Correction Layer ---
        # Map common hallucinations back to valid actions
        action_name = res.get("action", "").lower()
        if "ask" in action_name:
            res["action"] = "ask_patient"
        elif "test" in action_name:
            res["action"] = "request_test"
        elif "vital" in action_name:
            res["action"] = "request_vital"
        elif "assessment" in action_name or "diagnos" in action_name:
            res["action"] = "make_assessment"
            
        # If 'action' is totally missing but they provided a question
        if "action" not in res and "question" in res:
            res["action"] = "ask_patient"
            
        # --- NEW: MUST ASSESS ON FINAL STEP ---
        if int(observation['steps_taken']) >= MAX_STEPS - 1 and res["action"] != "make_assessment":
            res["action"] = "make_assessment"
            res["condition"] = res.get("condition", "Undetermined Condition")
            res["risk"] = res.get("risk", "Medium")
            res["next_step"] = res.get("next_step", "Refer to specialist")
        # ------------------------------------------

        return res
        
    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        # Default fallback action
        return {"action": "ask_patient", "question": "Can you describe your symptoms in more detail?"}

def run_task(task_name: str) -> None:
    """Runs a single clinical triage task end-to-end."""
    
    # 1. Reset Environment
    reset_resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name})
    if reset_resp.status_code != 200:
        print(f"Failed to reset task {task_name}: {reset_resp.text}")
        return
        
    data = reset_resp.json()
    session_id = data["session_id"]
    observation = data["observation"]
    
    print(f"[START] task={task_name} env=clinicaltriageenv model={MODEL_NAME}")
    
    all_rewards = []
    total_steps = 0
    
    # 2. Loop Actions
    for i in range(MAX_STEPS):
        total_steps += 1
        
        # a. Get Action from LLM
        action = get_llm_action(observation)
        
        # b. Send Step to Environment
        step_resp = requests.post(f"{SERVER_URL}/step", json={
            "session_id": session_id,
            "action": action
        })
        
        if step_resp.status_code != 200:
            print(f"[STEP] step={i+1} action={action.get('action')} reward=0.0 done=true error='{step_resp.text}'")
            break
            
        step_data = step_resp.json()
        new_obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        
        # --- NEW: Print readable output for the user ---
        action_type = action.get("action")
        if action_type == "ask_patient":
            print(f"  🩺 Doctor: {action.get('question')}")
            # The last two entries in history are the Q and A
            if len(new_obs["conversation_history"]) >= 2:
                print(f"  👤 Patient: {new_obs['conversation_history'][-1].replace('A: ', '')}")
        elif action_type == "request_vital" or action_type == "request_test":
            item = action.get("vital") or action.get("test")
            print(f"  🧪 Ordering {action_type}: {item}")
            if new_obs["tests_ordered"]:
                print(f"  📊 Result: {new_obs['tests_ordered'][-1]['result']}")
        elif action_type == "make_assessment":
            print(f"  📝 FINAL ASSESSMENT: {action.get('condition')} (Risk: {action.get('risk')})")
            print(f"  🚩 Next Step: {action.get('next_step')}")
        # -----------------------------------------------

        observation = new_obs
        all_rewards.append(reward)
        
        # c. Log Step (OpenEnv Format)
        print(f"[STEP] step={i+1} action={action.get('action')} reward={reward:.2f} done={str(done).lower()} error=null")
        
        if done:
            break
            
    # 3. Final Logging
    # The true episode score is exactly the final assessment score returned by the Grader.
    # Intermediate shaping rewards are kept for logging but not randomly summed.
    final_score = all_rewards[-1] if all_rewards else 0.0
    
    success = final_score >= SUCCESS_THRESHOLD
    rewards_str = ",".join([f"{r:.2f}" for r in all_rewards])
    
    print(f"[END] success={str(success).lower()} steps={total_steps} score={final_score:.2f} rewards={rewards_str}")

def run_all() -> None:
    """Runs all tasks in sequence."""
    tasks = ["easy_malaria", "tb_cough", "diabetic_sepsis"]
    for t in tasks:
        try:
            run_task(t)
            print("\n" + "="*50)
            time.sleep(5)  # Increased pause to avoid 429 rate limit errors
        except Exception as e:
            print(f"Failed to execute task {t}: {e}")

if __name__ == "__main__":
    run_all()
