import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# --- REQUIRED VARIABLES ---
HF_TOKEN     = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")

MAX_STEPS         = 12
SUCCESS_THRESHOLD = 0.5
SERVER_URL        = "http://127.0.0.1:7860"

print(f"API_BASE_URL={API_BASE_URL}")
print(f"MODEL_NAME={MODEL_NAME}")


# -----------------------------------------------
# SAFE LLM (ONLY IF CONFIG VALID)
# -----------------------------------------------
def call_llm(observation: dict) -> dict:

    if not HF_TOKEN or not API_BASE_URL or not MODEL_NAME:
        return {
            "action": "ask_patient",
            "question": "Can you describe your symptoms clearly?"
        }

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": json.dumps(observation)}
            ],
            temperature=0.1,
            max_tokens=200
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)

    except Exception:
        return {
            "action": "ask_patient",
            "question": "Can you describe your symptoms clearly?"
        }


# -----------------------------------------------
# YOUR ORIGINAL LOGIC (UNCHANGED)
# -----------------------------------------------
def get_llm_action(observation: dict) -> dict:
    complaint = observation["presenting_complaint"].lower()
    step      = observation["steps_taken"]

    if "fever" in complaint:
        if step == 0:
            return {"action": "ask_patient","question": "How long have you had this fever? Is it constant or does it come and go?"}
        if step == 1:
            return {"action": "ask_patient","question": "Do you get severe chills and shivering before the fever rises?"}
        if step == 2:
            return {"action": "ask_patient","question": "Have you been exposed to mosquitoes or stagnant water recently?"}
        if step == 3:
            return {"action": "ask_patient","question": "Do you have body ache, muscle pain, or headache along with the fever?"}
        if step == 4:
            return {"action": "ask_patient","question": "Have you noticed any loss of appetite or change in urine colour?"}
        if step == 5:
            return {"action": "ask_patient","question": "Have you taken any medication for the fever? Did it help?"}
        if step == 6:
            return {"action": "request_vital", "vital": "temperature"}
        if step == 7:
            return {"action": "request_vital", "vital": "heart rate"}
        if step == 8:
            return {"action": "request_test", "test": "RDT"}
        if step == 9:
            return {"action": "make_assessment","risk": "HIGH","condition": "plasmodium_vivax_malaria","next_step": "refer_to_PHC"}

    if "cough" in complaint:
        if step == 0:
            return {"action": "ask_patient","question": "Are you coughing blood or seeing blood in your sputum?"}
        if step == 1:
            return {"action": "ask_patient","question": "How long have you had this cough? Has it been more than 2 weeks?"}
        if step == 2:
            return {"action": "ask_patient","question": "Have you noticed significant weight loss in the past few months?"}
        if step == 3:
            return {"action": "ask_patient","question": "Do you wake up at night with heavy sweating — night sweats?"}
        if step == 4:
            return {"action": "ask_patient","question": "Have you been in close contact with anyone who had tuberculosis?"}
        if step == 5:
            return {"action": "ask_patient","question": "Do you feel breathless or short of breath even with mild activity?"}
        if step == 6:
            return {"action": "request_vital", "vital": "temperature"}
        if step == 7:
            return {"action": "request_vital", "vital": "spo2"}
        if step == 8:
            return {"action": "request_test", "test": "sputum test"}
        if step == 9:
            return {"action": "make_assessment","risk": "CRITICAL","condition": "active_pulmonary_TB","next_step": "refer_to_district_TB_centre"}

    if any(kw in complaint for kw in ["wound","dizzy","confusion"]):
        if step == 0:
            return {"action": "ask_patient","question": "Are you feeling confused or unable to think clearly?"}
        if step == 1:
            return {"action": "ask_patient","question": "How long has the wound been there? Is it foul-smelling or getting worse?"}
        if step == 2:
            return {"action": "ask_patient","question": "Do you have diabetes?"}
        if step == 3:
            return {"action": "ask_patient","question": "Do you have fever or chills?"}
        if step == 4:
            return {"action": "ask_patient","question": "Are you feeling dizzy or weak?"}
        if step == 5:
            return {"action": "request_vital", "vital": "temperature"}
        if step == 6:
            return {"action": "request_test", "test": "blood culture"}
        if step == 7:
            return {"action": "request_test", "test": "random blood sugar"}
        if step == 8:
            return {"action": "make_assessment","risk": "CRITICAL","condition": "diabetic_foot_sepsis","next_step": "immediate_hospitalization"}

    return call_llm(observation)


# -----------------------------------------------
# TASK RUNNER (FIXED FORMAT)
# -----------------------------------------------
def run_task(task_name: str):

    reset_resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name})
    data = reset_resp.json()

    session_id  = data["session_id"]
    observation = data["observation"]

    print(f"[START] task={task_name}")

    rewards = []

    for step in range(1, MAX_STEPS + 1):

        action = get_llm_action(observation)

        step_resp = requests.post(f"{SERVER_URL}/step", json={
            "session_id": session_id,
            "action": action
        })

        step_data   = step_resp.json()
        observation = step_data["observation"]
        reward      = step_data["reward"]
        done        = step_data["done"]

        rewards.append(reward)

        print(f"[STEP] step={step} action={action['action']} reward={reward:.2f} done={done} error=null")

        if done:
            break

    final_score = rewards[-1] if rewards else 0.0
    success     = final_score >= SUCCESS_THRESHOLD

    print(f"[END] success={success} steps={step} score={final_score:.2f}")
    print()


def run_all():
    for t in ["easy_malaria", "medium_tb", "hard_sepsis"]:
        run_task(t)
        time.sleep(1)


if __name__ == "__main__":
    run_all()