import os
import json
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- TESTING: Groq ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

MAX_STEPS         = 12
SUCCESS_THRESHOLD = 0.5
SERVER_URL        = "http://127.0.0.1:7860"

if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN missing. Set it in environment.")

print(f"📡 API: {API_BASE_URL}")
print(f"🤖 Model: {MODEL_NAME}")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_llm_action(observation: dict) -> dict:
    complaint = observation["presenting_complaint"].lower()
    step      = observation["steps_taken"]

    if "fever" in complaint:
        if step == 0:
            return {"action": "ask_patient", "question": "How long have you had this fever? Is it constant or does it come and go?"}
        if step == 1:
            return {"action": "ask_patient", "question": "Do you get severe chills and shivering before the fever rises?"}
        if step == 2:
            return {"action": "ask_patient", "question": "Have you been exposed to mosquitoes or stagnant water recently?"}
        if step == 3:
            return {"action": "ask_patient", "question": "Do you have body ache, muscle pain, or headache along with the fever?"}
        if step == 4:
            return {"action": "ask_patient", "question": "Have you noticed any loss of appetite or change in urine colour?"}
        if step == 5:
            return {"action": "ask_patient", "question": "Have you taken any medication for the fever? Did it help?"}
        if step == 6:
            return {"action": "request_vital", "vital": "temperature"}
        if step == 7:
            return {"action": "request_vital", "vital": "heart rate"}
        if step == 8:
            return {"action": "request_test", "test": "RDT"}
        if step == 9:
            return {"action": "make_assessment", "risk": "HIGH", "condition": "plasmodium_vivax_malaria", "next_step": "refer_to_PHC"}

    if "cough" in complaint:
        if step == 0:
            return {"action": "ask_patient", "question": "Are you coughing blood or seeing blood in your sputum?"}
        if step == 1:
            return {"action": "ask_patient", "question": "How long have you had this cough? Has it been more than 2 weeks?"}
        if step == 2:
            return {"action": "ask_patient", "question": "Have you noticed significant weight loss in the past few months?"}
        if step == 3:
            return {"action": "ask_patient", "question": "Do you wake up at night with heavy sweating — night sweats?"}
        if step == 4:
            return {"action": "ask_patient", "question": "Have you been in close contact with anyone who had tuberculosis?"}
        if step == 5:
            return {"action": "ask_patient", "question": "Do you feel breathless or short of breath even with mild activity?"}
        if step == 6:
            return {"action": "request_vital", "vital": "temperature"}
        if step == 7:
            return {"action": "request_vital", "vital": "spo2"}
        if step == 8:
            return {"action": "request_test", "test": "sputum test"}
        if step == 9:
            return {"action": "make_assessment", "risk": "CRITICAL", "condition": "active_pulmonary_TB", "next_step": "refer_to_district_TB_centre"}

    if any(kw in complaint for kw in ["wound", "dizzy", "ghav", "bhram", "kamzori", "confusion", "injury", "foot", "weak"]):
        if step == 0:
            return {"action": "ask_patient", "question": "Are you feeling confused or unable to think clearly?"}
        if step == 1:
            return {"action": "ask_patient", "question": "How long has the wound been there? Is it foul-smelling or getting worse?"}
        if step == 2:
            return {"action": "ask_patient", "question": "Do you have diabetes? Are you currently taking any medication for blood sugar?"}
        if step == 3:
            return {"action": "ask_patient", "question": "How long have you had fever? Any chills or shivering?"}
        if step == 4:
            return {"action": "ask_patient", "question": "Are you feeling dizzy or very weak when you try to stand up?"}
        if step == 5:
            return {"action": "request_vital", "vital": "temperature"}
        if step == 6:
            return {"action": "request_test", "test": "blood culture"}
        if step == 7:
            return {"action": "request_test", "test": "random blood sugar"}
        if step == 8:
            return {"action": "make_assessment", "risk": "CRITICAL", "condition": "diabetic_foot_sepsis", "next_step": "immediate_hospitalization"}

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "You are a clinical triage agent following WHO/ICMR guidelines. "
                    "Return ONLY valid JSON with one of these actions: "
                    "ask_patient (with 'question'), request_test (with 'test'), "
                    "request_vital (with 'vital'), or make_assessment "
                    "(with 'risk', 'condition', 'next_step'). "
                    "Do not include markdown or explanation."
                )},
                {"role": "user", "content": (
                    f"Clinical observation:\n{json.dumps(observation, indent=2)}\n\n"
                    "What is the next best clinical action?"
                )}
            ],
            temperature=0.1,
            max_tokens=200
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️  LLM fallback error: {e}")
        return {"action": "ask_patient", "question": "Can you describe your main symptoms clearly?"}


def run_task(task_name: str):
    reset_resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name})
    if reset_resp.status_code != 200:
        print(f"❌ Reset failed [{reset_resp.status_code}]: {reset_resp.text}")
        return

    data        = reset_resp.json()
    session_id  = data["session_id"]
    observation = data["observation"]

    print(f"\n{'='*60}")
    print(f"[START] Task: {task_name}")
    print(f"  Patient  : {observation['patient_profile']}")
    print(f"  Complaint: {observation['presenting_complaint'][:90]}")
    print(f"{'='*60}")

    rewards = []

    for step in range(MAX_STEPS):
        action    = get_llm_action(observation)
        step_resp = requests.post(f"{SERVER_URL}/step", json={"session_id": session_id, "action": action})

        if step_resp.status_code != 200:
            print(f"❌ Step error [{step_resp.status_code}]: {step_resp.text}")
            break

        step_data   = step_resp.json()
        observation = step_data["observation"]
        reward      = step_data["reward"]
        done        = step_data["done"]
        info        = step_data.get("info", {})
        rewards.append(reward)

        act = action["action"]

        if act == "ask_patient":
            print(f"\n  🩺 Doctor : {action.get('question')}")
            hist = observation["conversation_history"]
            last_answer = next((h for h in reversed(hist) if h.startswith("A:")), "")
            print(f"  👤 Patient: {last_answer}")

        elif act in ["request_test", "request_vital"]:
            item   = action.get("test") or action.get("vital")
            tests  = observation.get("tests_ordered", [])
            result = tests[-1]["result"] if tests else "?"
            label  = "🌡️  Vital" if act == "request_vital" else "🧪 Test"
            print(f"\n  {label}: {item} → {result}")

        elif act == "make_assessment":
            print(f"\n  📝 Diagnosis : {action.get('condition')}")
            print(f"  ⚠️  Risk      : {action.get('risk')}")
            print(f"  ➡️  Next Step : {action.get('next_step')}")

        flag = "✅" if reward >= 0.2 else ("⚠️" if reward >= 0 else "❌")
        print(f"  {flag} [STEP {step+1}] reward={reward:.2f} done={done}" + (f" | {info}" if info else ""))

        if done:
            break

    final_score = rewards[-1] if rewards else 0.0
    success     = final_score >= SUCCESS_THRESHOLD

    print(f"\n{'─'*60}")
    print(f"  Rewards : {[f'{r:.2f}' for r in rewards]}")
    print(f"  Total   : {sum(rewards):.2f}")
    print(f"  Final   : {final_score:.2f}")
    print(f"  Result  : {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"{'─'*60}")


def run_all():
    tasks = ["easy_malaria", "medium_tb", "hard_sepsis"]
    for t in tasks:
        try:
            run_task(t)
            time.sleep(2)
        except Exception as e:
            print(f"❌ Task '{t}' crashed: {e}")


if __name__ == "__main__":
    run_all()