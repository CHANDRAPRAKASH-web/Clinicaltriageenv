import os
import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Hackathon-mandated variable names ---
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

MAX_STEPS         = 12
SUCCESS_THRESHOLD = 0.5
SERVER_URL        = "http://127.0.0.1:7860"

if not API_KEY:
    print("⚠️ API_KEY not set")

print(f"📡 API: {API_BASE_URL}")
print(f"🤖 Model: {MODEL_NAME}")

# --- Module-level OpenAI client using injected env vars ---
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


# -----------------------------------------------
# LLM CALL — every action goes through here
# -----------------------------------------------
def call_llm(observation: dict) -> dict:
    system_prompt = """You are a rural health assistant AI helping a community health worker diagnose patients.

At each step, based on the observation, decide the next clinical action.

Return ONLY a valid JSON object — no explanation, no markdown, no extra text.

Use exactly one of these formats:

1. Ask the patient a question:
   {"action": "ask_patient", "question": "..."}

2. Request a vital sign:
   {"action": "request_vital", "vital": "temperature"}
   {"action": "request_vital", "vital": "heart rate"}
   {"action": "request_vital", "vital": "spo2"}

3. Request a diagnostic test:
   {"action": "request_test", "test": "RDT"}
   {"action": "request_test", "test": "sputum test"}
   {"action": "request_test", "test": "blood culture"}
   {"action": "request_test", "test": "random blood sugar"}

4. Make a final assessment (only after gathering enough info):
   {"action": "make_assessment", "risk": "HIGH", "condition": "plasmodium_vivax_malaria", "next_step": "refer_to_PHC"}
   {"action": "make_assessment", "risk": "CRITICAL", "condition": "active_pulmonary_TB", "next_step": "refer_to_district_TB_centre"}
   {"action": "make_assessment", "risk": "CRITICAL", "condition": "diabetic_foot_sepsis", "next_step": "immediate_hospitalization"}

Clinical guidance:
- FEVER → ask about duration, chills, mosquito exposure, body ache, appetite/urine change, medications → check temperature + heart rate → RDT test → assess for malaria
- COUGH → ask about blood in sputum, duration (>2 weeks), weight loss, night sweats, TB contact, breathlessness → check temperature + spo2 → sputum test → assess for TB
- WOUND / DIZZY / CONFUSION → ask about confusion, wound duration/smell, diabetes, fever, dizziness → check temperature → blood culture + random blood sugar → assess for sepsis

Follow the progression: ask questions first, then request vitals, then request tests, then make assessment.
Steps taken so far is shown in the observation — use it to decide what stage you are at."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(observation)}
            ],
            temperature=0.1,
            max_tokens=200
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}")
        return {"action": "ask_patient",
                "question": "Can you describe your main symptoms clearly?"}


# -----------------------------------------------
# TASK RUNNER
# -----------------------------------------------
def run_task(task_name: str):

    reset_resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name})
    data = reset_resp.json()

    session_id  = data["session_id"]
    observation = data["observation"]

    print(f"[START] Task: {task_name}")

    rewards = []

    for step in range(MAX_STEPS):

        action = call_llm(observation)

        step_resp = requests.post(f"{SERVER_URL}/step", json={
            "session_id": session_id,
            "action": action
        })

        step_data   = step_resp.json()
        observation = step_data["observation"]
        reward      = step_data["reward"]
        done        = step_data["done"]

        rewards.append(reward)

        print(f"[STEP] step={step+1} action={action['action']} reward={reward:.2f} done={done} error=null")

        if done:
            break

    final_score = rewards[-1] if rewards else 0.0
    success     = final_score >= SUCCESS_THRESHOLD

    print(f"[END] success={success} steps={step+1} score={final_score:.2f}")


def run_all():
    for t in ["easy_malaria", "medium_tb", "hard_sepsis"]:
        run_task(t)
        time.sleep(1)


if __name__ == "__main__":
    run_all()