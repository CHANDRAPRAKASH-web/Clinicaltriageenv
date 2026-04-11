import os
import json
import time
import requests
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# --- Hackathon-mandated variable names — NO fallback defaults ---
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

MAX_STEPS         = 12
SUCCESS_THRESHOLD = 0.5
SERVER_URL        = "http://127.0.0.1:7860"
BENCHMARK         = "clinical_triage"

print(f"📡 API: {API_BASE_URL}", flush=True)
print(f"🤖 Model: {MODEL_NAME}", flush=True)


# -----------------------------------------------
# STDOUT LOG HELPERS (mandatory exact format)
# -----------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# -----------------------------------------------
# LLM CALL — client passed in, never created at module level
# -----------------------------------------------
def call_llm(client, observation: dict) -> dict:
    system_prompt = """You are a rural health assistant AI helping a community health worker diagnose patients.

At each step, based on the observation, decide the next clinical action.

Return ONLY a valid JSON object — no explanation, no markdown, no extra text.

Use exactly one of these formats:

1. {"action": "ask_patient", "question": "..."}
2. {"action": "request_vital", "vital": "temperature"}
   {"action": "request_vital", "vital": "heart rate"}
   {"action": "request_vital", "vital": "spo2"}
3. {"action": "request_test", "test": "RDT"}
   {"action": "request_test", "test": "sputum test"}
   {"action": "request_test", "test": "blood culture"}
   {"action": "request_test", "test": "random blood sugar"}
4. {"action": "make_assessment", "risk": "HIGH", "condition": "plasmodium_vivax_malaria", "next_step": "refer_to_PHC"}
   {"action": "make_assessment", "risk": "CRITICAL", "condition": "active_pulmonary_TB", "next_step": "refer_to_district_TB_centre"}
   {"action": "make_assessment", "risk": "CRITICAL", "condition": "diabetic_foot_sepsis", "next_step": "immediate_hospitalization"}

Clinical guidance:
- FEVER → ask about duration, chills, mosquito exposure, body ache, appetite/urine → check temperature + heart rate → RDT → assess malaria
- COUGH → ask about blood in sputum, duration >2 weeks, weight loss, night sweats, TB contact, breathlessness → check temperature + spo2 → sputum test → assess TB
- WOUND/DIZZY/CONFUSION → ask about confusion, wound smell/duration, diabetes, fever, dizziness → check temperature → blood culture + random blood sugar → assess sepsis

Follow the progression: questions first → vitals → tests → final assessment."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": json.dumps(observation)}
            ],
            temperature=0.1,
            max_tokens=200
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return {"action": "ask_patient",
                "question": "Can you describe your main symptoms clearly?"}


# -----------------------------------------------
# TASK RUNNER
# -----------------------------------------------
def run_task(client, task_name: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name})
        reset_resp.raise_for_status()
        data = reset_resp.json()

        session_id  = data["session_id"]
        observation = data["observation"]

        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
            error = None

            action = call_llm(client, observation)

            try:
                step_resp = requests.post(f"{SERVER_URL}/step", json={
                    "session_id": session_id,
                    "action": action
                })
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                error = str(e)
                print(f"[DEBUG] Step request failed: {e}", flush=True)
                log_step(step=step, action=action["action"], reward=0.0, done=True, error=error)
                break

            observation = step_data["observation"]
            reward      = float(step_data["reward"])
            done        = bool(step_data["done"])

            rewards.append(reward)
            log_step(step=step, action=action["action"], reward=reward, done=done, error=error)

            if done:
                break

        score   = rewards[-1] if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# -----------------------------------------------
# MAIN — client created ONCE here using injected vars
# -----------------------------------------------
def main() -> None:
    from openai import OpenAI

    # Single client init — uses their injected API_BASE_URL and API_KEY
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    for task_name in ["easy_malaria", "medium_tb", "hard_sepsis"]:
        run_task(client, task_name)
        time.sleep(1)


if __name__ == "__main__":
    main()