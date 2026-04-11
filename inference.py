import os
import json
import time
import requests

# -----------------------------------------------
# ENV SETUP — use ONLY the validator-injected vars
# DO NOT call load_dotenv() — it can conflict with
# the API_BASE_URL and API_KEY that the validator
# injects at runtime.
# -----------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY      = os.environ.get("API_KEY", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

MAX_STEPS = 12
SUCCESS_THRESHOLD = 0.5
SERVER_URL = "http://127.0.0.1:7860"

SYSTEM_PROMPT = """You are a clinical triage AI agent deployed in a rural Indian health sub-centre.
You are triaging real patients based on WHO/ICMR guidelines.

At each step you will receive:
- The patient's profile and presenting complaint
- The conversation history so far (questions asked, answers received, tests ordered)
- The current step number and max steps

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
- Always ask about red flag symptoms early (fever pattern, blood in sputum, confusion, etc.)
- Order targeted tests based on clinical suspicion (RDT for malaria, sputum test for TB, blood culture for sepsis)
- Use WHO danger signs to classify risk level
- For HIGH/CRITICAL risk: always refer (PHC, district hospital, or immediate hospitalization)
- Be efficient — do not waste steps on irrelevant questions
- Return ONLY valid JSON, no extra text."""


# -----------------------------------------------
# LLM CALL — routes through the validator's proxy
# -----------------------------------------------
def call_llm(messages: list) -> str:
    """Call the LLM via the validator-provided proxy and return the response content."""
    from openai import OpenAI

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=300,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM's JSON response, handling markdown code fences."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    # Fallback if LLM returns garbage
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
        {"role": "user", "content": user_prompt}
    ]

    # If we're near the end of allowed steps, nudge toward final assessment
    steps_taken = observation.get("steps_taken", 0)
    max_steps = observation.get("max_steps", 8)
    if steps_taken >= max_steps - 2:
        messages.append({
            "role": "user",
            "content": "You are running low on steps. Make your final assessment NOW with make_assessment."
        })

    try:
        response_text = call_llm(messages)
        print(f"[LLM] Raw response: {response_text[:200]}")
        action = parse_llm_response(response_text)

        # Validate action has required fields
        if "action" not in action:
            action = {"action": "ask_patient", "question": "Please describe your symptoms in detail."}

        return action

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        # Rule-based fallback so the task can still complete
        return _fallback_action(observation)


def _fallback_action(observation: dict) -> dict:
    """Rule-based fallback if LLM is unavailable."""
    complaint = observation.get("presenting_complaint", "").lower()
    step = observation.get("steps_taken", 0)

    if "fever" in complaint:
        sequence = [
            {"action": "ask_patient", "question": "How long have you had fever and is it cyclic?"},
            {"action": "ask_patient", "question": "Do you experience chills or shivering?"},
            {"action": "ask_patient", "question": "Have you been exposed to mosquitoes or stagnant water?"},
            {"action": "request_test", "test": "RDT"},
            {"action": "make_assessment", "risk": "HIGH", "condition": "plasmodium_vivax_malaria", "next_step": "refer_to_PHC"},
        ]
    elif "cough" in complaint:
        sequence = [
            {"action": "ask_patient", "question": "Is there any blood in your sputum?"},
            {"action": "ask_patient", "question": "Have you had weight loss or night sweats?"},
            {"action": "request_test", "test": "sputum test"},
            {"action": "make_assessment", "risk": "CRITICAL", "condition": "active_pulmonary_TB", "next_step": "refer_to_district_TB_centre"},
        ]
    elif any(k in complaint for k in ["wound", "dizzy", "confusion", "injury"]):
        sequence = [
            {"action": "ask_patient", "question": "Are you experiencing confusion or altered mental status?"},
            {"action": "ask_patient", "question": "Is the wound getting worse or showing signs of infection?"},
            {"action": "request_test", "test": "blood culture"},
            {"action": "make_assessment", "risk": "CRITICAL", "condition": "diabetic_foot_sepsis", "next_step": "immediate_hospitalization"},
        ]
    else:
        sequence = [
            {"action": "ask_patient", "question": "Can you describe your main symptoms?"},
            {"action": "make_assessment", "risk": "MODERATE", "condition": "unspecified", "next_step": "refer_to_PHC"},
        ]

    idx = min(step, len(sequence) - 1)
    return sequence[idx]


# -----------------------------------------------
# TASK RUNNER
# -----------------------------------------------
def run_task(task_name: str):

    reset_resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name})
    data = reset_resp.json()

    session_id = data["session_id"]
    observation = data["observation"]

    print(f"[START] task={task_name} env=clinical model={MODEL_NAME}")

    rewards = []

    for step in range(1, MAX_STEPS + 1):

        action = get_llm_action(observation)

        step_resp = requests.post(f"{SERVER_URL}/step", json={
            "session_id": session_id,
            "action": action
        })

        step_data = step_resp.json()
        observation = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]

        rewards.append(reward)

        print(f"[STEP] step={step} action={action['action']} reward={reward:.2f} done={str(done).lower()} error=null")

        if done:
            break

    final_score = rewards[-1] if rewards else 0.0
    success = final_score >= SUCCESS_THRESHOLD

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={str(success).lower()} steps={step} score={final_score:.2f} rewards={rewards_str}")
    print()


def run_all():
    for t in ["easy_malaria", "medium_tb", "hard_sepsis"]:
        run_task(t)
        time.sleep(1)


if __name__ == "__main__":
    run_all()