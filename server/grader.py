from typing import Any, Dict, List

# =========================
# SAFE SCORE (CRITICAL)
# =========================

def safe_score(score: float) -> float:
    if score >= 1.0:
        return 0.99
    if score <= 0.0:
        return 0.01
    return score


# =========================
# CONSTANTS
# =========================

WHO_TRIAGE_KEYWORDS: List[str] = [
    "fever","cough","vomiting","diarrhea","breathing","chest pain",
    "headache","seizure","unconscious","lethargy","pain","wound",
    "medication","diabetes","tb","tuberculosis","malaria","mosquito",
    "travel","contact","exposure","weight","nutrition","night sweats",
    "chills","confusion","confused","mental status","temperature","pulse",
    "heart rate","spo2","sugar","glucose","discharge","smell","blood",
    "sputum","hemoptysis","weakness","fatigue","dizzy","dizziness"
]

EMERGENCY_CONDITIONS: List[str] = [
    "tb","tuberculosis","sepsis","meningitis","shock","stroke",
    "cardiac","diabetic_foot","hemorrhage","diabetic_sepsis"
]


# =========================
# UTILS
# =========================

def normalize(text: str) -> str:
    return (text or "").lower().replace("_", " ").strip()


def partial_match(a: str, b: str) -> bool:
    a, b = normalize(a), normalize(b)
    return a in b or b in a


def keyword_match(keywords: List[str], text: str) -> bool:
    text = normalize(text)
    return any(k in text for k in keywords)


def mentioned(history: List[str], keyword: str) -> bool:
    keyword = normalize(keyword)
    return any(keyword in normalize(h) for h in history)


# =========================
# INTERMEDIATE REWARD
# =========================

def intermediate_reward(action: Dict[str, Any],
                        task_data: Dict[str, Any],
                        state: Dict[str, Any]) -> float:

    action_type = action.get("action", "")
    red_flags = task_data.get("red_flags", [])
    optimal = task_data.get("optimal_tests", [])

    if action_type == "ask_patient":
        q = normalize(action.get("question", ""))

        if not q:
            return safe_score(0.02)

        if any(partial_match(flag, q) for flag in red_flags):
            return safe_score(0.25)

        if keyword_match(WHO_TRIAGE_KEYWORDS, q):
            return safe_score(0.08)

        return safe_score(0.02)

    if action_type == "request_test":
        query = normalize(action.get("test", ""))

        if any(partial_match(query, opt) for opt in optimal):
            return safe_score(0.32)

        return safe_score(0.08)

    return safe_score(0.02)


# =========================
# FINAL SCORE
# =========================

def final_score(action: Dict[str, Any],
                task_data: Dict[str, Any],
                state: Dict[str, Any]) -> float:

    gt = task_data.get("ground_truth", {})

    gt_risk = normalize(gt.get("risk", "")).upper()
    gt_cond = normalize(gt.get("condition", ""))
    gt_step = normalize(gt.get("next_step", ""))

    ar = normalize(action.get("risk", "")).upper()
    ac = normalize(action.get("condition", ""))
    an = normalize(action.get("next_step", ""))

    steps = state.get("steps_taken", 0)
    max_steps = task_data.get("max_steps", 8)

    # ---- scoring ----
    risk = 0.4 if ar == gt_risk else 0.2
    condition = 0.25 if partial_match(ac, gt_cond) else 0.1
    next_step = 0.2 if partial_match(an, gt_step) else 0.1
    efficiency = 0.15 * ((max_steps - steps) / max_steps)

    score = risk + condition + next_step + efficiency

    # ---- penalties ----
    if gt_risk == "CRITICAL" and ar in ["LOW", "MEDIUM"]:
        score = min(score, 0.3)

    # 🚨 FIXED: NEVER return 0 directly
    if an == "home_rest":
        combined = gt_cond + " " + task_data.get("task_id", "")
        if keyword_match(EMERGENCY_CONDITIONS, combined):
            return safe_score(0.01)

    return safe_score(score)


# =========================
# GRADER CLASS
# =========================

class Grader:

    def grade(self, assessment: Dict[str, Any]) -> Dict[str, Any]:

        action = assessment.get("action", {})
        task_data = assessment.get("task_data", {})
        state = assessment.get("state", {})

        if assessment.get("type") == "final":
            score = final_score(action, task_data, state)
            return {"score": safe_score(score), "type": "final"}

        score = intermediate_reward(action, task_data, state)
        return {"score": safe_score(score), "type": "intermediate"}