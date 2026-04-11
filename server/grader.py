from typing import Any, Dict, List

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

# Common medical word stems for fuzzy matching
STEM_PAIRS: List[tuple] = [
    ("confus", "confus"),        # confused / confusion / confusing
    ("fever", "febr"),
    ("cough", "cough"),
    ("bleed", "blood"),
    ("hemopt", "blood"),         # hemoptysis → blood
    ("sputum", "sputum"),
    ("wound", "wound"),
    ("infect", "infect"),
    ("sepsis", "septic"),
    ("dizz", "dizz"),
    ("weak", "weak"),
    ("fatigue", "tired"),
    ("mosquito", "insect"),
    ("malaria", "plasmodium"),
    ("tb", "tuberculosis"),
    ("hospital", "refer"),
    ("ambulance", "emergency"),
    ("sugar", "glucose"),
    ("diabet", "diabet"),
]


# =========================
# UTILITIES
# =========================

def normalize(text: str) -> str:
    return (text or "").lower().replace("_", " ").strip()


def stem_match(a: str, b: str) -> bool:
    """Check if two strings share a common medical stem."""
    for s1, s2 in STEM_PAIRS:
        if (s1 in a and s2 in b) or (s2 in a and s1 in b):
            return True
        if s1 in a and s1 in b:
            return True
        if s2 in a and s2 in b:
            return True
    return False


def partial_match(a: str, b: str) -> bool:
    a, b = normalize(a), normalize(b)

    if not a or not b:
        return False

    # Direct substring match
    if a in b or b in a:
        return True

    # Stem-based matching
    if stem_match(a, b):
        return True

    # Word-level match (words longer than 4 chars)
    a_words = [w for w in a.split() if len(w) > 4]
    b_words = [w for w in b.split() if len(w) > 4]

    if any(w in b for w in a_words) or any(w in a for w in b_words):
        return True

    # Word stem match across word lists
    for aw in a_words:
        for bw in b_words:
            if stem_match(aw, bw):
                return True

    return False


def keyword_match(keywords: List[str], text: str) -> bool:
    text = normalize(text)
    return any(k in text for k in keywords)


def mentioned(history: List[str], keyword: str) -> bool:
    keyword = normalize(keyword)
    return any(keyword in normalize(h) for h in history)


def mentioned_stem(history: List[str], keyword: str) -> bool:
    """Check if keyword OR its stem appears in history."""
    if mentioned(history, keyword):
        return True
    # Try stem match against each history entry
    for h in history:
        if stem_match(normalize(keyword), normalize(h)):
            return True
    return False


# =========================
# INTERMEDIATE REWARD
# =========================

def intermediate_reward(action: Dict[str, Any],
                        task_data: Dict[str, Any],
                        state: Dict[str, Any]) -> float:

    action_type = action.get("action", "")
    red_flags = task_data.get("red_flags", [])
    optimal = task_data.get("optimal_tests", []) + task_data.get("optimal_vitals", [])

    # ---- ASK ----
    if action_type == "ask_patient":
        q = normalize(action.get("question", ""))

        if not q:
            return 0.0

        # Red flag catch — highest reward
        if any(partial_match(flag, q) for flag in red_flags):
            return 0.25

        # General WHO triage keyword present
        if keyword_match(WHO_TRIAGE_KEYWORDS, q):
            return 0.08

        return 0.02

    # ---- TEST / VITAL ----
    if action_type in ["request_test", "request_vital"]:
        query = normalize(action.get("test") or action.get("vital") or "")

        if not query:
            return 0.0

        # Optimal test/vital ordered
        if any(partial_match(query, opt) for opt in optimal):
            return 0.32

        if action_type == "request_vital":
            if keyword_match(["temperature", "bp", "heart rate", "spo2", "pulse"], query):
                return 0.12
            return 0.06

        if action_type == "request_test":
            if keyword_match(["cbc", "blood", "xray", "culture", "glucose", "sputum", "rdt"], query):
                return 0.07
            return -0.04

    return 0.01


# =========================
# FINAL SCORE
# =========================

def final_score(action: Dict[str, Any],
                task_data: Dict[str, Any],
                state: Dict[str, Any]) -> float:

    gt = task_data.get("ground_truth", {})

    gt_risk  = normalize(gt.get("risk", "")).upper()
    gt_cond  = normalize(gt.get("condition", ""))
    gt_step  = normalize(gt.get("next_step", ""))

    ar = normalize(action.get("risk", "")).upper()
    ac = normalize(action.get("condition", ""))
    an = normalize(action.get("next_step", ""))

    steps     = state.get("steps_taken", 0)
    max_steps = task_data.get("max_steps", 8)
    history   = state.get("conversation_history", [])

    # ---- COMPONENTS ----
    risk = 0.40 if ar == gt_risk else (
        0.30 if ar in ["HIGH", "CRITICAL"] and gt_risk in ["HIGH", "CRITICAL"] else 0.0
    )

    condition = 0.25 if partial_match(ac, gt_cond) else 0.0
    next_step = 0.20 if partial_match(an, gt_step) else 0.0

    efficiency = 0.15 * ((max_steps - steps) / max_steps) if steps < max_steps else 0.0

    score = risk + condition + next_step + efficiency

    # ---- PENALTIES ----

    # Critical misclassification — hard cap
    if gt_risk == "CRITICAL" and ar in ["LOW", "MEDIUM"]:
        score = min(score, 0.20)

    # Dangerous recommendation for emergency condition
    if an == "home_rest":
        combined = gt_cond + " " + task_data.get("task_id", "")
        if keyword_match(EMERGENCY_CONDITIONS, combined):
            return 0.0

    # TB: must have asked about blood/sputum/hemoptysis
    if "tb" in gt_cond or "tuberculosis" in gt_cond:
        if not any(mentioned_stem(history, k) for k in ["blood", "sputum", "hemoptysis", "hemopt"]):
            score -= 0.30

    # Hidden red flag must have been uncovered
    hidden = task_data.get("hidden_flag")
    if hidden and not mentioned_stem(history, hidden):
        score = min(score, 0.20)

    return max(0.0, min(score, 1.0))


# =========================
# GRADER CLASS
# =========================

class Grader:

    def grade(self, assessment: Dict[str, Any]) -> Dict[str, Any]:

        action    = assessment.get("action", {})
        task_data = assessment.get("task_data", {})
        state     = assessment.get("state", {})

        if assessment.get("type") == "final":
            score = final_score(action, task_data, state)
            return {"score": score, "type": "final"}

        score = intermediate_reward(action, task_data, state)
        return {"score": score, "type": "intermediate"}