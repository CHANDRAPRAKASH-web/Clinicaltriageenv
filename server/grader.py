"""
Grader for ClinicalTriageEnv — Phase 3
Pure rule-based scoring using keyword matching (lowercase, partial match).
No ML involved.

Two core functions:
  - intermediate_reward: per-step shaping reward
  - final_score: end-of-episode composite score with safety penalties
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# WHO / IMNCI clinically relevant question keywords
# These represent standard triage assessment topics a competent health worker
# should ask about.  Used for the +0.05 "clinically relevant question" bonus.
# ---------------------------------------------------------------------------
WHO_TRIAGE_KEYWORDS: List[str] = [
    "fever", "cough", "diarrhea", "diarrhoea", "vomiting", "rash",
    "breathing", "breathless", "shortness of breath", "chest pain",
    "headache", "convulsion", "seizure", "unconscious", "lethargy",
    "dehydration", "drinking", "eating", "appetite", "breastfeed",
    "urination", "urine", "stool", "blood", "bleeding", "swelling",
    "wound", "injury", "pain", "medication", "medicine", "drug",
    "allergy", "diabetes", "hypertension", "bp", "blood pressure",
    "hiv", "tb", "tuberculosis", "malaria", "dengue", "typhoid",
    "pregnancy", "vaccination", "immunization", "weight", "nutrition",
    "mosquito", "travel", "contact", "exposure", "family history",
    "smoking", "alcohol", "tobacco", "night sweats", "chills",
    "confusion", "mental status", "consciousness", "orientation",
    "temperature", "pulse", "heart rate", "respiratory rate",
    "oxygen", "spo2", "sugar", "glucose",
]

# Emergency / critical conditions for the home_rest safety check
EMERGENCY_CONDITIONS: List[str] = [
    "tb", "tuberculosis", "sepsis", "meningitis", "emergency",
    "critical", "diabetic_foot", "hemorrhage", "haemorrhage",
    "shock", "stroke", "cardiac",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ═══════════════════════════════════════════════════════════════════════════

def _partial_match(val1: str, val2: str) -> bool:
    """Return True if any significant word overlaps between agent and ground truth."""
    v1 = val1.lower().replace("_", " ")
    v2 = val2.lower().replace("_", " ")
    if v1 in v2 or v2 in v1: return True
    
    # Specific medical generous matching
    generous_pairs = [
        ("septicemia", "sepsis"), ("infection", "sepsis"), ("bacteremia", "sepsis"),
        ("phc", "specialist"), ("refer", "consult"), ("hospital", "refer"), 
        ("tuberculosis", "tb"), ("pulmonary", "tb"), ("plasmodium", "malaria")
    ]
    for w1, w2 in generous_pairs:
        if (w1 in v1 and w2 in v2) or (w2 in v1 and w1 in v2):
            return True
            
    # Word-level overlap for words longer than 4 chars
    v1_words = [w for w in v1.split() if len(w) > 4]
    v2_words = [w for w in v2.split() if len(w) > 4]
    return any(w in v2 for w in v1_words) or any(w in v1 for w in v2_words)


def _any_keyword_in_text(keywords: List[str], text: str) -> bool:
    """Return True if any keyword partially matches text."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _conversation_mentions(conversation_history: List[str], keyword: str) -> bool:
    """Check whether a keyword has been mentioned across all Q/A turns."""
    kw = keyword.lower()
    return any(kw in entry.lower() for entry in conversation_history)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  intermediate_reward
# ═══════════════════════════════════════════════════════════════════════════

def intermediate_reward(action: Dict[str, Any],
                        task_data: Dict[str, Any],
                        state: Dict[str, Any]) -> float:
    """Return a small shaping reward for a single non-terminal step.

    Rules
    -----
    - ask_patient with question matching a red_flag keyword   → +0.15
    - ask_patient with a clinically relevant question (WHO)   → +0.05
    - request_test where test ∈ optimal_tests                 → +0.10
    - request_test where test ∉ optimal_tests (redundant)     → −0.05
    - anything else                                           →  0.00
    """
    action_type = action.get("action", "")
    red_flags: List[str] = task_data.get("red_flags", [])
    optimal_tests: List[str] = task_data.get("optimal_tests", [])

    # ── ask_patient ──────────────────────────────────────────────────────
    if action_type == "ask_patient":
        question = (action.get("question") or "").lower()
        if not question:
            return 0.0

        # Check red-flag match first (higher reward takes precedence)
        for flag in red_flags:
            if _partial_match(flag, question):
                return 0.25

        # Check WHO clinically relevant question
        if _any_keyword_in_text(WHO_TRIAGE_KEYWORDS, question):
            return 0.08

        return 0.02 # Tiny default reward for engaging the patient

    # ── request_vital / request_test ─────────────────────────────────────
    if action_type in ["request_test", "request_vital"]:
        # Allow cross-naming (LLM often puts test in vital field or vice versa)
        query = (action.get("test") or action.get("vital") or "").lower()
        if not query:
            return 0.0

        # High reward for optimal tests/vitals mentioned in task data
        all_optimal = optimal_tests + task_data.get("optimal_vitals", [])
        for opt in all_optimal:
            if _partial_match(opt, query) or _partial_match(query, opt):
                return 0.32

        # Standard bonus for basic medical checks (encourages thoroughness)
        BASIC_CHECKS = ["temperature", "bp", "blood pressure", "heart rate", "pulse", "spo2", "breathing"]
        if action_type == "request_vital":
            if _any_keyword_in_text(BASIC_CHECKS, query):
                return 0.12
            return 0.06

        # Small penalty for totally redundant tests, but partial credit for standard realistic labs
        if action_type == "request_test":
            standard_labs = ["blood", "cbc", "sputum", "xray", "x-ray", "culture", "sugar", "glucose", "smear", "swab", "urine"]
            if _any_keyword_in_text(standard_labs, query):
                return 0.07
            return -0.04

    # ── any other action ─────────────────────────────────────────────────
    return 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  2.  final_score
# ═══════════════════════════════════════════════════════════════════════════

def final_score(action: Dict[str, Any],
                task_data: Dict[str, Any],
                state: Dict[str, Any]) -> float:
    """Compute the terminal episode score ∈ [0.0, 1.0].

    Components (weighted sum)
    -------------------------
    - Correct risk classification   : 0.40
    - Correct condition identified  : 0.25
    - Correct next_step recommended : 0.20
    - Efficiency bonus              : 0.15

    Safety penalties (applied *after* weighted sum)
    ------------------------------------------------
    - CRITICAL patient classified LOW/MEDIUM   → cap at 0.20
    - next_step == "home_rest" for emergency   → force 0.0
    - blood in sputum not flagged in TB task   → −0.30
    - hidden_flag not identified (hard task)   → cap at 0.20
    """
    ground_truth: Dict[str, str] = task_data.get("ground_truth", {})
    gt_risk      = ground_truth.get("risk", "").upper()
    gt_condition = ground_truth.get("condition", "").lower()
    gt_next_step = ground_truth.get("next_step", "").lower()

    agent_risk      = (action.get("risk") or "").upper()
    agent_condition = (action.get("condition") or "").lower()
    agent_next_step = (action.get("next_step") or "").lower()

    steps_taken: int = state.get("steps_taken", 0)
    max_steps: int   = task_data.get("max_steps", 8)
    conversation_history: List[str] = state.get("conversation_history", [])

    # ── Component scores ─────────────────────────────────────────────────
    # 1) Risk classification (exact match)
    risk_score = 0.40 if agent_risk == gt_risk else 0.0
    # Give partial credit (0.30) if they say HIGH but it's CRITICAL, or vice-versa
    if not risk_score and agent_risk in ("HIGH", "CRITICAL") and gt_risk in ("HIGH", "CRITICAL"):
        risk_score = 0.30

    # 2) Condition identified (partial match)
    condition_score = 0.0
    if agent_condition and gt_condition:
        if _partial_match(agent_condition, gt_condition):
            condition_score = 0.25

    # 3) Next-step recommended (partial match)
    next_step_score = 0.0
    if agent_next_step and gt_next_step:
        if _partial_match(agent_next_step, gt_next_step):
            next_step_score = 0.20

    # 4) Efficiency bonus: proportional to steps saved
    if max_steps > 0 and steps_taken < max_steps:
        efficiency_score = 0.15 * ((max_steps - steps_taken) / max_steps)
    else:
        efficiency_score = 0.0

    # ── Weighted composite ───────────────────────────────────────────────
    score = risk_score + condition_score + next_step_score + efficiency_score

    # ── Safety penalties ─────────────────────────────────────────────────

    # P1: CRITICAL patient classified as LOW or MEDIUM → cap at 0.20
    if gt_risk == "CRITICAL" and agent_risk in ("LOW", "MEDIUM"):
        score = min(score, 0.20)

    # P2: next_step is "home_rest" for TB / sepsis / emergency → force 0.0
    if agent_next_step == "home_rest":
        task_condition = gt_condition
        task_id = task_data.get("task_id", "").lower()
        combined = f"{task_condition} {task_id}"
        if _any_keyword_in_text(EMERGENCY_CONDITIONS, combined):
            score = 0.0

    # P3: blood in sputum not flagged in TB task → −0.30
    task_id_lower = task_data.get("task_id", "").lower()
    if "tb" in task_id_lower or "tuberculosis" in gt_condition:
        complaint = task_data.get("complaint", "").lower()
        blood_discussed = (
            _conversation_mentions(conversation_history, "blood")
            or _conversation_mentions(conversation_history, "hemoptysis")
            or _conversation_mentions(conversation_history, "sputum")
            or "blood" in complaint
        )
        if not blood_discussed:
            score -= 0.30

    # P4: hidden_flag not identified in hard task → cap at 0.20
    hidden_flag = task_data.get("hidden_flag", "")
    if hidden_flag:
        flag_discussed = _conversation_mentions(
            conversation_history, hidden_flag
        )
        if not flag_discussed:
            score = min(score, 0.20)

    # ── Clamp to [0.0, 1.0] ─────────────────────────────────────────────
    return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════════════════
#  Grader class — thin dispatcher used by ClinicalTriageEnv
# ═══════════════════════════════════════════════════════════════════════════

class Grader:
    """Entry-point called by ``ClinicalTriageEnv.step()``.

    The environment passes a dict with ``type`` ("intermediate" | "final")
    plus `action`, `task_data`, and runtime `state`.  The Grader routes to
    the appropriate scoring function and returns ``{"score": float, ...}``.
    """

    def __init__(self) -> None:
        pass

    def grade(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        grade_type = assessment.get("type", "intermediate")
        action     = assessment.get("action", {})
        task_data  = assessment.get("task_data", {})
        state      = assessment.get("state", {})

        if grade_type == "final":
            score = final_score(action, task_data, state)
            return {
                "score": score,
                "type": "final",
                "details": {
                    "agent_risk": (action.get("risk") or ""),
                    "agent_condition": (action.get("condition") or ""),
                    "agent_next_step": (action.get("next_step") or ""),
                    "gt_risk": task_data.get("ground_truth", {}).get("risk", ""),
                    "gt_condition": task_data.get("ground_truth", {}).get("condition", ""),
                    "gt_next_step": task_data.get("ground_truth", {}).get("next_step", ""),
                },
            }

        # Default: intermediate
        score = intermediate_reward(action, task_data, state)
        return {
            "score": score,
            "type": "intermediate",
        }
