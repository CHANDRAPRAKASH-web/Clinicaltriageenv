import json
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from .grader import Grader, normalize, stem_match


class PatientProfile(BaseModel):
    age: str
    gender: str
    location: str
    complaint: str


class Observation(BaseModel):
    patient_profile: PatientProfile
    presenting_complaint: str
    conversation_history: List[str]
    tests_ordered: List[Dict[str, Any]]
    steps_taken: int
    max_steps: int
    done: bool
    total_reward: float = 0.0
    last_reward: float = 0.0
    final_grading_result: Optional[Dict[str, Any]] = None


class Action(BaseModel):
    action: str
    question: Optional[str] = None
    vital: Optional[str] = None
    test: Optional[str] = None
    risk: Optional[str] = None
    condition: Optional[str] = None
    next_step: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class ClinicalTriageEnv:

    def __init__(self, task_name: str) -> None:
        task_path = os.path.join(os.path.dirname(__file__), "tasks", f"{task_name}.json")

        if not os.path.exists(task_path):
            raise ValueError(f"Task JSON not found at: {task_path}")

        with open(task_path, 'r', encoding='utf-8') as f:
            self.task_data = json.load(f)

        self.grader = Grader()
        self.obs = self._init_observation()

    # -------------------------
    # INIT
    # -------------------------
    def _parse_patient(self, pt_str: str, complaint: str) -> PatientProfile:
        age, gender, location = "Unknown", "Unknown", pt_str

        match = re.match(r"(\d+)(M|F|O)", pt_str, re.IGNORECASE)
        if match:
            age = match.group(1)
            gender = match.group(2)
            parts = pt_str.split(',')
            location = parts[-1].strip() if len(parts) > 1 else pt_str

        return PatientProfile(
            age=age,
            gender=gender.upper(),
            location=location,
            complaint=complaint
        )

    def _init_observation(self) -> Observation:
        return Observation(
            patient_profile=self._parse_patient(
                self.task_data.get("patient", ""),
                self.task_data.get("complaint", "")
            ),
            presenting_complaint=self.task_data.get("complaint", ""),
            conversation_history=[],
            tests_ordered=[],
            steps_taken=0,
            max_steps=self.task_data.get("max_steps", 8),
            done=False
        )

    def reset(self) -> Observation:
        self.obs = self._init_observation()
        return self.obs

    def state(self) -> Observation:
        return self.obs

    # -------------------------
    # KEYWORD MATCH (IMPROVED)
    # -------------------------
    def _score_key_against_question(self, key: str, question: str) -> float:
        """
        Returns a match score between a simulated_answers key and the question.
        Higher = better match.
        """
        key_norm = normalize(key)
        q_norm   = normalize(question)

        # Exact substring
        if key_norm in q_norm:
            return 1.0

        # Stem match
        if stem_match(key_norm, q_norm):
            return 0.8

        # All significant tokens of key appear in question
        tokens = [t for t in key_norm.split() if len(t) > 3]
        if tokens and all(t in q_norm for t in tokens):
            return 0.7

        # Any significant token of key appears in question
        if tokens and any(t in q_norm for t in tokens):
            return 0.4

        return 0.0

    def _keyword_match(self, question: str, simulated_answers: Dict[str, str]) -> str:
        """
        Find the best matching answer from simulated_answers for the given question.
        Uses scored matching — picks highest scoring key.
        Falls back to symptom-aware defaults.
        """
        best_score  = 0.0
        best_answer = None

        for key, answer in simulated_answers.items():
            score = self._score_key_against_question(key, question)
            if score > best_score:
                best_score  = score
                best_answer = answer

        # Threshold: only use match if score is meaningful
        if best_score >= 0.4 and best_answer:
            return best_answer

        # Symptom-aware fallback — don't always say "No"
        q_lower = question.lower()
        if any(w in q_lower for w in ["pain", "cough", "fever", "bleed", "blood",
                                       "vomit", "wound", "confus", "dizz", "weak",
                                       "sweat", "weight", "mosquito", "chill"]):
            return "I'm not sure how to describe it exactly."

        return "No, nothing significant."

    # -------------------------
    # NORMALIZATION
    # -------------------------
    def _normalize_query(self, text: str) -> str:
        return (
            text.lower()
            .split(" to ")[0]
            .split(" for ")[0]
            .replace(" level", "")
            .replace(" monitoring", "")
            .strip()
        )

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action: Action) -> StepResult:

        if self.obs.done:
            return StepResult(
                observation=self.obs,
                reward=0.01,
                done=True,
                info={"error": "Episode already finished"}
            )

        reward = 0.0
        info: Dict[str, Any] = {}

        # ---------------- ASK ----------------
        if action.action == "ask_patient":
            if not action.question:
                info["error"] = "Missing question"
            else:
                answer = self._keyword_match(
                    action.question,
                    self.task_data.get("simulated_answers", {})
                )
                self.obs.conversation_history += [
                    f"Q: {action.question}",
                    f"A: {answer}"
                ]

        # ---------------- TEST / VITAL ----------------
        elif action.action in ["request_test", "request_vital"]:

            query = action.test or action.vital or ""
            query = self._normalize_query(query)

            # Prevent duplicate spam
            existing = [t["name"].lower() for t in self.obs.tests_ordered]
            if query in existing:
                return StepResult(
                    observation=self.obs,
                    reward=0.01,
                    done=False,
                    info={"warning": f"Duplicate test/vital: {query}"}
                )

            source = (
                self.task_data.get("tests", {})
                if action.action == "request_test"
                else self.task_data.get("vitals", {})
            )

            # Fuzzy lookup in test/vital results
            value = self._fuzzy_lookup(query, source)

            self.obs.tests_ordered.append({
                "type": action.action,
                "name": query,
                "result": value
            })

            self.obs.conversation_history.append(
                f"{action.action.upper()} [{query}] → {value}"
            )

        # ---------------- FINAL ASSESSMENT ----------------
        elif action.action == "make_assessment":

            self.obs.done = True

            grade = self.grader.grade({
                "type": "final",
                "action": action.model_dump(),
                "task_data": self.task_data,
                "state": {
                    "steps_taken": self.obs.steps_taken,
                    "conversation_history": self.obs.conversation_history,
                    "tests_ordered": self.obs.tests_ordered
                }
            })

            reward = grade.get("score", 0.01)
            self.obs.final_grading_result = grade

        else:
            info["error"] = f"Unknown action: {action.action}"

        # ---------------- STEP COUNT ----------------
        self.obs.steps_taken += 1

        if self.obs.steps_taken >= self.obs.max_steps:
            self.obs.done = True

        # ---------------- INTERMEDIATE REWARD ----------------
        if not self.obs.done and action.action != "make_assessment":
            try:
                interm = self.grader.grade({
                    "type": "intermediate",
                    "action": action.model_dump(),
                    "task_data": self.task_data,
                    "state": {
                        "steps_taken": self.obs.steps_taken,
                        "conversation_history": self.obs.conversation_history,
                        "tests_ordered": self.obs.tests_ordered
                    }
                })
                reward = interm.get("score", 0.01)
            except Exception as e:
                info["grader_error"] = str(e)
                reward = 0.01

        self.obs.last_reward = reward
        self.obs.total_reward += reward

        return StepResult(
            observation=self.obs,
            reward=reward,
            done=self.obs.done,
            info=info
        )

    # -------------------------
    # FUZZY TEST/VITAL LOOKUP
    # -------------------------
    def _fuzzy_lookup(self, query: str, source: Dict[str, str]) -> str:
        """
        Look up a test/vital result with fuzzy matching.
        Tries exact → substring → stem → fallback.
        """
        query = normalize(query)

        # Exact match
        if query in source:
            return source[query]

        # Substring match
        for key, val in source.items():
            k = normalize(key)
            if query in k or k in query:
                return val

        # Stem match
        for key, val in source.items():
            if stem_match(query, normalize(key)):
                return val

        return "Result: Normal / Not available"