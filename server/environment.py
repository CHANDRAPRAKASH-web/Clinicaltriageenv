import json
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from .grader import Grader

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
        # Load the corresponding JSON from tasks/
        task_path = os.path.join(os.path.dirname(__file__), "tasks", f"{task_name}.json")
        if not os.path.exists(task_path):
            raise ValueError(f"Task JSON not found at: {task_path}")
            
        with open(task_path, 'r', encoding='utf-8') as f:
            self.task_data = json.load(f)
            
        self.grader = Grader()
        self.obs = self._init_observation()

    def _parse_patient(self, pt_str: str, complaint: str) -> PatientProfile:
        age = "Unknown"
        gender = "Unknown"
        location = pt_str
        
        # Regex to try parsing out "28M" or "55F"
        match = re.match(r"(\d+)(M|F|O)", pt_str, re.IGNORECASE)
        if match:
            age = match.group(1)
            gender = match.group(2)
            parts = pt_str.split(',')
            location = parts[-1].strip() if len(parts) > 1 else pt_str.replace(match.group(0), '').strip()
            
        return PatientProfile(
            age=age, 
            gender=gender.upper(), 
            location=location, 
            complaint=complaint
        )

    def _init_observation(self) -> Observation:
        patient_str = self.task_data.get("patient", "")
        complaint = self.task_data.get("complaint", "")
        
        profile = self._parse_patient(patient_str, complaint)
        
        return Observation(
            patient_profile=profile,
            presenting_complaint=complaint,
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

    def _keyword_match(self, question: str, simulated_answers: Dict[str, str]) -> str:
        question_lower = question.lower()
        for key, answer in simulated_answers.items():
            clean_key = key.lower().replace("_", " ")
            key_parts = key.lower().split("_")
            
            if (key.lower() in question_lower or 
                clean_key in question_lower or 
                any(p in question_lower for p in key_parts if len(p) > 2)):
                return answer
        
        # --- NEW: Fallback Logic to prevent 'I don't understand' loops ---
        negative_triggers = ["cough", "pain", "headache", "vomit", "diarrhea", "appetite", "weight", "injury", "blood"]
        if any(t in question_lower for t in negative_triggers):
            return "No, I don't have that symptom."
            
        return "No, nothing else like that."

    def _map_medical_term(self, term: str) -> str:
        """Maps common long names to JSON keys."""
        synonyms = {
            "complete blood count": "cbc",
            "blood count": "cbc",
            "blood culture": "blood_culture",
            "oxygen": "spo2",
            "saturation": "spo2",
            "blood glucose": "blood_sugar",
            "blood sugar": "blood_sugar",
            "glucose": "blood_sugar",
            "sugar": "blood_sugar",
            "hemoptysis": "blood",
            "coughing blood": "blood",
            "sweating": "sweat",
            "rapid test": "rdt",
            "malaria": "rdt",
            "rdt": "rdt",
            "cbnaat": "sputum_test",
            "tb test": "sputum_test",
            "xray": "x_ray",
            "chest xray": "x_ray",
            "sputum": "sputum_test",
            "breathing": "respiratory_rate",
            "pulse": "heart_rate",
            "bp": "bp",
            "blood pressure": "bp"
        }
        term_clean = term.lower().strip()
        for syn, official in synonyms.items():
            if syn in term_clean:
                return official
        return term_clean

    def step(self, action: Action) -> StepResult:
        if self.obs.done:
            return StepResult(
                observation=self.obs,
                reward=0.0,
                done=True,
                info={"error": "Episode is already done."}
            )
            
        reward = 0.0
        info: Dict[str, Any] = {}
        
        # Execution of the action
        if action.action == "ask_patient":
            if action.question:
                sim_answers = self.task_data.get("simulated_answers", {})
                answer = self._keyword_match(action.question, sim_answers)
                self.obs.conversation_history.append(f"Q: {action.question}")
                self.obs.conversation_history.append(f"A: {answer}")
            else:
                info["error"] = "ask_patient action requires a 'question' field"
                
        elif action.action == "request_vital":
            query_str = action.vital if (action.vital and action.vital != "string") else action.test
            # --- NEW: CLEANING LOGIC ---
            # Remove fluff like "to check", "to assess", "level", "monitoring"
            query_str = query_str.split(" to ")[0].split(" for ")[0].lower().replace(" level", "").replace(" monitoring", "").strip()
            
            queries = [q.strip() for q in query_str.replace(" and ", ",").split(",")]
            
            for query in queries:
                if not query or query == "string": continue
                vitals = self.task_data.get("vitals", {})
                mapped_query = self._map_medical_term(query)
                val = vitals.get(mapped_query, vitals.get(query.lower().replace(" ", "_")))
                
                if not val:
                    # --- NEW: General Vitals Fallback ---
                    GENERAL_VITALS = {
                        "blood_pressure": "120/80 mmHg (Normal)",
                        "bp": "120/80 mmHg (Normal)",
                        "heart_rate": "72 bpm (Normal)",
                        "pulse": "72 bpm (Normal)",
                        "temperature": "98.6 F (Normal)",
                        "oxygen": "98% (Normal)",
                        "spo2": "98% (Normal)",
                        "weight": f"{self.task_data.get('weight', '70')} kg",
                        "respiratory_rate": "16 breaths/min (Normal)"
                    }
                    for v_key, v_val in GENERAL_VITALS.items():
                        if v_key in mapped_query.lower():
                            val = v_val
                            break

                if not val:
                    for v_k, v_v in vitals.items():
                        if query.lower() in v_k.lower() or v_k.lower() in query.lower():
                            val = v_v
                            break
                
                val = val or "Not available at this facility"
                self.obs.tests_ordered.append({"type": "vital", "name": query, "result": val})
                self.obs.conversation_history.append(f"Vital Check [{query}]: {val}")
                
        elif action.action == "request_test":
            query_str = action.test if (action.test and action.test != "string") else action.vital
            # --- NEW: CLEANING LOGIC ---
            query_str = query_str.split(" to ")[0].split(" for ")[0].lower().replace(" level", "").replace(" monitoring", "").strip()
            
            queries = [q.strip() for q in query_str.replace(" and ", ",").split(",")]
            
            for query in queries:
                if not query or query == "string": continue
                tests = self.task_data.get("tests", {})
                mapped_query = self._map_medical_term(query)
                val = tests.get(mapped_query, tests.get(query.lower().replace(" ", "_")))
                
                if not val:
                    # --- NEW: General Lab Fallback ---
                    # If the task doesn't have it, assume it's a standard test with normal results
                    GENERAL_LABS = {
                        "lft": "Normal (Bilirubin: 0.8 mg/dL, ALT/AST: normal)",
                        "liver": "Normal (Bilirubin: 0.8 mg/dL, ALT/AST: normal)",
                        "kft": "Normal (Creatinine: 0.9 mg/dL, Urea: normal)",
                        "kidney": "Normal (Creatinine: 0.9 mg/dL, Urea: normal)",
                        "lipid": "Total Cholesterol: 180 mg/dL (Normal)",
                        "urinalysis": "Clear, No glucose, No protein",
                        "urine": "Clear, No glucose, No protein",
                        "hiv": "Non-reactive (Negative)",
                        "culture": "Negative (No growth detected after 24 hours)",
                        "typhoid": "Negative (Widal test non-reactive)",
                        "dengue": "NS1 Negative",
                        "heart": "ECG shows Normal Sinus Rhythm",
                        "ecg": "Normal Sinus Rhythm"
                    }
                    for lab_key, lab_val in GENERAL_LABS.items():
                        if lab_key in mapped_query.lower():
                            val = lab_val
                            break

                if not val:
                    # Final attempt: Fuzzy match against task keys
                    for t_k, t_v in tests.items():
                        if query.lower() in t_k.lower() or t_k.lower() in query.lower():
                            val = t_v
                            break
                    # If still no match, check if it's the "Optimal" test the AI is trying to guess
                    if not val:
                        optimal = self.task_data.get("optimal_tests", [])
                        for opt_test in optimal:
                            if opt_test.lower() in query.lower() or query.lower() in opt_test.lower():
                                val = tests.get(opt_test, "Pending/Positive")
                                break
                            
                val = val or "Normal (No abnormalities detected)"
                self.obs.tests_ordered.append({"type": "test", "name": query, "result": val})
                self.obs.conversation_history.append(f"Lab Test [{query}]: {val}")
                
        elif action.action == "make_assessment":
            self.obs.done = True
            self.obs.conversation_history.append(f"FINAL ASSESSMENT: Risk={action.risk}, Condition={action.condition}, NextStep={action.next_step}")
            try:
                # End state Grader call
                grade_res = self.grader.grade({
                    "type": "final",
                    "action": action.model_dump(),
                    "task_data": self.task_data,
                    "state": {
                        "steps_taken": self.obs.steps_taken,
                        "conversation_history": list(self.obs.conversation_history),
                        "tests_ordered": [t.copy() for t in self.obs.tests_ordered],
                    },
                })
                if isinstance(grade_res, dict):
                    reward = grade_res.get("score", 0.0)
                    info["grading_result"] = grade_res
                    self.obs.final_grading_result = grade_res
            except Exception as e:
                info["grading_error"] = str(e)
                
        self.obs.steps_taken += 1
        
        if self.obs.steps_taken >= self.obs.max_steps:
            self.obs.done = True
            
        # Optional intermediate reward check
        if not self.obs.done and action.action != "make_assessment":
            try:
                interm_res = self.grader.grade({
                    "type": "intermediate",
                    "action": action.model_dump(),
                    "task_data": self.task_data,
                    "state": {
                        "steps_taken": self.obs.steps_taken,
                        "conversation_history": list(self.obs.conversation_history),
                        "tests_ordered": [t.copy() for t in self.obs.tests_ordered],
                    },
                })
                if isinstance(interm_res, dict):
                    reward = interm_res.get("score", 0.0)
            except Exception:
                pass
                
        self.obs.last_reward = reward
        self.obs.total_reward += reward
                
        return StepResult(
            observation=self.obs,
            reward=reward,
            done=self.obs.done,
            info=info
        )
