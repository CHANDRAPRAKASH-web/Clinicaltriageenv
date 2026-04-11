"""Smoke tests for grader.py — run with: python test_grader.py"""
from server.grader import Grader, intermediate_reward, final_score

g = Grader()
passed = 0
total = 0

def check(name, actual, expected, op="eq"):
    global passed, total
    total += 1
    if op == "eq":
        ok = actual == expected
    elif op == "approx":
        ok = abs(actual - expected) < 0.01
    elif op == "lte":
        ok = actual <= expected
    else:
        ok = False
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  {status}  {name}: got={actual:.4f}  expected={expected}")
    if ok:
        passed += 1

print("=== Intermediate Reward Tests ===")

# 1) ask_patient matching red_flag → +0.15
res = g.grade({
    "type": "intermediate",
    "action": {"action": "ask_patient", "question": "Do you have cyclic_fever?"},
    "task_data": {"red_flags": ["cyclic_fever", "fatigue"], "optimal_tests": ["RDT"]},
    "state": {}
})
check("red_flag match", res["score"], 0.15)

# 2) ask_patient WHO relevant → +0.05
res = g.grade({
    "type": "intermediate",
    "action": {"action": "ask_patient", "question": "Are you taking any medication?"},
    "task_data": {"red_flags": ["cyclic_fever"], "optimal_tests": ["RDT"]},
    "state": {}
})
check("WHO relevant question", res["score"], 0.05)

# 3) ask_patient irrelevant → 0.0
res = g.grade({
    "type": "intermediate",
    "action": {"action": "ask_patient", "question": "What is your favorite color?"},
    "task_data": {"red_flags": ["cyclic_fever"], "optimal_tests": ["RDT"]},
    "state": {}
})
check("irrelevant question", res["score"], 0.0)

# 4) request_test optimal → +0.10
res = g.grade({
    "type": "intermediate",
    "action": {"action": "request_test", "test": "RDT"},
    "task_data": {"red_flags": [], "optimal_tests": ["RDT"]},
    "state": {}
})
check("optimal test", res["score"], 0.10)

# 5) request_test redundant → -0.05
res = g.grade({
    "type": "intermediate",
    "action": {"action": "request_test", "test": "chest_xray"},
    "task_data": {"red_flags": [], "optimal_tests": ["RDT"]},
    "state": {}
})
check("redundant test", res["score"], -0.05)

# 6) request_vital → 0.0
res = g.grade({
    "type": "intermediate",
    "action": {"action": "request_vital", "vital": "temperature"},
    "task_data": {"red_flags": [], "optimal_tests": ["RDT"]},
    "state": {}
})
check("request_vital (neutral)", res["score"], 0.0)


print("\n=== Final Score Tests ===")

# 7) Perfect assessment (easy malaria, 4 steps)
res = g.grade({
    "type": "final",
    "action": {"action": "make_assessment", "risk": "HIGH", "condition": "plasmodium_vivax_malaria", "next_step": "refer_to_PHC"},
    "task_data": {
        "task_id": "easy_malaria",
        "ground_truth": {"risk": "HIGH", "condition": "plasmodium_vivax_malaria", "next_step": "refer_to_PHC"},
        "max_steps": 8
    },
    "state": {"steps_taken": 4, "conversation_history": [], "tests_ordered": []}
})
# 0.40 + 0.25 + 0.20 + 0.15*(4/8) = 0.925
check("perfect easy malaria", res["score"], 0.925, "approx")

# 8) CRITICAL misclassified as LOW → cap 0.2
res = g.grade({
    "type": "final",
    "action": {"action": "make_assessment", "risk": "LOW", "condition": "active_pulmonary_TB", "next_step": "refer_to_district_TB_centre"},
    "task_data": {
        "task_id": "tb_cough",
        "ground_truth": {"risk": "CRITICAL", "condition": "active_pulmonary_TB", "next_step": "refer_to_district_TB_centre"},
        "max_steps": 8
    },
    "state": {"steps_taken": 3, "conversation_history": ["Q: blood?", "A: yes blood in sputum"], "tests_ordered": []}
})
check("CRITICAL→LOW cap", res["score"], 0.20, "lte")

# 9) home_rest for TB → force 0.0
res = g.grade({
    "type": "final",
    "action": {"action": "make_assessment", "risk": "CRITICAL", "condition": "active_pulmonary_TB", "next_step": "home_rest"},
    "task_data": {
        "task_id": "tb_cough",
        "ground_truth": {"risk": "CRITICAL", "condition": "active_pulmonary_TB", "next_step": "refer_to_district_TB_centre"},
        "max_steps": 8
    },
    "state": {"steps_taken": 3, "conversation_history": ["Q: blood?", "A: yes blood"], "tests_ordered": []}
})
check("home_rest for TB", res["score"], 0.0)

# 10) TB task, blood NOT discussed → -0.3 deduction
res = g.grade({
    "type": "final",
    "action": {"action": "make_assessment", "risk": "CRITICAL", "condition": "active_pulmonary_TB", "next_step": "refer_to_district_TB_centre"},
    "task_data": {
        "task_id": "tb_cough",
        "ground_truth": {"risk": "CRITICAL", "condition": "active_pulmonary_TB", "next_step": "refer_to_district_TB_centre"},
        "max_steps": 8
    },
    "state": {"steps_taken": 3, "conversation_history": ["Q: cough?", "A: yes"], "tests_ordered": []}
})
# 0.40 + 0.25 + 0.20 + 0.15*(5/8) - 0.3 = 0.6438
check("blood not flagged in TB", res["score"], 0.6438, "approx")

# 11) hidden_flag (confusion) NOT discussed → cap 0.2
res = g.grade({
    "type": "final",
    "action": {"action": "make_assessment", "risk": "CRITICAL", "condition": "diabetic_foot_sepsis", "next_step": "emergency_ambulance"},
    "task_data": {
        "task_id": "diabetic_sepsis",
        "ground_truth": {"risk": "CRITICAL", "condition": "diabetic_foot_sepsis", "next_step": "emergency_ambulance"},
        "hidden_flag": "confusion",
        "max_steps": 8
    },
    "state": {"steps_taken": 4, "conversation_history": ["Q: wound?", "A: it wont heal"], "tests_ordered": []}
})
check("hidden_flag miss cap", res["score"], 0.20, "lte")

# 12) hidden_flag (confusion) IS discussed → no cap
res = g.grade({
    "type": "final",
    "action": {"action": "make_assessment", "risk": "CRITICAL", "condition": "diabetic_foot_sepsis", "next_step": "emergency_ambulance"},
    "task_data": {
        "task_id": "diabetic_sepsis",
        "ground_truth": {"risk": "CRITICAL", "condition": "diabetic_foot_sepsis", "next_step": "emergency_ambulance"},
        "hidden_flag": "confusion",
        "max_steps": 8
    },
    "state": {"steps_taken": 4, "conversation_history": ["Q: confusion?", "A: yes I feel confused"], "tests_ordered": []}
})
check("hidden_flag found", res["score"], 0.925, "approx")

# 13) Wrong everything
res = g.grade({
    "type": "final",
    "action": {"action": "make_assessment", "risk": "LOW", "condition": "common_cold", "next_step": "home_rest"},
    "task_data": {
        "task_id": "easy_malaria",
        "ground_truth": {"risk": "HIGH", "condition": "plasmodium_vivax_malaria", "next_step": "refer_to_PHC"},
        "max_steps": 8
    },
    "state": {"steps_taken": 2, "conversation_history": [], "tests_ordered": []}
})
check("everything wrong", res["score"], 0.15 * (6/8), "approx")

print(f"\n{'='*40}")
print(f"Results: {passed}/{total} tests passed")
