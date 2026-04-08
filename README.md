---
title: ClinicalTriageEnv
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

<div align="center">

# 🏥 ClinicalTriageEnv

**A rigorous OpenEnv-compliant simulation platform for evaluating Medical AI agents in resource-constrained clinical settings.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Uvicorn-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)](https://www.docker.com/)
[![WHO Guidelines](https://img.shields.io/badge/Guidelines-WHO%20%7C%20ICMR-red)](https://www.who.int/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 Overview

**ClinicalTriageEnv** is designed to act as a testing ground for Medical AI agents operating in high-stakes, resource-constrained environments — such as rural India, where qualified doctors are scarce and every diagnostic decision carries real consequences.

Grounded in **WHO** and **ICMR** (Indian Council of Medical Research) diagnostic guidelines, this environment challenges LLMs to:

- Conduct patient dialogues effectively
- Order diagnostic tests selectively without wasting resources
- Render high-quality, safe clinical assessments

> ⚠️ **AI triage systems must be precise, cost-conscious, and safe.** This environment enforces all three.

---

## 🎬 Demo

> *(Screenshot or GIF of agent interaction goes here)*

---

## ⚙️ Action Space

The agent navigates the environment using **one action per step**, chosen from the following four types:

| Action | Required Fields | Example Payload |
|---|---|---|
| `ask_patient` | `question` | `{"action": "ask_patient", "question": "Have you noticed any blood in your sputum?"}` |
| `request_test` | `test` | `{"action": "request_test", "test": "Sputum AFB Stain"}` |
| `request_vital` | `vital` | `{"action": "request_vital", "vital": "blood pressure"}` |
| `make_assessment` | `risk`, `condition`, `next_step` | `{"action": "make_assessment", "risk": "CRITICAL", "condition": "Diabetic Sepsis", "next_step": "IV Antibiotics & Ambulance"}` |

---

## 🔭 Observation Space

At every step, the agent receives a fully updated JSON state:

| Field | Type | Description |
|---|---|---|
| `patient_profile` | `String` | Static demographic and chief complaint data |
| `conversation_history` | `List` | Running log of AI questions and simulated patient responses |
| `vitals_taken` | `Dictionary` | Vital check results acquired so far |
| `lab_results` | `Dictionary` | Diagnostic test results acquired so far |
| `completed_steps` | `Integer` | Standardized tracking of current workflow efficiency |

---

## 🩺 Task Descriptions

Three progressive difficulty tiers test different clinical reasoning capabilities:

### 🟢 Easy — Malaria
> A 28-year-old farmer presents with cyclic fever.

**Challenge:** Recognize classic vector-borne exposure and complete a standard infectious disease workup.
**Step Limit:** 12 steps

---

### 🟡 Medium — Tuberculosis
> A 45-year-old urban worker presents with a chronic 3-week cough.

**Challenge:** The agent must not dismiss this as a standard flu. It is required to ask the critical **bloody sputum** red-flag question.
**Step Limit:** 10 steps

---

### 🔴 Hard — Diabetic Sepsis
> A 68-year-old diabetic presents with a minor foot wound — but is exhibiting signs of confusion.

**Challenge:** A stealthy systemic infection. The agent must identify confusion as a sepsis symptom and immediately escalate to **CRITICAL**, rather than prescribing a generic foot ointment.
**Step Limit:** 8 steps

---

## 🏆 Reward Function

Scores are bounded strictly within `[0.0, 1.0]` and consist of base evaluation components plus critical safety modifiers.

### Base Components

| Component | Weight | Description |
|---|---|---|
| **Risk Classification** | `+0.40` | Accurately identifying triage urgency (e.g., LOW vs. CRITICAL) |
| **Condition Diagnosis** | `+0.25` | Correctly naming the underlying pathology |
| **Next Step Plan** | `+0.20` | Selecting the right treatment or referral pathway |
| **Efficiency Bonus** | `+0.15` | Formula: `0.15 × (Steps Saved / Max Steps)` — faster diagnosis yields higher rewards |

### Intermediate Modifiers & Penalties

| Trigger | Modifier | Effect |
|---|---|---|
| Optimal / Red Flag Query | `+0.25` to `+0.32` | Strong reinforcement for catching disease-specific danger markers |
| Basic WHO Checks | `+0.08` to `+0.12` | Positive reinforcement for thorough baseline checks (temperature, BP) |
| Missed Red Flag | `−0.20` | Penalty for failing to ask the disease's primary danger question before assessing |
| Danger Misclassification | `−0.30` | Severe penalty for assigning LOW/MODERATE risk to a HIGH/CRITICAL patient |
| Step Exhaustion | `−0.20` | Penalty for maxing out the step counter without forming an assessment |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker *(optional)*
- A valid API key from OpenAI or Groq

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-org/ClinicalTriageEnv.git
cd ClinicalTriageEnv
```

**2. Set up a virtual environment**

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_groq_or_openai_key
```

**4. Start the server**

```bash
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

**5. Run the AI agent**

In a separate terminal:

```bash
python inference.py
```

> 💡 Alternatively, use the provided **Dockerfile** to containerize the entire setup.

---

## 📊 Baseline Scores

Performance benchmarks using modern LLMs (e.g., Llama-3-70B, GPT-4o):

| Difficulty | Condition | Expected Score |
|---|---|---|
| 🟢 Easy | Malaria | ~0.85 |
| 🟡 Medium | Tuberculosis | ~0.70 |
| 🔴 Hard | Diabetic Sepsis | ~0.55 |

---

## 🗂️ Project Structure

```
ClinicalTriageEnv/
├── server/
│   └── main.py          # FastAPI server & environment logic
├── tasks/               # Task definitions (Easy, Medium, Hard)
├── inference.py         # Agent runner script
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request. Ensure any new tasks are grounded in WHO or ICMR clinical guidelines.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

For questions or collaborations, please open a GitHub issue or reach out via the repository's discussion board.

---

<div align="center">
  <sub>Built with ❤️ to bridge the healthcare gap in underserved communities.</sub>
</div>
