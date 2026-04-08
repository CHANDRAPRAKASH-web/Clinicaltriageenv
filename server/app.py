import uuid
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .environment import ClinicalTriageEnv, Observation, Action, StepResult

app = FastAPI(title="ClinicalTriageEnv API")

# Dictionary to hold environment instances by session ID
sessions: Dict[str, ClinicalTriageEnv] = {}

# Map task names to the corresponding json filename prefixes
TASK_MAP = {
    "easy_malaria": "easy",
    "medium_tb": "medium",
    "hard_sepsis": "hard"
}

class ResetRequest(BaseModel):
    task: str = "easy_malaria"
    session_id: Optional[str] = None

class ResetResponse(BaseModel):
    session_id: str
    observation: Observation

class StepRequest(BaseModel):
    session_id: str
    action: Action

@app.post("/reset", response_model=ResetResponse)
def reset_env(request: ResetRequest = ResetRequest()):
    # Retrieve mapped filename prefix, defaulting to the raw task string just in case, but fail if not found
    task_file = TASK_MAP.get(request.task)
    if not task_file:
        raise HTTPException(status_code=400, detail=f"Invalid task name: {request.task}")
    
    # Auto-generate session_id if none is provided
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        env = ClinicalTriageEnv(task_name=task_file)
        obs = env.reset()
        sessions[session_id] = env
        return ResetResponse(session_id=session_id, observation=obs)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResult)
def step_env(request: StepRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env = sessions[request.session_id]
    try:
        result = env.step(request.action)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=Observation)
def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env = sessions[session_id]
    return env.state()

@app.get("/health")
def health_check():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

