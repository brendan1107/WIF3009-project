from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.agents import run_agent

router = APIRouter()


class AgentRequest(BaseModel):
    draft_state: dict
    user_question: str
    interaction_id: Optional[str] = None


@router.post("/agent")
def agent_endpoint(req: AgentRequest) -> dict:
    prompt = f"""
    Current draft state: {req.draft_state}
    User question: {req.user_question}
    """
    answer = run_agent(user_message=prompt, previous_interaction_id=req.interaction_id)
    return {"response": answer}
