from typing import TypedDict, List, Dict, Any
from google import genai
from langgraph.graph import StateGraph, START, END
from app.core.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

class Driver(TypedDict):
    feature: str
    value: float
    impact_on_win_prob: float

class AgentState(TypedDict):
    expected_base_win_prob: float
    top_drivers: List[Driver]
    explanation: str

COACH_SYSTEM_INSTRUCTION = """
You are an expert e-sports draft analyst and tactical head coach.
Your job is to act as a drafting co-pilot. You will receive a pre-calculated JSON payload containing:
1. The expected base win probability.
2. The top SHAP drivers/features affecting the draft, along with their impact values.

Analyze the telemetry and output a strict, 1-2 sentence tactical coaching advice.
Identify the "Drafting Trap" by finding the driver with the largest negative `impact_on_win_prob` and write a strict, one-sentence tactical warning about it.
Do not hallucinate or make claims that are not supported by the input telemetry. Citing specific percentages or metrics from the input is encouraged.
Keep your response concise, action-oriented, and easy to read on the screen.
"""

def coaching_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node that translates raw SHAP and win-prob data into coaching text."""
    user_prompt = f"Telemetry Payload:\n" \
                  f"- Expected Base Win Probability: {state['expected_base_win_prob']:.4f}\n" \
                  f"- Top Drivers: {state['top_drivers']}"

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=user_prompt,
        config={"system_instruction": COACH_SYSTEM_INSTRUCTION}
    )

    return {"explanation": response.text.strip()}

# Build the LangGraph State Machine
workflow = StateGraph(AgentState)
workflow.add_node("coach", coaching_node)
workflow.add_edge(START, "coach")
workflow.add_edge("coach", END)

# Compile into a deployable application layer
agent_app = workflow.compile()
