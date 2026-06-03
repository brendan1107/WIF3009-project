import json
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel
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
    active_team: str
    active_action: str
    active_role: str
    blue_picks: List[str]
    red_picks: List[str]
    blue_bans: List[str]
    red_bans: List[str]
    recommended_bans: List[Dict[str, Any]]
    recommended_picks: List[Dict[str, Any]]
    matchup_counters: List[Dict[str, Any]]

    # Structured response fields
    draft_warning: str
    recommendations: str
    counter_analysis: str


class UnifiedResponse(BaseModel):
    draft_warning: str
    recommendations: str
    counter_analysis: str


ANALYST_SYSTEM_INSTRUCTION = """
You are an expert e-sports draft analyst, lane coach, and matchups specialist.
Your job is to act as a drafting co-pilot. You will receive a rich JSON payload containing the current draft state, telemetry, and lane matchup metrics.

You must analyze this payload and provide a JSON response with the following three fields:
1. `draft_warning`: A strict 1-sentence tactical warning about the drafting trap (the driver with the largest negative SHAP impact). If no negative drivers exist, suggest maintaining composition balance. Maximum of 12 words. Do NOT use emojis.
2. `recommendations`: Suggest exactly 2 or 3 optimal champion pick or ban targets. If the active action is 'pick', your recommendations MUST be STRICTLY targeted to the currently active role being picked (e.g. TOP, JUNGLE, MID, BOTTOM, SUPPORT). For each recommended champion, you MUST include a short one-sentence explanation strictly based on top SHAP drivers and telemetry (e.g. synergy scores or average win rate impacts). Format this field as a plain-text numbered list (e.g., '1. ChampionA: Explanation. 2. ChampionB: Explanation. 3. ChampionC: Explanation.') and do NOT return a JSON list or array. Maximum 12 words per hero explanation. Do NOT use emojis.
3. `counter_analysis`: Identify lane matchup counters and explain them:
   - If active head-to-head matchups exist (both teams locked same role): identify the most critical head-to-head matchup (deviation from 50%) and write a single, highly-focused tactical lane matchup summary. Maximum of 15 words.
   - If no direct head-to-head matchup is locked yet (picks are still in progress): analyze the opponent's already picked champions and write a highly targeted tactical counter recommendation for the active role (maximum of 15 words).
   - If no picks are locked yet (e.g. early bans): write a targeted tactical guidance on the primary lane counter-picks to prioritize or ban (maximum of 15 words).
   - Do NOT use emojis.

Ensure all outputs are clear, action-oriented, and highly readable.
"""


def analyst_node(state: AgentState) -> Dict[str, Any]:
    """Unified LangGraph node that translates rich draft telemetry into structured analyst JSON in one turn."""
    user_prompt = (
        f"Telemetry Payload:\n"
        f"- Expected Base Win Probability: {state['expected_base_win_prob']:.4f}\n"
        f"- Top Drivers: {state['top_drivers']}\n"
        f"- Active Team: {state.get('active_team', '')}\n"
        f"- Active Action: {state.get('active_action', '')}\n"
        f"- Active Role: {state.get('active_role', '')}\n"
        f"- Blue Picks: {state.get('blue_picks', [])}\n"
        f"- Red Picks: {state.get('red_picks', [])}\n"
        f"- Blue Bans: {state.get('blue_bans', [])}\n"
        f"- Red Bans: {state.get('red_bans', [])}\n"
        f"- Matchup Counters: {state.get('matchup_counters', [])}\n"
    )

    if state.get("recommended_bans"):
        user_prompt += f"- Pre-calculated Recommended Bans (with simulated win rates if picked by opponent): {state['recommended_bans']}\n"

    if state.get("recommended_picks"):
        user_prompt += f"- Pre-calculated Recommended Picks (with simulated win rates): {state['recommended_picks']}\n"

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=user_prompt,
            config={
                "system_instruction": ANALYST_SYSTEM_INSTRUCTION,
                "response_mime_type": "application/json",
                "response_schema": UnifiedResponse,
            },
        )

        data = json.loads(response.text.strip())
        return {
            "draft_warning": data.get("draft_warning", ""),
            "recommendations": data.get("recommendations", ""),
            "counter_analysis": data.get("counter_analysis", ""),
            "explanation": response.text.strip(),
        }
    except Exception as e:
        print(f"Error generating unified analyst response: {e}")
        return {
            "draft_warning": "No critical composition vulnerabilities detected.",
            "recommendations": "Follow recommended picks and bans to secure advantage.",
            "counter_analysis": "Picks in progress. Review lane stats.",
            "explanation": "",
        }


# Build the LangGraph State Machine (Optimized to Single Turn)
workflow = StateGraph(AgentState)
workflow.add_node("analyst", analyst_node)

workflow.add_edge(START, "analyst")
workflow.add_edge("analyst", END)

agent_app = workflow.compile()
