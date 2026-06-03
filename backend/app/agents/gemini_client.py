import json
from google import genai
from app.core.config import settings
from app.schemas import agent_tools, TOOL_DISPATCH

client = genai.Client(api_key=settings.GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """
You are an expert e-sports draft analyst. You have access to a 
calibrated predictive model and SHAP explainability tools.

When a user asks for a draft recommendation, follow this reasoning process:
1. Call get_win_probability to establish the current baseline.
2. Call simulate_pick for at least 2-3 candidate champions to compare outcomes.
3. Call get_shap_explanation on your top candidate to understand why it performs best.
4. Synthesize your findings into a concise recommendation that cites specific probability 
   values and SHAP contributions. Never make a claim you cannot back with a tool result.
"""

tools = agent_tools


def run_agent(user_message: str, previous_interaction_id: str | None = None) -> str:
    """Run the draft analyst agent, automatically resolving all tool calls.

    Args:
        user_message: The user's input text or a list of function_result dicts.
        previous_interaction_id: Pass the ID from a previous interaction to
            continue a multi-turn conversation (server keeps history).

    Returns:
        The final text response from the model.
    """
    interaction = client.interactions.create(
        model=settings.GEMINI_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools,
        input=user_message,
        previous_interaction_id=previous_interaction_id,
    )

    # Agentic loop — keep resolving function calls until the model is done
    while True:
        calls = [s for s in interaction.steps if s.type == "function_call"]
        if not calls:
            break

        results = []
        for call in calls:
            fn = TOOL_DISPATCH.get(call.name)
            try:
                output = (
                    fn(**call.arguments)
                    if fn
                    else {"error": f"Unknown tool: {call.name}"}
                )
            except Exception as exc:
                output = {"error": str(exc)}

            results.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [{"type": "text", "text": json.dumps(output)}],
                }
            )

        interaction = client.interactions.create(
            model=settings.GEMINI_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
            tools=tools,
            previous_interaction_id=interaction.id,
            input=results,
        )

    # Return the text from the final model_output step
    for step in reversed(interaction.steps):
        if step.type == "model_output":
            return step.content[0].text

    return ""
