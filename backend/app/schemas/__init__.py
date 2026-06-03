from .payloads import (
    # Python callables (for local execution after a function_call step)
    get_win_probability,
    get_shap_explanation,
    get_champion_meta_stats,
    simulate_pick,
    # JSON Schema dicts (pass these in tools= to interactions.create())
    TOOL_GET_WIN_PROBABILITY,
    TOOL_GET_SHAP_EXPLANATION,
    TOOL_SIMULATE_PICK,
    TOOL_GET_CHAMPION_META_STATS,
    # Dispatch map: tool name → local callable
    TOOL_DISPATCH,
)

# The list of tool schema dicts to pass to the Interactions API
agent_tools = [
    TOOL_GET_WIN_PROBABILITY,
    TOOL_GET_SHAP_EXPLANATION,
    TOOL_SIMULATE_PICK,
    TOOL_GET_CHAMPION_META_STATS,
]

__all__ = [
    "agent_tools",
    "TOOL_DISPATCH",
    "get_win_probability",
    "get_shap_explanation",
    "simulate_pick",
    "get_champion_meta_stats",
]
