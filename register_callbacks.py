"""Command registration and agent bridge for /inspect.

Wires the /inspect slash command using @register_command decorator.
Bridges to agent history and launches the TUI.

This module is the entry point loaded by code_puppy's plugin system.
All imports are from public APIs only — no reaching into prune or other
internal modules.
"""

from __future__ import annotations

from code_puppy.agents.agent_manager import get_current_agent
from code_puppy.command_line.command_registry import register_command
from code_puppy.messaging import emit_error

from .inspect_menu import InspectMenu
from .inspect_model import build_inspect_entries


@register_command(
    name="inspect",
    description="Read-only deep inspector for conversation history",
    usage="/inspect",
    aliases=["i"],
    category="plugin",
    detailed_help=(
        "Opens a split-pane TUI showing every message in the current\n"
        "conversation with full detail — no truncation, no summary mode.\n"
        "Read-only: nothing is modified.\n\n"
        "Keys: j/k navigate, h/l scroll detail, c copy detail, q quit."
    ),
)
def handle_inspect_command(command: str) -> bool:
    """Launch the /inspect TUI with the current conversation history.

    Args:
        command: The full command string (e.g. "/inspect")

    Returns:
        True to indicate the command was handled
    """
    del command  # unused — /inspect takes no arguments

    # Get the current agent
    agent = get_current_agent()
    if agent is None:
        emit_error("No active agent — cannot inspect conversation history.")
        return True

    # Extract message history from the agent
    raw_history = agent.get_message_history()
    if not raw_history:
        emit_error("Conversation history is empty — nothing to inspect.")
        return True

    # Build InspectEntry objects from raw pydantic-ai messages
    entries = build_inspect_entries(raw_history)
    if not entries:
        emit_error("No valid messages found in conversation history.")
        return True

    # Launch the TUI
    menu = InspectMenu(entries)
    menu.run()

    return True
