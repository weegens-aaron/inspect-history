"""Data model and pydantic-ai introspection for /inspect.

This module provides the core data structures for the /inspect TUI:
- PartDetail: captures every field from a pydantic-ai message part
- UsageDetail: token usage snapshot from ModelResponse
- InspectEntry: one message with all parts, metadata, computed fields
- build_inspect_entries(): walks raw history, returns list[InspectEntry]

Zero truncation: all content is preserved in full.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence


# -- palette -----------------------------------------------------------------
# Style strings for prompt_toolkit formatted-text tuples. Self-contained
# (copied from prune pattern) so this plugin has no external dependencies.

C_CURSOR = "bold fg:ansicyan"
C_USER = "fg:ansigreen"
C_ASSISTANT = "fg:ansiblue"
C_TOOL = "fg:ansiyellow"
C_TOOL_RETURN = "fg:ansiyellow"
C_RETRY = "fg:ansired"
C_SYSTEM = "bold fg:ansicyan"
C_THINKING = "fg:ansimagenta"
C_FILE = "fg:ansiwhite"
C_DIM = "fg:ansibrightblack"
C_HEADER = "dim cyan"
C_FOOTER = "fg:ansigreen"
C_WARN = "fg:ansiyellow"
C_ERROR = "fg:ansired"

# Detail-pane semantics: this is a deep-dive tool, so favour COLOUR over dim.
# Metadata keys get a distinct hue while their values render in the terminal's
# default foreground (theme-safe, high-contrast) instead of muted grey. Section
# labels (content:, args:, usage:) pop in bold so blocks are easy to scan.
C_KEY = "fg:ansicyan"
C_LABEL = "bold fg:ansicyan"


# -- tool classification -----------------------------------------------------

_WRITE_TOOLS = {
    "edit_file",
    "create_file",
    "replace_in_file",
    "delete_snippet",
    "delete_file",
}
_SHELL_TOOLS = {"agent_run_shell_command"}
_BROWSER_PREFIX = "browser_"
_TERMINAL_PREFIX = "terminal_"


def classify_tool(tool_name: str) -> str:
    """Return a short icon hinting at the tool's side-effect kind."""
    if tool_name in _WRITE_TOOLS:
        return "[W]"
    if tool_name in _SHELL_TOOLS:
        return "[!]"
    if tool_name.startswith(_BROWSER_PREFIX):
        return "[B]"
    if tool_name.startswith(_TERMINAL_PREFIX):
        return "[T]"
    return "[.]"


# -- data classes ------------------------------------------------------------


@dataclass
class PartDetail:
    """One part within a message, preserving all pydantic-ai fields."""

    part_kind: str
    # "system-prompt" | "user-prompt" | "tool-return" | "retry-prompt"
    # | "text" | "thinking" | "tool-call" | "file"
    # | "builtin-tool-call" | "builtin-tool-return"

    content: str  # Full content, never truncated

    # Tool-related (populated for tool-call / tool-return / retry-prompt)
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_args: dict | None = None  # Full args dict for tool-call parts

    # Thinking-related
    signature: str | None = None  # ThinkingPart.signature

    # Metadata from the part itself
    timestamp: datetime | None = None  # Part-level timestamp
    dynamic_ref: str | None = None  # SystemPromptPart.dynamic_ref

    # Provider details (TextPart, ThinkingPart, ToolCallPart carry these)
    provider_name: str | None = None
    provider_details: dict | None = None
    part_id: str | None = None  # TextPart.id, ThinkingPart.id

    @property
    def role_color(self) -> str:
        """Return the appropriate color constant for this part kind."""
        colors = {
            "system-prompt": C_SYSTEM,
            "user-prompt": C_USER,
            "text": C_ASSISTANT,
            "thinking": C_THINKING,
            "tool-call": C_TOOL,
            "builtin-tool-call": C_TOOL,
            "tool-return": C_TOOL_RETURN,
            "builtin-tool-return": C_TOOL_RETURN,
            "retry-prompt": C_RETRY,
            "file": C_FILE,
        }
        return colors.get(self.part_kind, C_DIM)


@dataclass
class UsageDetail:
    """Token usage snapshot from a ModelResponse."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    input_audio_tokens: int = 0
    output_audio_tokens: int = 0
    cache_audio_read_tokens: int = 0
    details: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def has_cache(self) -> bool:
        """True if any cache tokens were used."""
        return self.cache_write_tokens > 0 or self.cache_read_tokens > 0

    @property
    def has_audio(self) -> bool:
        """True if any audio tokens were used."""
        return (
            self.input_audio_tokens > 0
            or self.output_audio_tokens > 0
            or self.cache_audio_read_tokens > 0
        )


@dataclass
class InspectEntry:
    """One message in the conversation, capturing everything."""

    history_index: int
    kind: str  # "request" | "response"
    role: str  # "system" | "user" | "assistant" | "tool-return" | "mixed" | "unknown"
    parts: list[PartDetail] = field(default_factory=list)
    preview: str = ""  # Short summary for list pane (~80 chars)

    # Message-level metadata
    timestamp: datetime | None = None
    instructions: str | None = None  # ModelRequest.instructions
    run_id: str | None = None
    metadata: dict | None = None

    # Response-only fields
    model_name: str | None = None
    provider_name: str | None = None
    provider_url: str | None = None
    provider_details: dict | None = None
    provider_response_id: str | None = None
    # provider_request_id removed - deprecated in pydantic-ai
    finish_reason: str | None = None
    usage: UsageDetail | None = None

    # Computed convenience fields
    tool_call_count: int = 0
    thinking_block_count: int = 0
    has_system_prompt: bool = False

    @property
    def role_color(self) -> str:
        """Return the appropriate color constant for this entry's role."""
        colors = {
            "system": C_SYSTEM,
            "user": C_USER,
            "assistant": C_ASSISTANT,
            "tool-return": C_TOOL_RETURN,
            "mixed": C_DIM,
            "unknown": C_DIM,
        }
        return colors.get(self.role, C_DIM)

    @property
    def role_short(self) -> str:
        """Return a short 3-char label for the role."""
        labels = {
            "system": "SYS",
            "user": "USR",
            "assistant": "AST",
            "tool-return": "TRT",
            "mixed": "MIX",
            "unknown": "???",
        }
        return labels.get(self.role, "???")


# -- string helpers ----------------------------------------------------------


def _short_str(value: Any, limit: int = 80) -> str:
    """Truncate a value to a short preview string.

    If truncation occurs, result is exactly `limit` chars including '...' suffix.
    """
    if value is None:
        return ""
    s = str(value).replace("\n", " ").replace("\r", " ").strip()
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


# -- pydantic-ai introspection -----------------------------------------------


def _extract_part(part: Any) -> PartDetail | None:
    """Extract a PartDetail from a pydantic-ai part object."""
    try:
        from pydantic_ai.messages import (
            SystemPromptPart,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )

        # Optional part types that may not exist in all pydantic-ai versions
        try:
            from pydantic_ai.messages import ThinkingPart
        except ImportError:
            ThinkingPart = None  # type: ignore[assignment, misc]

        try:
            from pydantic_ai.messages import RetryPromptPart
        except ImportError:
            RetryPromptPart = None  # type: ignore[assignment, misc]

        try:
            from pydantic_ai.messages import FilePart
        except ImportError:
            FilePart = None  # type: ignore[assignment, misc]

        try:
            from pydantic_ai.messages import BuiltinToolCallPart
        except ImportError:
            BuiltinToolCallPart = None  # type: ignore[assignment, misc]

        try:
            from pydantic_ai.messages import BuiltinToolReturnPart
        except ImportError:
            BuiltinToolReturnPart = None  # type: ignore[assignment, misc]

    except ImportError:
        # pydantic-ai not available - return generic part
        return PartDetail(
            part_kind="unknown",
            content=str(part),
        )

    # SystemPromptPart
    if isinstance(part, SystemPromptPart):
        return PartDetail(
            part_kind="system-prompt",
            content=str(part.content),
            timestamp=getattr(part, "timestamp", None),
            dynamic_ref=getattr(part, "dynamic_ref", None),
        )

    # UserPromptPart
    if isinstance(part, UserPromptPart):
        content = part.content
        if isinstance(content, str):
            content_str = content
        elif isinstance(content, Sequence) and not isinstance(content, str):
            # List of content items - join text parts
            text_parts = [str(item) for item in content if isinstance(item, str)]
            content_str = " ".join(text_parts) if text_parts else str(content)
        else:
            content_str = str(content)

        return PartDetail(
            part_kind="user-prompt",
            content=content_str,
            timestamp=getattr(part, "timestamp", None),
        )

    # TextPart
    if isinstance(part, TextPart):
        return PartDetail(
            part_kind="text",
            content=str(part.content),
            part_id=getattr(part, "id", None),
            provider_name=getattr(part, "provider_name", None),
            provider_details=getattr(part, "provider_details", None),
        )

    # ThinkingPart
    if ThinkingPart is not None and isinstance(part, ThinkingPart):
        return PartDetail(
            part_kind="thinking",
            content=str(getattr(part, "content", "")),
            signature=getattr(part, "signature", None),
            part_id=getattr(part, "id", None),
            provider_name=getattr(part, "provider_name", None),
            provider_details=getattr(part, "provider_details", None),
        )

    # ToolCallPart
    if isinstance(part, ToolCallPart):
        try:
            args_dict = part.args_as_dict()
        except Exception:
            args_dict = {}

        return PartDetail(
            part_kind="tool-call",
            content=f"{part.tool_name}(...)",
            tool_name=part.tool_name,
            tool_call_id=getattr(part, "tool_call_id", None),
            tool_args=args_dict if isinstance(args_dict, dict) else {},
            part_id=getattr(part, "id", None),
            provider_name=getattr(part, "provider_name", None),
            provider_details=getattr(part, "provider_details", None),
        )

    # ToolReturnPart
    if isinstance(part, ToolReturnPart):
        content = part.content
        content_str = str(content) if content is not None else ""

        return PartDetail(
            part_kind="tool-return",
            content=content_str,
            tool_name=getattr(part, "tool_name", None),
            tool_call_id=getattr(part, "tool_call_id", None),
            timestamp=getattr(part, "timestamp", None),
        )

    # RetryPromptPart
    if RetryPromptPart is not None and isinstance(part, RetryPromptPart):
        content = part.content
        if isinstance(content, str):
            content_str = content
        else:
            # List of ErrorDetails
            content_str = str(content)

        return PartDetail(
            part_kind="retry-prompt",
            content=content_str,
            tool_name=getattr(part, "tool_name", None),
            tool_call_id=getattr(part, "tool_call_id", None),
            timestamp=getattr(part, "timestamp", None),
        )

    # FilePart
    if FilePart is not None and isinstance(part, FilePart):
        return PartDetail(
            part_kind="file",
            content="<binary file content>",
            part_id=getattr(part, "id", None),
            provider_name=getattr(part, "provider_name", None),
            provider_details=getattr(part, "provider_details", None),
        )

    # BuiltinToolCallPart
    if BuiltinToolCallPart is not None and isinstance(part, BuiltinToolCallPart):
        try:
            args_dict = part.args_as_dict()
        except Exception:
            args_dict = {}

        return PartDetail(
            part_kind="builtin-tool-call",
            content=f"{part.tool_name}(...)",
            tool_name=part.tool_name,
            tool_call_id=getattr(part, "tool_call_id", None),
            tool_args=args_dict if isinstance(args_dict, dict) else {},
            part_id=getattr(part, "id", None),
            provider_name=getattr(part, "provider_name", None),
            provider_details=getattr(part, "provider_details", None),
        )

    # BuiltinToolReturnPart
    if BuiltinToolReturnPart is not None and isinstance(part, BuiltinToolReturnPart):
        content = getattr(part, "content", "")
        content_str = str(content) if content is not None else ""

        return PartDetail(
            part_kind="builtin-tool-return",
            content=content_str,
            tool_name=getattr(part, "tool_name", None),
            tool_call_id=getattr(part, "tool_call_id", None),
            timestamp=getattr(part, "timestamp", None),
            provider_name=getattr(part, "provider_name", None),
            provider_details=getattr(part, "provider_details", None),
        )

    # Unknown part type - fall through to generic
    return PartDetail(
        part_kind="unknown",
        content=str(part),
    )


def _extract_usage(usage: Any) -> UsageDetail:
    """Extract UsageDetail from a pydantic-ai RequestUsage object."""
    return UsageDetail(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_write_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_tokens", 0) or 0,
        input_audio_tokens=getattr(usage, "input_audio_tokens", 0) or 0,
        output_audio_tokens=getattr(usage, "output_audio_tokens", 0) or 0,
        cache_audio_read_tokens=getattr(usage, "cache_audio_read_tokens", 0) or 0,
        details=dict(getattr(usage, "details", {}) or {}),
    )


def _determine_role(parts: list[PartDetail]) -> str:
    """Determine the overall role based on part kinds."""
    if not parts:
        return "unknown"

    kinds = {p.part_kind for p in parts}

    # System prompt takes precedence
    if "system-prompt" in kinds:
        # If ONLY system prompt, role is system
        if kinds == {"system-prompt"}:
            return "system"
        # If system + user, still treat as system (bundled first message)
        if "user-prompt" in kinds:
            return "system"

    # Pure user prompt
    if kinds == {"user-prompt"}:
        return "user"

    # Pure tool returns
    if kinds <= {"tool-return", "retry-prompt", "builtin-tool-return"}:
        return "tool-return"

    # Assistant response (text, thinking, tool calls)
    if kinds <= {"text", "thinking", "tool-call", "builtin-tool-call", "file"}:
        return "assistant"

    # Mixed content
    if "user-prompt" in kinds:
        return "user"

    return "mixed"


def _generate_preview(parts: list[PartDetail], limit: int = 80) -> str:
    """Generate a short preview string from parts."""
    # Prefer text content for preview
    for part in parts:
        if part.part_kind == "text" and part.content:
            return _short_str(part.content, limit)
        if part.part_kind == "user-prompt" and part.content:
            return _short_str(part.content, limit)

    # Count tool calls
    tool_calls = [p for p in parts if p.part_kind in ("tool-call", "builtin-tool-call")]
    if tool_calls:
        if len(tool_calls) == 1:
            return f"<1 tool call: {tool_calls[0].tool_name}>"
        return f"<{len(tool_calls)} tool calls>"

    # Tool returns
    tool_returns = [
        p for p in parts if p.part_kind in ("tool-return", "builtin-tool-return")
    ]
    if tool_returns:
        if len(tool_returns) == 1:
            return f"<tool return: {tool_returns[0].tool_name}>"
        return f"<{len(tool_returns)} tool returns>"

    # System prompt
    for part in parts:
        if part.part_kind == "system-prompt" and part.content:
            return _short_str(part.content, limit)

    # Thinking
    for part in parts:
        if part.part_kind == "thinking" and part.content:
            return _short_str(f"[thinking] {part.content}", limit)

    # Fallback
    if parts:
        return f"<{len(parts)} part(s)>"
    return "<empty>"


def _extract_message(message: Any, history_index: int) -> InspectEntry | None:
    """Extract an InspectEntry from a pydantic-ai message object."""
    try:
        from pydantic_ai.messages import ModelRequest, ModelResponse
    except ImportError:
        # pydantic-ai not available
        return InspectEntry(
            history_index=history_index,
            kind="unknown",
            role="unknown",
            parts=[PartDetail(part_kind="unknown", content=str(message))],
            preview=_short_str(message),
        )

    parts_list: list[PartDetail] = []

    if isinstance(message, ModelRequest):
        # Extract all parts
        raw_parts = getattr(message, "parts", []) or []
        for raw_part in raw_parts:
            part_detail = _extract_part(raw_part)
            if part_detail is not None:
                parts_list.append(part_detail)

        role = _determine_role(parts_list)
        preview = _generate_preview(parts_list)

        # Count computed fields
        tool_call_count = sum(
            1 for p in parts_list if p.part_kind in ("tool-call", "builtin-tool-call")
        )
        thinking_count = sum(1 for p in parts_list if p.part_kind == "thinking")
        has_system = any(p.part_kind == "system-prompt" for p in parts_list)

        return InspectEntry(
            history_index=history_index,
            kind="request",
            role=role,
            parts=parts_list,
            preview=preview,
            timestamp=getattr(message, "timestamp", None),
            instructions=getattr(message, "instructions", None),
            run_id=getattr(message, "run_id", None),
            metadata=getattr(message, "metadata", None),
            tool_call_count=tool_call_count,
            thinking_block_count=thinking_count,
            has_system_prompt=has_system,
        )

    if isinstance(message, ModelResponse):
        # Extract all parts
        raw_parts = getattr(message, "parts", []) or []
        for raw_part in raw_parts:
            part_detail = _extract_part(raw_part)
            if part_detail is not None:
                parts_list.append(part_detail)

        role = _determine_role(parts_list)
        preview = _generate_preview(parts_list)

        # Extract usage
        raw_usage = getattr(message, "usage", None)
        usage = _extract_usage(raw_usage) if raw_usage else None

        # Count computed fields
        tool_call_count = sum(
            1 for p in parts_list if p.part_kind in ("tool-call", "builtin-tool-call")
        )
        thinking_count = sum(1 for p in parts_list if p.part_kind == "thinking")

        return InspectEntry(
            history_index=history_index,
            kind="response",
            role=role,
            parts=parts_list,
            preview=preview,
            timestamp=getattr(message, "timestamp", None),
            run_id=getattr(message, "run_id", None),
            metadata=getattr(message, "metadata", None),
            model_name=getattr(message, "model_name", None),
            provider_name=getattr(message, "provider_name", None),
            provider_url=getattr(message, "provider_url", None),
            provider_details=getattr(message, "provider_details", None),
            provider_response_id=getattr(message, "provider_response_id", None),
            finish_reason=getattr(message, "finish_reason", None),
            usage=usage,
            tool_call_count=tool_call_count,
            thinking_block_count=thinking_count,
            has_system_prompt=False,
        )

    # Unknown message type
    return InspectEntry(
        history_index=history_index,
        kind="unknown",
        role="unknown",
        parts=[PartDetail(part_kind="unknown", content=str(message))],
        preview=_short_str(message),
    )


def _apply_system_prompt(
    entry: InspectEntry, prev_instructions: str | None
) -> str | None:
    """Surface the system prompt as a real system-prompt part, exactly once.

    code-puppy stores the system prompt in ``ModelRequest.instructions`` rather
    than as a ``SystemPromptPart``, and pydantic-ai stamps ``instructions`` onto
    EVERY request it replays — including tool-return and follow-up user
    requests. Left untouched, the detail pane would render the full system
    prompt on a TRT (tool-return) message, which is the bug this guards against.

    Strategy:
    - The first time a given system prompt appears, synthesize a
      ``system-prompt`` part from it so it lives in a genuine system message
      (and recompute role/preview accordingly).
    - On every subsequent message carrying the same instructions, strip them so
      the system prompt never leaks into user / tool-return messages.
    - If the instructions change mid-conversation (e.g. model switch), the new
      prompt is surfaced again as its own system message.

    Args:
        entry: The entry to normalize (mutated in place).
        prev_instructions: The last system prompt already surfaced.

    Returns:
        The system prompt to remember for the next message's dedupe check.
    """
    instr = entry.instructions
    if not instr:
        return prev_instructions

    if instr == prev_instructions:
        # Already shown earlier — strip so it does not leak into this message.
        entry.instructions = None
        return prev_instructions

    # First occurrence (or a changed prompt) — surface it as a system part.
    if not any(p.part_kind == "system-prompt" for p in entry.parts):
        entry.parts.insert(0, PartDetail(part_kind="system-prompt", content=str(instr)))
        entry.has_system_prompt = True
        entry.role = _determine_role(entry.parts)
        entry.preview = _generate_preview(entry.parts)

    # Cleared so the detail pane shows it via the part, not a duplicate field.
    entry.instructions = None
    return instr


def build_inspect_entries(raw_history: list[Any]) -> list[InspectEntry]:
    """Build InspectEntry list from raw pydantic-ai message history.

    Unlike prune's build_message_entries(), this function:
    - Never skips messages (even pure system prompts are shown)
    - Never truncates content
    - Preserves all metadata and provider details
    - Surfaces the system prompt (carried in ``instructions``) once, as a real
      system message, instead of repeating it on every request

    Args:
        raw_history: List of pydantic-ai ModelRequest/ModelResponse objects

    Returns:
        List of InspectEntry objects, one per message, in order
    """
    entries: list[InspectEntry] = []
    last_instructions: str | None = None

    for idx, message in enumerate(raw_history):
        entry = _extract_message(message, idx)
        if entry is not None:
            last_instructions = _apply_system_prompt(entry, last_instructions)
            entries.append(entry)

    return entries


__all__ = [
    # Palette
    "C_ASSISTANT",
    "C_CURSOR",
    "C_DIM",
    "C_ERROR",
    "C_FILE",
    "C_FOOTER",
    "C_HEADER",
    "C_KEY",
    "C_LABEL",
    "C_RETRY",
    "C_SYSTEM",
    "C_THINKING",
    "C_TOOL",
    "C_TOOL_RETURN",
    "C_USER",
    "C_WARN",
    # Data classes
    "InspectEntry",
    "PartDetail",
    "UsageDetail",
    # Functions
    "build_inspect_entries",
    "classify_tool",
]
