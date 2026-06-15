"""Pure rendering functions for /inspect TUI.

This module contains pure functions that take InspectEntry data and return
prompt_toolkit formatted-text tuples. No state mutation, no TUI logic — just
data → visual representation.

Functions:
- render_list(): list pane with cursor, role badges, previews
- render_row(): single row in list pane
- render_detail(): detail pane with message metadata, parts list
- render_part_detail(): single part with all fields

Scrolling is handled via viewport-based slicing: caller passes scroll offset,
renderer slices visible lines and appends scroll indicators.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from .inspect_model import (
    C_ASSISTANT,
    C_CURSOR,
    C_DIM,
    C_FILE,
    C_HEADER,
    C_KEY,
    C_LABEL,
    C_RETRY,
    C_SYSTEM,
    C_THINKING,
    C_TOOL,
    C_TOOL_RETURN,
    C_USER,
    C_WARN,
    classify_tool,
)

if TYPE_CHECKING:
    from .inspect_model import InspectEntry, PartDetail

# Type alias for prompt_toolkit formatted text
FormattedText = list[tuple[str, str]]

# Width constants
LIST_PANE_MIN_WIDTH = 40
DETAIL_PANE_MIN_WIDTH = 50

# Detail-line semantic kinds. Every logical line the detail pane builds is
# tagged with one of these AT BUILD TIME — where the structural context is
# known — so styling never has to re-sniff prefixes or guess whether a colon
# belongs to a metadata key or a line of code in the content body.
K_BLANK = "blank"
K_MSG_HEADER = "msg_header"  # === USER (request) ===
K_RULE = "rule"  # decorative horizontal divider
K_SECTION = "section"  # PARTS (n) title
K_PART_HEADER = "part_header"  # [1/3] >>> tool-call: read_file
K_LABEL = "label"  # content:, args:, usage: section labels
K_KV = "kv"  # key: value metadata
K_CONTENT = "content"  # verbatim content body (never restyled/split)

# A detail row is (text, kind). render_part_detail() / _build_detail_lines()
# expose just the text for the clipboard + back-compat; render_detail() uses
# the kinds to colour each line.
DetailRow = tuple[str, str]


def _rows(kind: str, lines: list[str]) -> list[DetailRow]:
    """Tag a batch of plain lines with a single semantic kind."""
    return [(line, kind) for line in lines]


# -- helper functions --------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_tokens(count: int) -> str:
    """Format token count with thousands separator."""
    return f"{count:,}"


def _format_timestamp(ts: datetime | None) -> str:
    """Format timestamp for display."""
    if ts is None:
        return "—"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


# Characters that compose horizontal divider lines. Dividers are truncated
# (not wrapped) so they never spill a stray box-char onto the next line.
_DIVIDER_CHARS = frozenset("=-─")


def _is_divider(line: str) -> bool:
    """Return True if a line is a horizontal divider (repeated rule chars)."""
    stripped = line.strip()
    return len(stripped) >= 3 and all(c in _DIVIDER_CHARS for c in stripped)


def _wrap_one_line(body: str, width: int) -> list[str]:
    """Wrap a single string to ``width`` columns WITHOUT dropping a character.

    This is the zero-truncation guarantee in code form. ``textwrap.wrap`` is
    deliberately NOT used here because it silently drops whitespace at break
    points (e.g. ``"a   b"`` can lose the spaces) and collapses runs of
    whitespace — both of which destroy content in a tool whose whole job is to
    show EVERYTHING verbatim.

    The algorithm: walk the string in ``width``-sized windows. Prefer to break
    just *after* the last space in the window so words stay intact, but the
    space itself is KEPT on the line. If there is no usable break point (a long
    unbroken token), hard-break at ``width``. Every character of ``body`` ends
    up in exactly one returned piece, so ``"".join(result) == body``.

    Args:
        body: The text to wrap (no leading indent — caller re-adds it).
        width: Max columns per piece. Values <= 0 mean "do not wrap".

    Returns:
        List of pieces, each ``len(piece) <= width``, concatenating back to
        the original ``body`` exactly.
    """
    if width <= 0 or len(body) <= width:
        return [body]

    pieces: list[str] = []
    remaining = body
    while len(remaining) > width:
        window = remaining[:width]
        # Break after the last space so words stay whole, but keep the space.
        break_at = window.rfind(" ")
        if break_at <= 0:
            break_at = width  # unbroken token -> hard break, lose nothing
        else:
            break_at += 1  # retain the space on this line
        pieces.append(remaining[:break_at])
        remaining = remaining[break_at:]
    pieces.append(remaining)
    return pieces


def _wrap_detail_lines(lines: list[str], width: int) -> list[str]:
    """Wrap logical lines to fit ``width`` columns, preserving indentation.

    This is the heart of the detail-pane fix: prompt_toolkit's own line
    wrapping turns one long logical line into several *visual* lines, which
    breaks viewport-based scroll math (offsets count logical lines, the
    terminal renders visual lines). By wrapping here we guarantee
    1 logical line == 1 visual line, so scrolling stays accurate and content
    never overflows the pane.

    Zero truncation: wrapping is lossless (see ``_wrap_one_line``). No content
    character is ever dropped — long lines wrap, they do not get cut off.

    - Blank lines are preserved.
    - Divider lines are truncated, not wrapped (they are decorative repeated
      rule chars, never content).
    - Indented lines keep their indent on every continuation line, so the
      style-detection logic in render_detail() (which keys off line prefixes)
      keeps working and the structure stays readable.

    Args:
        lines: Logical text lines from _build_detail_lines().
        width: Target column width. Values <= 0 disable wrapping.

    Returns:
        A new list of lines, each no wider than ``width``.
    """
    return [text for text, _ in _wrap_detail_rows([(ln, K_CONTENT) for ln in lines], width)]


def _wrap_detail_rows(rows: list[DetailRow], width: int) -> list[DetailRow]:
    """Wrap (text, kind) rows to ``width``, carrying each kind onto every piece.

    Same lossless, indentation-preserving algorithm as _wrap_detail_lines (it
    delegates here) — the only addition is that a wrapped line's semantic kind
    rides along to all of its continuation pieces, so colouring survives a wrap.
    """
    if width <= 0:
        return rows

    wrapped: list[DetailRow] = []
    for text, kind in rows:
        if text == "":
            wrapped.append(("", kind))
            continue
        if len(text) <= width:
            wrapped.append((text, kind))
            continue
        if _is_divider(text):
            wrapped.append((text[:width], kind))
            continue

        indent = len(text) - len(text.lstrip(" "))
        indent_str = " " * indent
        body = text[indent:]
        avail = max(1, width - indent)
        pieces = _wrap_one_line(body, avail)
        wrapped.extend((indent_str + piece, kind) for piece in pieces)
    return wrapped


def _render_scroll_status(
    scroll_offset: int,
    content_height: int,
    total_lines: int,
    width: int = 0,
) -> FormattedText:
    """Render the persistent scroll/position status line for the detail pane.

    Always returns a single dim line so the user knows exactly where they are
    in the message:

    - ``▲ TOP   Lines 1-20 of 82 · 24%   ↓ below`` at the very top
    - ``Lines 21-40 of 82 · 49%   ↑ above   ↓ below`` in the middle
    - ``─── END OF MESSAGE ─── Lines 63-82 of 82 · 100%`` at the bottom

    Styling is intentionally muted (C_DIM) so it informs without distracting.

    Args:
        scroll_offset: First visible content line (0-based).
        content_height: Number of content lines shown (excludes this status
            line itself).
        total_lines: Total logical lines in the message.
        width: When > 0, the status line is truncated to this width so it never
            overflows the detail pane.

    Returns:
        A single-element formatted-text list (or empty if there is nothing).
    """
    if total_lines <= 0:
        return []

    start = scroll_offset + 1
    end = min(scroll_offset + content_height, total_lines)
    at_top = scroll_offset <= 0
    at_bottom = end >= total_lines
    pct = int(round(100 * end / total_lines))
    pos = f"Lines {start}-{end} of {total_lines} · {pct}%"

    if at_bottom:
        # Clear end-of-message banner (also covers the all-fits-on-screen case).
        text = f"─── END OF MESSAGE ─── {pos}"
        if not at_top:
            text += "   ↑ above"
    elif at_top:
        text = f"▲ TOP   {pos}   ↓ below"
    else:
        text = f"{pos}   ↑ above   ↓ below"

    if width > 0:
        text = _truncate(text, width)

    return [(C_DIM, text)]


def _maybe_parse_json(value: object) -> object:
    """Expand a JSON-encoded string into the real structure for pretty-printing.

    Tool args frequently arrive as a *string* containing JSON (e.g. a nested
    object serialized by the model). Showing that as one opaque blob defeats
    the whole point of a deep inspector, so when a string clearly looks like a
    JSON object/array we parse it and let _format_dict render it expanded.

    Conservative by design: only strings that start with ``{`` or ``[`` are
    even attempted, only dict/list results are unwrapped (scalars like
    ``"123"`` are left untouched), and any parse error falls back to the
    original value. Lossless either way — nothing is dropped.
    """
    if not isinstance(value, str):
        return value
    head = value.lstrip()[:1]
    if head not in ("{", "["):
        return value
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    return parsed if isinstance(parsed, (dict, list)) else value


def _format_dict(
    d: dict, indent: int = 2, _depth: int = 0, max_depth: int = 20
) -> list[str]:
    """Format a dict as indented YAML-like lines (JSON-string values expanded).

    Recursion is capped at *max_depth*. Tool arguments -- and the JSON we expand
    out of them via _maybe_parse_json -- are external, attacker-influenceable
    data; a pathologically deep payload would otherwise recurse until Python
    raises RecursionError and takes the inspector down. Past the cap we render a
    collapsed placeholder ({...} / [...]) instead of descending further: lossy
    only for absurd depth, and never a crash.
    """
    lines: list[str] = []
    prefix = " " * indent
    for key, value in d.items():
        value = _maybe_parse_json(value)
        if isinstance(value, dict):
            if _depth >= max_depth:
                lines.append(f"{prefix}{key}: {{...}}")
                continue
            lines.append(f"{prefix}{key}:")
            lines.extend(_format_dict(value, indent + 2, _depth + 1, max_depth))
        elif isinstance(value, list):
            if _depth >= max_depth:
                lines.append(f"{prefix}{key}: [...]")
                continue
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  -")
                    lines.extend(
                        _format_dict(item, indent + 4, _depth + 1, max_depth)
                    )
                else:
                    lines.append(f"{prefix}  - {item}")
        elif isinstance(value, str) and "\n" in value:
            # Multi-line string
            lines.append(f"{prefix}{key}: |")
            for line in value.split("\n"):
                lines.append(f"{prefix}  {line}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return lines


# -- row rendering -----------------------------------------------------------


def render_row(
    entry: InspectEntry,
    is_cursor: bool,
    width: int,
) -> FormattedText:
    """Render a single row in the list pane.

    Format: "▶ IDX  ROLE  PREVIEW..."
    - IDX: history_index, right-aligned 3 digits
    - ROLE: 3-char colored badge (USR, AST, SYS, TRT, MIX)
    - PREVIEW: truncated preview text

    Args:
        entry: The InspectEntry to render
        is_cursor: Whether this row has cursor focus
        width: Available width for the row

    Returns:
        Formatted text list for prompt_toolkit
    """
    result: FormattedText = []

    # Cursor indicator (2 chars: "▶ " or "  ")
    cursor_char = "▶ " if is_cursor else "  "
    cursor_style = C_CURSOR if is_cursor else ""
    result.append((cursor_style, cursor_char))

    # Index (3 chars + 2 space padding)
    result.append(("", f"{entry.history_index:3d}  "))

    # Role badge (3 chars + 2 space padding)
    result.append((entry.role_color, f"{entry.role_short}  "))

    # Preview (remaining width)
    preview_width = max(10, width - 2 - 5 - 5)  # cursor + idx + role
    preview = _truncate(entry.preview, preview_width)
    result.append(("", preview))

    # Newline
    result.append(("", "\n"))

    return result


# -- list pane rendering -----------------------------------------------------


def render_list(
    entries: list[InspectEntry],
    cursor_index: int,
    viewport_height: int,
    scroll_offset: int,
    width: int,
) -> FormattedText:
    """Render the list pane with cursor, role badges, and previews.

    Handles scrolling by slicing visible entries and showing scroll indicators.

    Args:
        entries: All InspectEntry objects
        cursor_index: Index of cursor in entries
        viewport_height: Number of visible rows (excluding header/footer)
        scroll_offset: First visible row index
        width: Available width for the pane

    Returns:
        Formatted text list for prompt_toolkit
    """
    result: FormattedText = []

    # Header line
    header_text = "/inspect"
    result.append((C_HEADER, _truncate(header_text, width - 1)))
    result.append(("", "\n"))

    # Separator
    result.append((C_DIM, "─" * min(width - 1, 60)))
    result.append(("", "\n"))

    if not entries:
        result.append((C_DIM, "  (no messages)"))
        result.append(("", "\n"))
        return result

    total_entries = len(entries)

    # Clamp scroll offset
    max_scroll = max(0, total_entries - viewport_height)
    scroll_offset = min(scroll_offset, max_scroll)
    scroll_offset = max(0, scroll_offset)

    # Scroll-up indicator
    if scroll_offset > 0:
        result.append((C_WARN, f"   ↑ {scroll_offset} more above"))
        result.append(("", "\n"))

    # Visible entries
    end_offset = scroll_offset + viewport_height
    visible_entries = entries[scroll_offset:end_offset]

    for i, entry in enumerate(visible_entries):
        actual_index = scroll_offset + i
        is_cursor = actual_index == cursor_index
        row = render_row(entry, is_cursor, width)
        result.extend(row)

    # Scroll-down indicator
    remaining_below = total_entries - end_offset
    if remaining_below > 0:
        result.append((C_WARN, f"   ↓ {remaining_below} more below"))
        result.append(("", "\n"))

    # Footer with cursor position
    result.append(("", "\n"))
    result.append((C_DIM, f"cursor {cursor_index + 1}/{total_entries}"))
    result.append(("", "\n"))

    return result


# -- part detail rendering ---------------------------------------------------


# Part kind to marker mapping for quick visual scanning
_PART_MARKERS: dict[str, str] = {
    "text": "TXT",
    "user-prompt": "USR",
    "system-prompt": "SYS",
    "tool-call": ">>>",
    "builtin-tool-call": ">>>",
    "tool-return": "<<<",
    "builtin-tool-return": "<<<",
    "thinking": "...",
    "file": "FIL",
    "retry-prompt": "RTY",
}


def _get_part_marker(part_kind: str) -> str:
    """Get the marker for a part kind, with fallback."""
    return _PART_MARKERS.get(part_kind, "---")


def _part_detail_rows(
    part: PartDetail,
    part_index: int,
    total_parts: int,
) -> list[DetailRow]:
    """Build a single part as (text, kind) rows.

    Thinking blocks are always shown fully expanded. Every line is tagged with
    its semantic kind so render_detail() can colour it without re-parsing.
    """
    rows: list[DetailRow] = []

    # Part divider (except for first part)
    if part_index > 0:
        rows.append(("─" * 35, K_RULE))
        rows.append(("", K_BLANK))

    # Part header with marker: [1/3] >>> tool-call: read_file
    marker = _get_part_marker(part.part_kind)
    part_num = f"[{part_index + 1}/{total_parts}]"
    if part.tool_name:
        part_header = f"{part_num} {marker} {part.part_kind}: {part.tool_name}"
        # Add tool classification marker for write/shell tools
        tool_marker = classify_tool(part.tool_name)
        part_header = f"{part_header} {tool_marker}"
    else:
        part_header = f"{part_num} {marker} {part.part_kind}"
    rows.append((part_header, K_PART_HEADER))
    rows.append(("", K_BLANK))  # Blank line after header for breathing room

    # Part metadata section (indented, grouped)
    metadata_lines: list[str] = []
    if part.tool_call_id:
        metadata_lines.append(f"    tool_call_id: {part.tool_call_id}")
    if part.part_id:
        metadata_lines.append(f"    part_id: {part.part_id}")
    if part.timestamp:
        metadata_lines.append(f"    timestamp: {_format_timestamp(part.timestamp)}")
    if part.signature:
        metadata_lines.append(f"    signature: {part.signature}")
    if part.dynamic_ref:
        metadata_lines.append(f"    dynamic_ref: {part.dynamic_ref}")
    if part.provider_name:
        metadata_lines.append(f"    provider: {part.provider_name}")

    if metadata_lines:
        rows.extend(_rows(K_KV, metadata_lines))
        rows.append(("", K_BLANK))  # Blank line after metadata

    # Provider details (nested dict)
    if part.provider_details:
        rows.append(("    provider_details:", K_LABEL))
        rows.extend(_rows(K_KV, _format_dict(part.provider_details, indent=6)))
        rows.append(("", K_BLANK))

    # Tool args (for tool-call parts) - pretty-printed with clear indentation
    if part.tool_args:
        rows.append(("    args:", K_LABEL))
        rows.extend(_rows(K_KV, _format_dict(part.tool_args, indent=6)))
        rows.append(("", K_BLANK))

    # Content section — thinking blocks are always fully expanded. Content is
    # tagged K_CONTENT so it renders verbatim in the default foreground and is
    # NEVER split on colons (a line of code is not a key/value pair).
    if part.content:
        rows.append(("    content:", K_LABEL))
        for content_line in part.content.split("\n"):
            rows.append((f"      {content_line}", K_CONTENT))

    rows.append(("", K_BLANK))  # Blank line after part
    return rows


def render_part_detail(
    part: PartDetail,
    part_index: int,
    total_parts: int,
) -> list[str]:
    """Render a single part with all fields as plain text lines.

    Thin wrapper over _part_detail_rows() that drops the semantic kinds — kept
    for the clipboard path and as the public, test-facing API.
    """
    return [text for text, _ in _part_detail_rows(part, part_index, total_parts)]


# -- detail pane rendering ---------------------------------------------------


def _detail_rows(
    entry: InspectEntry,
) -> list[DetailRow]:
    """Build all detail (text, kind) rows for an entry.

    Separated from render_detail() so scroll math + wrapping run on the logical
    lines, and so the kinds are assigned where structural context is known.
    """
    rows: list[DetailRow] = []

    # === MESSAGE HEADER ===
    role_label = entry.role.upper()
    rows.append((f"=== {role_label} ({entry.kind}) ===", K_MSG_HEADER))
    rows.append(("", K_BLANK))  # Blank line after header

    # --- Message Metadata ---
    rows.append((f"  history index: {entry.history_index}", K_KV))
    rows.append((f"  parts: {len(entry.parts)}", K_KV))

    if entry.timestamp:
        rows.append((f"  timestamp: {_format_timestamp(entry.timestamp)}", K_KV))

    # Response-specific metadata (grouped together)
    if entry.kind == "response":
        rows.append(("", K_BLANK))  # Blank line before response details
        if entry.model_name:
            rows.append((f"  model: {entry.model_name}", K_KV))
        if entry.provider_name:
            rows.append((f"  provider: {entry.provider_name}", K_KV))
        if entry.provider_url:
            rows.append((f"  provider_url: {entry.provider_url}", K_KV))
        if entry.provider_response_id:
            rows.append((f"  response_id: {entry.provider_response_id}", K_KV))
        if entry.finish_reason:
            rows.append((f"  finish_reason: {entry.finish_reason}", K_KV))

        # Usage stats (on its own line for readability)
        if entry.usage:
            rows.append(("", K_BLANK))  # Blank line before usage
            u = entry.usage
            rows.append(("  usage:", K_LABEL))
            rows.append((f"    input:  {_format_tokens(u.input_tokens)}", K_KV))
            rows.append((f"    output: {_format_tokens(u.output_tokens)}", K_KV))
            if u.has_cache:
                rows.append(
                    (f"    cache_read:  {_format_tokens(u.cache_read_tokens)}", K_KV)
                )
                rows.append(
                    (f"    cache_write: {_format_tokens(u.cache_write_tokens)}", K_KV)
                )
            if u.has_audio:
                rows.append(
                    (f"    audio_in:  {_format_tokens(u.input_audio_tokens)}", K_KV)
                )
                rows.append(
                    (f"    audio_out: {_format_tokens(u.output_audio_tokens)}", K_KV)
                )

    # Request-specific metadata
    if entry.kind == "request":
        if entry.instructions:
            rows.append(("", K_BLANK))  # Blank line before instructions
            rows.append((f"  instructions: {entry.instructions}", K_KV))

    if entry.run_id:
        rows.append(("", K_BLANK))  # Blank line before run_id
        rows.append((f"  run_id: {entry.run_id}", K_KV))

    if entry.metadata:
        rows.append(("", K_BLANK))  # Blank line before metadata dict
        rows.append(("  metadata:", K_LABEL))
        rows.extend(_rows(K_KV, _format_dict(entry.metadata, indent=4)))

    if entry.provider_details:
        rows.append(("", K_BLANK))  # Blank line before provider_details
        rows.append(("  provider_details:", K_LABEL))
        rows.extend(_rows(K_KV, _format_dict(entry.provider_details, indent=4)))

    # === PARTS SECTION ===
    rows.append(("", K_BLANK))  # Blank line before parts divider
    rows.append(("=" * 50, K_RULE))
    rows.append((f"  PARTS ({len(entry.parts)})", K_SECTION))
    rows.append(("=" * 50, K_RULE))
    rows.append(("", K_BLANK))  # Blank line after parts header

    # Render each part
    for i, part in enumerate(entry.parts):
        rows.extend(_part_detail_rows(part, i, len(entry.parts)))

    return rows


def _build_detail_lines(
    entry: InspectEntry,
) -> list[str]:
    """Plain-text view of the detail rows (clipboard + back-compat API)."""
    return [text for text, _ in _detail_rows(entry)]


# Role-header colours, longest-prefix-first so "=== TOOL-RETURN" never matches
# a shorter prefix by accident.
_MSG_HEADER_STYLES: tuple[tuple[str, str], ...] = (
    ("=== SYSTEM", C_SYSTEM),
    ("=== USER", C_USER),
    ("=== ASSISTANT", C_ASSISTANT),
    ("=== TOOL-RETURN", C_TOOL_RETURN),
)

# Part-header colours keyed by the part-kind substring. Order matters: the
# tool-call/return checks come before "text" so a tool literally named
# "...text..." can't steal the assistant colour.
_PART_HEADER_STYLES: tuple[tuple[str, str], ...] = (
    ("tool-call", C_TOOL),
    ("builtin-tool-call", C_TOOL),
    ("tool-return", C_TOOL_RETURN),
    ("builtin-tool-return", C_TOOL_RETURN),
    ("thinking", C_THINKING),
    ("system-prompt", C_SYSTEM),
    ("user-prompt", C_USER),
    ("retry-prompt", C_RETRY),
    ("file", C_FILE),
    ("text", C_ASSISTANT),
)


def _msg_header_style(text: str) -> str:
    for prefix, style in _MSG_HEADER_STYLES:
        if text.startswith(prefix):
            return style
    return "bold"


def _part_header_style(text: str) -> str:
    for needle, style in _PART_HEADER_STYLES:
        if needle in text:
            return style
    return "bold"


def _style_kv(text: str) -> FormattedText:
    """Colour a ``key: value`` line: key in C_KEY, value in the default fg.

    Default-foreground values are deliberate — bright and theme-safe, the
    opposite of the old muted-grey-everything look. If the line has no usable
    colon (e.g. a wrapped continuation piece) the whole thing renders as a
    value so nothing turns cyan by accident.
    """
    indent = len(text) - len(text.lstrip(" "))
    body = text[indent:]
    colon = body.find(":")
    value = body[colon + 1 :]  # keeps its own leading space
    # Only treat this as a real key:value line when the colon is immediately
    # followed by a space or end-of-line -- exactly how every metadata row and
    # _format_dict line is emitted ("key: value", "key:", "key: |"). This
    # rejects colons that live *inside* a value -- URLs (https://...),
    # timestamps (12:34:56) -- which can otherwise surface on a wrapped K_KV
    # continuation piece and mis-colour part of the value as a key.
    if colon <= 0 or (value and not value.startswith(" ")):
        return [("", text)]
    key = body[:colon]
    return [(C_KEY, " " * indent + key + ":"), ("", value)]


def _style_detail_line(text: str, kind: str) -> FormattedText:
    """Map one (text, kind) detail row to styled formatted-text segments.

    Returns segments WITHOUT a trailing newline so the caller controls line
    breaks (and the 1-logical-line == 1-visual-line scroll invariant holds).
    """
    if kind == K_BLANK or text == "":
        return [("", "")]
    if kind == K_MSG_HEADER:
        return [(_msg_header_style(text), text)]
    if kind == K_RULE:
        return [(C_DIM, text)]
    if kind == K_SECTION:
        return [(C_LABEL, text)]
    if kind == K_PART_HEADER:
        return [(_part_header_style(text), text)]
    if kind == K_LABEL:
        return [(C_LABEL, text)]
    if kind == K_KV:
        return _style_kv(text)
    # K_CONTENT and anything else: verbatim, default foreground, never split.
    return [("", text)]


def render_detail(
    entry: InspectEntry | None,
    viewport_height: int,
    scroll_offset: int,
    width: int = 0,
) -> tuple[FormattedText, int, int]:
    """Render the detail pane for an entry with scrolling support.

    Thinking blocks are always shown fully expanded.

    Args:
        entry: The InspectEntry to render (None if no selection)
        viewport_height: Number of visible lines
        scroll_offset: First visible line index
        width: Target column width for wrapping. <= 0 disables wrapping
            (handy for tests). Real callers pass the detail pane width so
            content wraps to the pane instead of the terminal edge.

    Returns:
        Tuple of (formatted_text, total_lines, clamped_scroll_offset)
        The clamped scroll offset allows caller to update state.
    """
    result: FormattedText = []

    if entry is None:
        result.append((C_DIM, "(no message selected)"))
        return result, 0, 0

    # Build tagged rows, then wrap them to the pane width so 1 logical line
    # maps to exactly 1 rendered line (keeps scroll math honest). The kind
    # rides along through the wrap so colouring survives a line break.
    all_rows = _wrap_detail_rows(_detail_rows(entry), width)
    total_lines = len(all_rows)

    # Reserve the last viewport row for the persistent scroll-status line so it
    # is ALWAYS visible (top marker, position, end-of-message banner). Content
    # therefore renders into one fewer row than the raw viewport height.
    content_height = max(1, viewport_height - 1)

    # Clamp scroll offset
    max_scroll = max(0, total_lines - content_height)
    scroll_offset = min(scroll_offset, max_scroll)
    scroll_offset = max(0, scroll_offset)

    # Slice viewport
    end_offset = scroll_offset + content_height
    visible_rows = all_rows[scroll_offset:end_offset]

    # Colour each line by its semantic kind (assigned at build time).
    for text, kind in visible_rows:
        result.extend(_style_detail_line(text, kind))
        result.append(("", "\n"))

    # Persistent scroll/position status line — always shown so the user knows
    # where they are, including a clear end-of-message banner at the bottom.
    status = _render_scroll_status(scroll_offset, content_height, total_lines, width)
    if status:
        result.append(("", "\n"))
        result.extend(status)

    return result, total_lines, scroll_offset


# -- convenience: format to plain text for clipboard -------------------------


def detail_to_plain_text(
    entry: InspectEntry,
) -> str:
    """Convert entry detail to plain text for clipboard copy.

    Thinking blocks are always included fully expanded.

    Args:
        entry: The InspectEntry to convert

    Returns:
        Plain text string suitable for clipboard
    """
    lines = _build_detail_lines(entry)
    return "\n".join(lines)


__all__ = [
    "FormattedText",
    "detail_to_plain_text",
    "render_detail",
    "render_list",
    "render_part_detail",
    "render_row",
]
