"""Unit tests for inspect_render.py."""

from datetime import datetime

import pytest

from inspect_history.inspect_model import InspectEntry, PartDetail, UsageDetail
from inspect_history.inspect_render import (
    detail_to_plain_text,
    render_detail,
    render_list,
    render_part_detail,
    render_row,
)
from inspect_history.inspect_render import (
    _wrap_detail_lines,
    _wrap_one_line,
)


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def user_entry():
    """Simple user message entry."""
    return InspectEntry(
        history_index=0,
        kind="request",
        role="user",
        parts=[
            PartDetail(
                part_kind="user-prompt",
                content="Hello, how are you?",
            )
        ],
        preview="Hello, how are you?",
        timestamp=datetime(2026, 6, 11, 12, 0, 0),
    )


@pytest.fixture
def assistant_entry():
    """Simple assistant text response."""
    return InspectEntry(
        history_index=1,
        kind="response",
        role="assistant",
        parts=[
            PartDetail(
                part_kind="text",
                content="I'm doing well, thank you for asking!",
            )
        ],
        preview="I'm doing well, thank you for asking!",
        timestamp=datetime(2026, 6, 11, 12, 0, 5),
        model_name="gpt-4o",
        provider_name="openai",
        finish_reason="stop",
        usage=UsageDetail(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
        ),
    )


@pytest.fixture
def tool_call_entry():
    """Assistant response with tool calls."""
    return InspectEntry(
        history_index=2,
        kind="response",
        role="assistant",
        parts=[
            PartDetail(
                part_kind="tool-call",
                content="read_file(...)",
                tool_name="read_file",
                tool_call_id="tc_001",
                tool_args={"file_path": "/test.py", "num_lines": 100},
            ),
            PartDetail(
                part_kind="tool-call",
                content="grep(...)",
                tool_name="grep",
                tool_call_id="tc_002",
                tool_args={"search_string": "def main", "directory": "."},
            ),
        ],
        preview="<2 tool calls>",
        tool_call_count=2,
    )


@pytest.fixture
def thinking_entry():
    """Assistant response with thinking block."""
    return InspectEntry(
        history_index=3,
        kind="response",
        role="assistant",
        parts=[
            PartDetail(
                part_kind="thinking",
                content="Let me analyze this problem step by step...",
                signature="sig_abc123",
            ),
            PartDetail(
                part_kind="text",
                content="Here's my answer.",
            ),
        ],
        preview="Here's my answer.",
        thinking_block_count=1,
    )


@pytest.fixture
def system_entry():
    """System prompt entry."""
    return InspectEntry(
        history_index=0,
        kind="request",
        role="system",
        parts=[
            PartDetail(
                part_kind="system-prompt",
                content="You are a helpful assistant.",
                dynamic_ref="default_system",
            )
        ],
        preview="You are a helpful assistant.",
        has_system_prompt=True,
    )


@pytest.fixture
def tool_return_entry():
    """Tool return entry."""
    return InspectEntry(
        history_index=4,
        kind="request",
        role="tool-return",
        parts=[
            PartDetail(
                part_kind="tool-return",
                content='{"success": true, "data": "file contents here"}',
                tool_name="read_file",
                tool_call_id="tc_001",
            )
        ],
        preview="<tool return: read_file>",
    )


@pytest.fixture
def sample_entries(user_entry, assistant_entry, tool_call_entry):
    """List of sample entries for list rendering tests."""
    return [user_entry, assistant_entry, tool_call_entry]


# -- render_row tests --------------------------------------------------------


class TestRenderRow:
    """Tests for render_row()."""

    def test_basic_row(self, user_entry):
        """Test basic row rendering."""
        result = render_row(user_entry, is_cursor=False, width=80)

        # Result should be a list of (style, text) tuples
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

        # Flatten to text
        text = "".join(t[1] for t in result)

        # Should contain index
        assert "  0" in text
        # Should contain role badge
        assert "USR" in text
        # Should contain preview
        assert "Hello" in text
        # Should end with newline
        assert text.endswith("\n")

    def test_cursor_row(self, user_entry):
        """Test row with cursor."""
        result = render_row(user_entry, is_cursor=True, width=80)
        text = "".join(t[1] for t in result)

        # Should have cursor indicator
        assert "▶" in text

    def test_non_cursor_row(self, user_entry):
        """Test row without cursor."""
        result = render_row(user_entry, is_cursor=False, width=80)
        text = "".join(t[1] for t in result)

        # Should not have cursor indicator
        assert "▶" not in text

    def test_role_badges(self, user_entry, assistant_entry, system_entry):
        """Test different role badges."""
        user_text = "".join(t[1] for t in render_row(user_entry, False, 80))
        assistant_text = "".join(t[1] for t in render_row(assistant_entry, False, 80))
        system_text = "".join(t[1] for t in render_row(system_entry, False, 80))

        assert "USR" in user_text
        assert "AST" in assistant_text
        assert "SYS" in system_text

    def test_preview_truncation(self):
        """Test that long previews are truncated."""
        entry = InspectEntry(
            history_index=0,
            kind="request",
            role="user",
            preview="x" * 200,
        )
        result = render_row(entry, False, width=50)
        text = "".join(t[1] for t in result)

        # Preview should be truncated
        assert "…" in text


# -- render_list tests -------------------------------------------------------


class TestRenderList:
    """Tests for render_list()."""

    def test_empty_list(self):
        """Test rendering empty entry list."""
        result = render_list(
            entries=[],
            cursor_index=0,
            viewport_height=20,
            scroll_offset=0,
            width=80,
        )

        text = "".join(t[1] for t in result)
        assert "(no messages)" in text

    def test_header_contains_inspect(self, sample_entries):
        """Test that header shows /inspect."""
        result = render_list(
            entries=sample_entries,
            cursor_index=0,
            viewport_height=20,
            scroll_offset=0,
            width=80,
        )

        text = "".join(t[1] for t in result)
        assert "/inspect" in text

    def test_cursor_position_shown(self, sample_entries):
        """Test cursor position indicator."""
        result = render_list(
            entries=sample_entries,
            cursor_index=1,
            viewport_height=20,
            scroll_offset=0,
            width=80,
        )

        text = "".join(t[1] for t in result)
        assert "cursor 2/3" in text  # 1-indexed display

    def test_scroll_indicator_up(self, sample_entries):
        """Test scroll-up indicator when content is scrolled."""
        result = render_list(
            entries=sample_entries,
            cursor_index=2,
            viewport_height=1,  # Only 1 visible
            scroll_offset=2,  # Scrolled down 2
            width=80,
        )

        text = "".join(t[1] for t in result)
        assert "↑" in text
        assert "2 more above" in text

    def test_scroll_indicator_down(self, sample_entries):
        """Test scroll-down indicator when more content below."""
        result = render_list(
            entries=sample_entries,
            cursor_index=0,
            viewport_height=1,  # Only 1 visible
            scroll_offset=0,
            width=80,
        )

        text = "".join(t[1] for t in result)
        assert "↓" in text
        assert "2 more below" in text

    def test_all_entries_visible(self, sample_entries):
        """Test no scroll indicators when all entries visible."""
        result = render_list(
            entries=sample_entries,
            cursor_index=0,
            viewport_height=20,  # Plenty of room
            scroll_offset=0,
            width=80,
        )

        text = "".join(t[1] for t in result)
        assert "more above" not in text
        assert "more below" not in text


# -- render_part_detail tests ------------------------------------------------


class TestRenderPartDetail:
    """Tests for render_part_detail()."""

    def test_text_part(self):
        """Test rendering a text part."""
        part = PartDetail(
            part_kind="text",
            content="Hello world",
        )
        lines = render_part_detail(part, 0, 1)

        # Header should include marker and part kind
        assert any(
            "[1/1]" in line and "TXT" in line and "text" in line for line in lines
        )
        assert any("Hello world" in line for line in lines)

    def test_tool_call_part(self):
        """Test rendering a tool call part."""
        part = PartDetail(
            part_kind="tool-call",
            content="read_file(...)",
            tool_name="read_file",
            tool_call_id="tc_123",
            tool_args={"file_path": "/test.py"},
        )
        lines = render_part_detail(part, 0, 2)

        assert any("tool-call: read_file" in line for line in lines)
        assert any("tool_call_id: tc_123" in line for line in lines)
        assert any("file_path:" in line for line in lines)

    def test_tool_classification_icons(self):
        """Test tool classification icons appear."""
        write_part = PartDetail(
            part_kind="tool-call",
            content="create_file(...)",
            tool_name="create_file",
            tool_call_id="tc_1",
        )
        shell_part = PartDetail(
            part_kind="tool-call",
            content="agent_run_shell_command(...)",
            tool_name="agent_run_shell_command",
            tool_call_id="tc_2",
        )

        write_lines = render_part_detail(write_part, 0, 1)
        shell_lines = render_part_detail(shell_part, 0, 1)

        assert any("[W]" in line for line in write_lines)
        assert any("[!]" in line for line in shell_lines)

    def test_thinking_always_expanded(self):
        """Test thinking blocks are always shown fully expanded."""
        part = PartDetail(
            part_kind="thinking",
            content="Deep thoughts here...",
            signature="sig_abc",
        )
        lines = render_part_detail(part, 0, 1)

        assert any("Deep thoughts here" in line for line in lines)
        assert not any("chars collapsed" in line for line in lines)

    def test_part_with_provider_details(self):
        """Test part with provider details."""
        part = PartDetail(
            part_kind="text",
            content="Hello",
            provider_name="anthropic",
            provider_details={"model_version": "2024-01"},
        )
        lines = render_part_detail(part, 0, 1)

        assert any("provider: anthropic" in line for line in lines)
        assert any("model_version" in line for line in lines)


# -- render_detail tests -----------------------------------------------------


class TestRenderDetail:
    """Tests for render_detail()."""

    def test_none_entry(self):
        """Test rendering when no entry selected."""
        result, total, offset = render_detail(
            entry=None,
            viewport_height=20,
            scroll_offset=0,
        )

        text = "".join(t[1] for t in result)
        assert "(no message selected)" in text
        assert total == 0
        assert offset == 0

    def test_user_entry_header(self, user_entry):
        """Test user entry header."""
        result, total, offset = render_detail(
            entry=user_entry,
            viewport_height=100,
            scroll_offset=0,
        )

        text = "".join(t[1] for t in result)
        assert "USER" in text
        assert "(request)" in text

    def test_assistant_entry_header(self, assistant_entry):
        """Test assistant entry header."""
        result, total, offset = render_detail(
            entry=assistant_entry,
            viewport_height=100,
            scroll_offset=0,
        )

        text = "".join(t[1] for t in result)
        assert "ASSISTANT" in text
        assert "(response)" in text

    def test_metadata_shown(self, assistant_entry):
        """Test response metadata is shown."""
        result, total, offset = render_detail(
            entry=assistant_entry,
            viewport_height=100,
            scroll_offset=0,
        )

        text = "".join(t[1] for t in result)
        assert "model: gpt-4o" in text
        assert "provider: openai" in text
        assert "finish_reason: stop" in text

    def test_usage_shown(self, assistant_entry):
        """Test usage stats are shown."""
        result, total, offset = render_detail(
            entry=assistant_entry,
            viewport_height=100,
            scroll_offset=0,
        )

        text = "".join(t[1] for t in result)
        assert "usage:" in text
        assert "100" in text  # input tokens
        assert "50" in text  # output tokens

    def test_scroll_clamp(self, assistant_entry):
        """Test scroll offset is clamped."""
        _, total, offset = render_detail(
            entry=assistant_entry,
            viewport_height=100,
            scroll_offset=1000,  # Way too high
        )

        # Should be clamped to max valid offset
        assert offset <= max(0, total - 100)

    def test_scroll_indicators(self, tool_call_entry):
        """Test scroll indicators when content exceeds viewport."""
        result, total, offset = render_detail(
            entry=tool_call_entry,
            viewport_height=5,  # Very small viewport
            scroll_offset=2,  # Some scroll
        )

        text = "".join(t[1] for t in result)
        # Should show indicators
        assert "↑" in text or "↓" in text

    def test_position_status_always_shown(self, tool_call_entry):
        """Position status (Lines X-Y of Z · NN%) is always present."""
        result, total, _ = render_detail(
            entry=tool_call_entry,
            viewport_height=5,
            scroll_offset=1,
        )
        text = "".join(t[1] for t in result)
        assert "Lines " in text
        assert f"of {total}" in text
        assert "%" in text

    def test_top_marker_at_top(self, tool_call_entry):
        """A TOP marker appears when scrolled to the very top with more below."""
        result, _, _ = render_detail(
            entry=tool_call_entry,
            viewport_height=5,  # small enough that content overflows
            scroll_offset=0,
        )
        text = "".join(t[1] for t in result)
        assert "TOP" in text
        assert "↓ below" in text

    def test_end_of_message_banner_at_bottom(self, tool_call_entry):
        """END OF MESSAGE banner appears when scrolled to the bottom."""
        result, total, _ = render_detail(
            entry=tool_call_entry,
            viewport_height=5,
            scroll_offset=10_000,  # clamp to bottom
        )
        text = "".join(t[1] for t in result)
        assert "END OF MESSAGE" in text
        assert "100%" in text

    def test_end_banner_when_everything_fits(self, user_entry):
        """When all content fits on screen, still show the end banner."""
        result, _, _ = render_detail(
            entry=user_entry,
            viewport_height=1000,
            scroll_offset=0,
        )
        text = "".join(t[1] for t in result)
        assert "END OF MESSAGE" in text

    def test_status_respects_width(self):
        """The status line must not overflow the requested pane width."""
        long_content = "word " * 200
        entry = InspectEntry(
            history_index=0,
            kind="request",
            role="user",
            parts=[PartDetail(part_kind="user-prompt", content=long_content)],
            preview="words",
        )
        width = 40
        result, _, _ = render_detail(
            entry=entry,
            viewport_height=10,
            scroll_offset=0,
            width=width,
        )
        for _style, text in result:
            if text == "\n":
                continue
            for visual_line in text.split("\n"):
                assert len(visual_line) <= width, repr(visual_line)

    def test_parts_separator(self, user_entry):
        """Test parts section separator."""
        result, _, _ = render_detail(
            entry=user_entry,
            viewport_height=100,
            scroll_offset=0,
        )

        text = "".join(t[1] for t in result)
        assert "parts" in text.lower()

    def test_returns_total_lines(self, assistant_entry):
        """Test that total line count is returned."""
        _, total, _ = render_detail(
            entry=assistant_entry,
            viewport_height=100,
            scroll_offset=0,
        )

        assert total > 0

    def test_thinking_always_expanded(self, thinking_entry):
        """Test thinking blocks are always rendered fully expanded."""
        expanded, _, _ = render_detail(
            entry=thinking_entry,
            viewport_height=100,
            scroll_offset=0,
        )

        expanded_text = "".join(t[1] for t in expanded)

        assert "analyze this problem" in expanded_text
        assert "chars collapsed" not in expanded_text

    def test_width_wraps_long_lines(self):
        """Long content must wrap to the pane width, not overflow it."""
        long_content = (
            "This is a very long line of content that absolutely must be "
            "wrapped so it does not overflow the detail pane and create the "
            "ghost-text and overlapping artifacts described in inspect-dvi."
        )
        entry = InspectEntry(
            history_index=0,
            kind="request",
            role="user",
            parts=[PartDetail(part_kind="user-prompt", content=long_content)],
            preview=long_content[:30],
        )
        width = 40
        result, total, _ = render_detail(
            entry=entry,
            viewport_height=1000,
            scroll_offset=0,
            width=width,
        )

        # No rendered line may exceed the requested width.
        for style, text in result:
            if text == "\n":
                continue
            for visual_line in text.split("\n"):
                assert len(visual_line) <= width, repr(visual_line)

    def test_width_zero_disables_wrapping(self):
        """width<=0 leaves long lines intact (test/back-compat path)."""
        long_content = "x" * 200
        entry = InspectEntry(
            history_index=0,
            kind="request",
            role="user",
            parts=[PartDetail(part_kind="user-prompt", content=long_content)],
            preview="x",
        )
        result, _, _ = render_detail(
            entry=entry,
            viewport_height=1000,
            scroll_offset=0,
            width=0,
        )
        text = "".join(t[1] for t in result)
        assert long_content in text

    def test_wrapping_increases_total_lines(self):
        """Wrapping a wide entry should produce more lines than no-wrap."""
        long_content = "word " * 100
        entry = InspectEntry(
            history_index=0,
            kind="request",
            role="user",
            parts=[PartDetail(part_kind="user-prompt", content=long_content)],
            preview="words",
        )
        _, unwrapped_total, _ = render_detail(
            entry=entry, viewport_height=1000, scroll_offset=0, width=0
        )
        _, wrapped_total, _ = render_detail(
            entry=entry, viewport_height=1000, scroll_offset=0, width=30
        )
        assert wrapped_total > unwrapped_total


# -- lossless wrapping (zero-truncation) tests -------------------------------


class TestLosslessWrapping:
    """Wrapping must never drop a single character (inspect-rtp)."""

    @pytest.mark.parametrize(
        "body",
        [
            "a    b   verylongwordhere    trailing   ",
            "   leading spaces preserved across the wrap boundary too",
            "trailing whitespace at the very end must survive          ",
            "x" * 200,  # single unbroken token
            "word " * 50,  # many short words + spaces
            "\ttabs\tand    runs   of  whitespace\tmust\tnot\tcollapse",
            "short",  # shorter than width -> returned unchanged
            "",  # empty
        ],
    )
    def test_wrap_one_line_is_lossless(self, body):
        """Joining the pieces must reproduce the input byte-for-byte."""
        width = 13
        pieces = _wrap_one_line(body, width)
        assert "".join(pieces) == body
        for piece in pieces:
            assert len(piece) <= width

    def test_wrap_one_line_no_wrap_when_width_zero(self):
        assert _wrap_one_line("anything at all", 0) == ["anything at all"]

    def test_wrap_detail_lines_preserves_all_content(self):
        """Stripping the re-added indent must recover the original content."""
        lines = [
            "      a    b   verylongwordhere    trailing   ",
            "      " + "z" * 100,
            "      normal short line",
            "",
        ]
        width = 20
        wrapped = _wrap_detail_lines(lines, width)
        # Reconstruct: continuation lines all carry the 6-space indent.
        reconstructed = []
        buf = ""
        for wl in wrapped:
            if wl == "":
                if buf:
                    reconstructed.append(buf)
                    buf = ""
                reconstructed.append("")
                continue
            buf += wl[6:] if wl.startswith("      ") else wl
        if buf:
            reconstructed.append(buf)
        # The non-blank original bodies (sans indent) must all be present whole.
        joined = "".join(reconstructed)
        assert "a    b   verylongwordhere    trailing   " in joined
        assert "z" * 100 in joined
        assert "normal short line" in joined
        for wl in wrapped:
            assert len(wl) <= width

    def test_render_detail_does_not_drop_inner_whitespace(self):
        """End-to-end: weird whitespace content survives render_detail wrapping."""
        content = "alpha    beta   gamma_is_a_really_long_token_here    delta"
        entry = InspectEntry(
            history_index=0,
            kind="request",
            role="user",
            parts=[PartDetail(part_kind="user-prompt", content=content)],
            preview="alpha",
        )
        result, _, _ = render_detail(
            entry=entry, viewport_height=1000, scroll_offset=0, width=24
        )
        # Strip the 6-space content indent and rejoin to confirm nothing
        # between tokens was eaten by the wrapper.
        rendered = "".join(t[1] for t in result if t[1] != "\n")
        compact = "".join(rendered.split())
        assert "alphabetagamma_is_a_really_long_token_heredelta" in compact


# -- detail_to_plain_text tests ----------------------------------------------


class TestDetailToPlainText:
    """Tests for detail_to_plain_text()."""

    def test_returns_string(self, user_entry):
        """Test that result is a string."""
        result = detail_to_plain_text(user_entry)
        assert isinstance(result, str)

    def test_contains_content(self, user_entry):
        """Test content is included."""
        result = detail_to_plain_text(user_entry)

        assert "USER" in result
        assert "Hello" in result

    def test_thinking_always_expanded(self, thinking_entry):
        """Test thinking content is always fully included."""
        expanded = detail_to_plain_text(thinking_entry)

        assert "analyze this problem" in expanded
        assert "chars collapsed" not in expanded
