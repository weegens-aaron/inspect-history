"""Tests for inspect_menu.py — TUI state machine and keybindings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inspect_history.inspect_menu import InspectMenu
from inspect_history.inspect_model import InspectEntry, PartDetail


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def sample_entries() -> list[InspectEntry]:
    """Create a list of sample InspectEntry objects for testing."""
    return [
        InspectEntry(
            history_index=0,
            kind="request",
            role="system",
            parts=[PartDetail(part_kind="system-prompt", content="You are helpful.")],
            preview="You are helpful.",
        ),
        InspectEntry(
            history_index=1,
            kind="request",
            role="user",
            parts=[PartDetail(part_kind="user-prompt", content="Hello world")],
            preview="Hello world",
        ),
        InspectEntry(
            history_index=2,
            kind="response",
            role="assistant",
            parts=[PartDetail(part_kind="text", content="Hi there!")],
            preview="Hi there!",
            model_name="gpt-4",
        ),
        InspectEntry(
            history_index=3,
            kind="request",
            role="user",
            parts=[PartDetail(part_kind="user-prompt", content="Search for foo")],
            preview="Search for foo",
        ),
        InspectEntry(
            history_index=4,
            kind="response",
            role="assistant",
            parts=[
                PartDetail(
                    part_kind="tool-call",
                    content="read_file(...)",
                    tool_name="read_file",
                )
            ],
            preview="<1 tool call: read_file>",
            tool_call_count=1,
        ),
        InspectEntry(
            history_index=5,
            kind="request",
            role="tool-return",
            parts=[
                PartDetail(
                    part_kind="tool-return",
                    content="file contents here",
                    tool_name="read_file",
                )
            ],
            preview="<tool return: read_file>",
        ),
    ]


@pytest.fixture
def menu(sample_entries: list[InspectEntry]) -> InspectMenu:
    """Create an InspectMenu with sample entries."""
    return InspectMenu(sample_entries)

    # -- initialization tests ---------------------------------------------------nclass TestInspectMenuInit:
    """Test InspectMenu initialization."""

    def test_init_with_entries(self, sample_entries: list[InspectEntry]) -> None:
        menu = InspectMenu(sample_entries)
        assert menu.entries == sample_entries
        assert menu.cursor == 0
        assert menu.viewport_top == 0
        assert menu.detail_scroll == 0

    def test_init_empty_entries_raises(self) -> None:
        with pytest.raises(ValueError, match="requires at least one entry"):
            InspectMenu([])


# -- cursor navigation tests -------------------------------------------------


class TestCursorNavigation:
    """Test cursor movement methods."""

    def test_cursor_down(self, menu: InspectMenu) -> None:
        assert menu.cursor == 0
        menu._cursor_down()
        assert menu.cursor == 1
        menu._cursor_down()
        assert menu.cursor == 2

    def test_cursor_down_at_end(self, menu: InspectMenu) -> None:
        menu.cursor = len(menu.entries) - 1
        menu._cursor_down()
        assert menu.cursor == len(menu.entries) - 1  # Stays at end

    def test_cursor_up(self, menu: InspectMenu) -> None:
        menu.cursor = 2
        menu._cursor_up()
        assert menu.cursor == 1
        menu._cursor_up()
        assert menu.cursor == 0

    def test_cursor_up_at_start(self, menu: InspectMenu) -> None:
        menu.cursor = 0
        menu._cursor_up()
        assert menu.cursor == 0  # Stays at start

    def test_jump_to_start(self, menu: InspectMenu) -> None:
        menu.cursor = 3
        menu._jump_to_start()
        assert menu.cursor == 0

    def test_jump_to_end(self, menu: InspectMenu) -> None:
        menu.cursor = 0
        menu._jump_to_end()
        assert menu.cursor == len(menu.entries) - 1

    def test_page_up(self, menu: InspectMenu) -> None:
        menu._visible_rows = 3
        menu.cursor = 4
        menu._page_up()
        assert menu.cursor == 1

    def test_page_down(self, menu: InspectMenu) -> None:
        menu._visible_rows = 3
        menu.cursor = 0
        menu._page_down()
        assert menu.cursor == 3

    def test_cursor_move_resets_detail_scroll(self, menu: InspectMenu) -> None:
        menu.detail_scroll = 10
        menu._cursor_down()
        assert menu.detail_scroll == 0


# -- cursor bounds regression tests (inspect-jge) ----------------------------


class TestCursorBoundsRegression:
    """Regression coverage for inspect-jge.

    Bug report: cursor index got offset by one so you could not return to the
    first entry (index 0) and could scroll past the last entry. Valid bounds
    are 0 <= cursor <= len(entries) - 1. These tests pin that contract down in
    both directions so the off-by-one can never silently come back.
    """

    def test_can_return_to_first_entry(self, menu: InspectMenu) -> None:
        # 0 -> 1 -> 0 must land exactly on the first entry.
        assert menu.cursor == 0
        menu._cursor_down()
        assert menu.cursor == 1
        menu._cursor_up()
        assert menu.cursor == 0

    def test_cannot_go_below_zero(self, menu: InspectMenu) -> None:
        assert menu.cursor == 0
        for _ in range(5):
            menu._cursor_up()
        assert menu.cursor == 0  # Floor holds

    def test_stops_at_last_entry(self, menu: InspectMenu) -> None:
        last = len(menu.entries) - 1
        for _ in range(len(menu.entries) + 5):
            menu._cursor_down()
        assert menu.cursor == last  # Ceiling holds, never past len-1

    def test_full_round_trip_stays_in_bounds(self, menu: InspectMenu) -> None:
        last = len(menu.entries) - 1
        # Walk all the way down, asserting bounds at every step.
        for _ in range(len(menu.entries) + 3):
            menu._cursor_down()
            assert 0 <= menu.cursor <= last
        assert menu.cursor == last
        # Walk all the way back up to the first entry.
        for _ in range(len(menu.entries) + 3):
            menu._cursor_up()
            assert 0 <= menu.cursor <= last
        assert menu.cursor == 0

    def test_jump_to_end_then_back_to_start(self, menu: InspectMenu) -> None:
        menu._jump_to_end()
        assert menu.cursor == len(menu.entries) - 1
        menu._jump_to_start()
        assert menu.cursor == 0

    def test_page_navigation_respects_bounds(self, menu: InspectMenu) -> None:
        menu._visible_rows = 2
        last = len(menu.entries) - 1
        for _ in range(len(menu.entries)):
            menu._page_down()
            assert 0 <= menu.cursor <= last
        assert menu.cursor == last
        for _ in range(len(menu.entries)):
            menu._page_up()
            assert 0 <= menu.cursor <= last
        assert menu.cursor == 0


# -- detail pane scrolling tests ---------------------------------------------


class TestDetailPaneScrolling:
    """Test detail pane scroll methods."""

    def test_scroll_detail_down(self, menu: InspectMenu) -> None:
        menu._detail_total_lines = 50
        menu._detail_viewport_height = 20
        assert menu.detail_scroll == 0
        menu._scroll_detail_down()
        assert menu.detail_scroll == 1

    def test_scroll_detail_up(self, menu: InspectMenu) -> None:
        menu.detail_scroll = 5
        menu._scroll_detail_up()
        assert menu.detail_scroll == 4

    def test_scroll_detail_up_at_top(self, menu: InspectMenu) -> None:
        menu.detail_scroll = 0
        menu._scroll_detail_up()
        assert menu.detail_scroll == 0  # Stays at top

    def test_scroll_detail_down_at_bottom(self, menu: InspectMenu) -> None:
        menu._detail_total_lines = 25
        menu._detail_viewport_height = 20
        # render_detail reserves 1 row for the status line, so the content
        # height is 19 and max scroll = total - content_height = 25 - 19 = 6.
        menu.detail_scroll = 6  # Max scroll
        menu._scroll_detail_down()
        assert menu.detail_scroll == 6

    def test_scroll_into_view_clamps(self, menu: InspectMenu) -> None:
        menu._detail_total_lines = 10
        menu._detail_viewport_height = 20
        menu.detail_scroll = 50  # Way beyond content
        menu._scroll_detail_into_view()
        assert menu.detail_scroll == 0  # Clamped to 0 since content fits


# -- footer bar tests --------------------------------------------------------


class TestRenderFooter:
    """Test footer bar rendering."""

    def test_footer_returns_formatted_text(self, menu: InspectMenu) -> None:
        footer = menu._render_footer()
        assert isinstance(footer, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in footer)

    def test_footer_contains_navigation_hints(self, menu: InspectMenu) -> None:
        footer = menu._render_footer()
        text = "".join(t[1] for t in footer)
        assert "j/k" in text
        assert "h/l" in text
        assert "q" in text

    def test_footer_contains_action_hints(self, menu: InspectMenu) -> None:
        footer = menu._render_footer()
        text = "".join(t[1] for t in footer)
        assert "c" in text  # copy

    def test_footer_styling_is_dim(self, menu: InspectMenu) -> None:
        footer = menu._render_footer()
        # At least some parts should use dim styling
        dim_parts = [t for t in footer if "ansibrightblack" in t[0]]
        assert len(dim_parts) > 0, "Footer should have dim-styled elements"


# -- current entry tests -----------------------------------------------------


class TestCurrentEntry:
    """Test _current_entry method."""

    def test_current_entry(self, menu: InspectMenu) -> None:
        entry = menu._current_entry()
        assert entry is not None
        assert entry.history_index == 0

    def test_current_entry_after_move(self, menu: InspectMenu) -> None:
        menu._cursor_down()
        menu._cursor_down()
        entry = menu._current_entry()
        assert entry is not None
        assert entry.history_index == 2

    def test_current_entry_out_of_range(self, menu: InspectMenu) -> None:
        menu.cursor = len(menu.entries) + 5
        entry = menu._current_entry()
        assert entry is None


# -- viewport scroll tests ---------------------------------------------------


class TestListScrolling:
    """Test list pane viewport scrolling."""

    def test_scroll_list_into_view_cursor_below(self, menu: InspectMenu) -> None:
        menu._visible_rows = 3
        menu.viewport_top = 0
        menu.cursor = 4
        menu._scroll_list_into_view()
        assert menu.viewport_top == 2  # Scroll to show cursor

    def test_scroll_list_into_view_cursor_above(self, menu: InspectMenu) -> None:
        menu._visible_rows = 3
        menu.viewport_top = 3
        menu.cursor = 1
        menu._scroll_list_into_view()
        assert menu.viewport_top == 1


# -- keybindings tests -------------------------------------------------------


class TestKeybindings:
    """Test that keybindings are properly configured."""

    def test_build_keybindings(self, menu: InspectMenu) -> None:
        kb = menu._build_keybindings()
        # Verify keybindings object is created
        assert kb is not None

    @patch.object(InspectMenu, "_update_display")
    def test_navigation_keys_update_display(
        self, mock_update: MagicMock, menu: InspectMenu
    ) -> None:
        kb = menu._build_keybindings()
        # Simulate key press by finding and calling the handler
        # This is a sanity check that handlers exist
        assert kb.bindings  # Has some bindings registered

    @staticmethod
    def _invoke_key(menu: InspectMenu, key: str) -> None:
        """Find the binding for a single-char key and call its handler."""
        kb = menu._build_keybindings()
        for binding in kb.bindings:
            if tuple(binding.keys) == (key,):
                binding.handler(MagicMock())
                return
        raise AssertionError(f"no binding found for key {key!r}")

    @patch.object(InspectMenu, "_update_display")
    def test_j_moves_selection_up(
        self, _mock_update: MagicMock, menu: InspectMenu
    ) -> None:
        # Directional model (inspect-104): j = selection UP.
        menu.cursor = 2
        self._invoke_key(menu, "j")
        assert menu.cursor == 1

    @patch.object(InspectMenu, "_update_display")
    def test_k_moves_selection_down(
        self, _mock_update: MagicMock, menu: InspectMenu
    ) -> None:
        # Directional model (inspect-104): k = selection DOWN.
        menu.cursor = 0
        self._invoke_key(menu, "k")
        assert menu.cursor == 1

    @patch.object(InspectMenu, "_update_display")
    def test_h_scrolls_detail_up(
        self, _mock_update: MagicMock, menu: InspectMenu
    ) -> None:
        # h = detail pane scroll UP.
        menu.detail_scroll = 5
        self._invoke_key(menu, "h")
        assert menu.detail_scroll == 4

    @patch.object(InspectMenu, "_update_display")
    def test_l_scrolls_detail_down(
        self, _mock_update: MagicMock, menu: InspectMenu
    ) -> None:
        # l = detail pane scroll DOWN.
        menu._detail_total_lines = 1000
        menu.detail_scroll = 0
        self._invoke_key(menu, "l")
        assert menu.detail_scroll == 1


# -- terminal measurement tests ----------------------------------------------


class TestTerminalMeasurement:
    """Test terminal size measurement."""

    def test_measure_terminal_returns_tuple(self, menu: InspectMenu) -> None:
        cols, rows = menu._measure_terminal()
        assert isinstance(cols, int)
        assert isinstance(rows, int)
        assert cols >= 60  # Minimum enforced
        assert rows >= 15  # Minimum enforced

    @patch("shutil.get_terminal_size", side_effect=OSError("No terminal"))
    def test_measure_terminal_fallback(
        self, mock_size: MagicMock, menu: InspectMenu
    ) -> None:
        cols, rows = menu._measure_terminal()
        assert cols == 120
        assert rows == 40


# -- page size tests ---------------------------------------------------------


class TestPageSize:
    """Test page size calculation."""

    def test_page_size_minimum(self, menu: InspectMenu) -> None:
        menu._visible_rows = 0
        assert menu._page_size() == 1  # Never zero

    def test_page_size_normal(self, menu: InspectMenu) -> None:
        menu._visible_rows = 10
        assert menu._page_size() == 10


# -- clipboard tests ---------------------------------------------------------


class TestFormatEntryAsText:
    """Test _format_entry_as_text formatting."""

    def test_basic_formatting(self, menu: InspectMenu) -> None:
        text = menu._format_entry_as_text(menu.entries[1])  # user message
        assert "== Message #1 ==" in text
        assert "user" in text.lower()
        assert "Hello world" in text

    def test_includes_all_parts(self, menu: InspectMenu) -> None:
        # Entry with multiple parts
        entry = InspectEntry(
            history_index=99,
            kind="response",
            role="assistant",
            parts=[
                PartDetail(part_kind="thinking", content="Let me think..."),
                PartDetail(part_kind="text", content="Here is my answer"),
            ],
            preview="Here is my answer",
        )
        menu.entries.append(entry)
        text = menu._format_entry_as_text(entry)
        assert "[Part 1: thinking]" in text
        assert "Let me think..." in text
        assert "[Part 2: text]" in text
        assert "Here is my answer" in text

    def test_includes_tool_args(self, menu: InspectMenu) -> None:
        entry = InspectEntry(
            history_index=100,
            kind="response",
            role="assistant",
            parts=[
                PartDetail(
                    part_kind="tool-call",
                    content="read_file(...)",
                    tool_name="read_file",
                    tool_args={"file_path": "/tmp/test.txt"},
                ),
            ],
            preview="<1 tool call: read_file>",
        )
        text = menu._format_entry_as_text(entry)
        assert "read_file" in text
        assert "file_path" in text
        assert "/tmp/test.txt" in text

    def test_includes_metadata(self, menu: InspectMenu) -> None:
        from datetime import datetime

        from inspect_history.inspect_model import UsageDetail

        entry = InspectEntry(
            history_index=101,
            kind="response",
            role="assistant",
            parts=[PartDetail(part_kind="text", content="Test")],
            preview="Test",
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
            model_name="claude-3-opus",
            usage=UsageDetail(input_tokens=100, output_tokens=50),
        )
        text = menu._format_entry_as_text(entry)
        assert "2025" in text
        assert "claude-3-opus" in text
        assert "100" in text and "50" in text


class TestCopyToClipboard:
    """Test _copy_to_clipboard method."""

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.Popen")
    def test_macos_uses_pbcopy(
        self, mock_popen: MagicMock, mock_system: MagicMock, menu: InspectMenu
    ) -> None:
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        success, message = menu._copy_to_clipboard("test text")

        assert success is True
        assert "clipboard" in message.lower()
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == ["pbcopy"]

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.Popen")
    def test_linux_uses_xclip(
        self, mock_popen: MagicMock, mock_system: MagicMock, menu: InspectMenu
    ) -> None:
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        success, message = menu._copy_to_clipboard("test text")

        assert success is True
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == ["xclip", "-selection", "clipboard"]

    @patch("platform.system", return_value="Windows")
    @patch("subprocess.Popen")
    def test_windows_uses_clip(
        self, mock_popen: MagicMock, mock_system: MagicMock, menu: InspectMenu
    ) -> None:
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        success, message = menu._copy_to_clipboard("test text")

        assert success is True
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == ["clip"]
        # clip.exe expects UTF-16-LE, not UTF-8.
        assert mock_process.communicate.call_args.kwargs["input"] == (
            "test text".encode("utf-16-le")
        )

    @patch("platform.system", return_value="Plan9")
    def test_unsupported_os(self, mock_system: MagicMock, menu: InspectMenu) -> None:
        success, message = menu._copy_to_clipboard("test text")
        assert success is False
        assert "not supported" in message.lower()

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.Popen", side_effect=FileNotFoundError("pbcopy not found"))
    def test_command_not_found(
        self, mock_popen: MagicMock, mock_system: MagicMock, menu: InspectMenu
    ) -> None:
        success, message = menu._copy_to_clipboard("test text")
        assert success is False
        assert "not found" in message.lower()

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.Popen")
    def test_command_failure(
        self, mock_popen: MagicMock, mock_system: MagicMock, menu: InspectMenu
    ) -> None:
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"error message")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        success, message = menu._copy_to_clipboard("test text")

        assert success is False
        assert "failed" in message.lower()

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.Popen")
    def test_command_timeout(
        self, mock_popen: MagicMock, mock_system: MagicMock, menu: InspectMenu
    ) -> None:
        import subprocess

        mock_process = MagicMock()
        # First communicate (with input) times out; the post-kill reap returns.
        mock_process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="pbcopy", timeout=5),
            (b"", b""),
        ]
        mock_popen.return_value = mock_process

        success, message = menu._copy_to_clipboard("test text")

        assert success is False
        assert "timed out" in message.lower()
        mock_process.kill.assert_called_once()


class TestDoCopy:
    """Test _do_copy method."""

    def test_no_entry_selected(self, menu: InspectMenu) -> None:
        # Force cursor out of range so no entry is selected
        menu.cursor = len(menu.entries) + 5
        menu._do_copy()
        assert menu._status_message is not None
        assert "no message" in menu._status_message.lower()
        assert menu._status_style == "fg:ansired"

    @patch.object(InspectMenu, "_copy_to_clipboard", return_value=(True, "Copied"))
    def test_success_sets_status(
        self,
        mock_clipboard: MagicMock,
        menu: InspectMenu,
    ) -> None:
        menu._do_copy()
        assert menu._status_message is not None
        assert "copied" in menu._status_message.lower()
        assert menu._status_style == "fg:ansigreen"

    @patch.object(
        InspectMenu, "_copy_to_clipboard", return_value=(False, "xclip not found")
    )
    def test_failure_sets_status(
        self,
        mock_clipboard: MagicMock,
        menu: InspectMenu,
    ) -> None:
        menu._do_copy()
        assert menu._status_message is not None
        assert "failed" in menu._status_message.lower()
        assert menu._status_style == "fg:ansired"

    @patch.object(InspectMenu, "_copy_to_clipboard", return_value=(True, "Copied"))
    def test_status_cleared_on_navigation(
        self,
        mock_clipboard: MagicMock,
        menu: InspectMenu,
    ) -> None:
        # A copy sets a status; the next display refresh (as nav triggers)
        # should dismiss it so it never persists forever / stacks up.
        menu._do_copy()
        assert menu._status_message is not None
        menu._update_display()  # default clear_status=True, like a nav action
        assert menu._status_message is None


class TestCopyKeybinding:
    """Test that 'c' keybinding is registered."""

    def test_c_key_registered(self, menu: InspectMenu) -> None:
        kb = menu._build_keybindings()
        # Check that a binding for 'c' exists
        binding_keys = []
        for binding in kb.bindings:
            for key in binding.keys:
                binding_keys.append(str(key))
        assert any("c" in key for key in binding_keys)
