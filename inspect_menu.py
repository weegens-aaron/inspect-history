"""Interactive TUI for /inspect — read-only deep inspector.

This is the state machine that handles user input and coordinates rendering.
Split-pane layout: list (left) shows message summaries with cursor navigation,
detail (right) shows full message content with all parts expanded.

State:
    entries: list[InspectEntry] — all messages in conversation history
    cursor: int — current row in list pane
    viewport_top: int — list pane scroll offset
    detail_scroll: int — detail pane scroll offset

Read-only: /inspect never modifies history. No confirmation on exit.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Frame

from .inspect_model import InspectEntry
from .inspect_render import render_detail, render_list

try:
    from code_puppy.messaging import emit_info, emit_warning
except ImportError:
    # Fallback for testing without full code_puppy environment
    def emit_info(msg: str) -> None:
        print(f"[INFO] {msg}")

    def emit_warning(msg: str) -> None:
        print(f"[WARN] {msg}")


if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent


class InspectMenu:
    """prompt_toolkit split-panel TUI for /inspect."""

    def __init__(self, entries: list[InspectEntry]) -> None:
        if not entries:
            raise ValueError("InspectMenu requires at least one entry")

        self.entries = entries

        # Cursor and list viewport state
        self.cursor: int = 0
        self.viewport_top: int = 0

        # Detail pane scroll state
        self.detail_scroll: int = 0
        self._detail_total_lines: int = 0

        # Viewport dimensions — set in run() once terminal size is known
        self._visible_rows: int = 20
        self._detail_viewport_height: int = 20
        self._list_width: int = 40
        self._detail_width: int = 60

        # prompt_toolkit controls
        self.list_control: FormattedTextControl | None = None
        self.detail_control: FormattedTextControl | None = None
        self.footer_control: FormattedTextControl | None = None

    # ── entries ───────────────────────────────────────────────────────────

    def _current_entry(self) -> InspectEntry | None:
        """Return the entry under the cursor, or None if list is empty."""
        if not self.entries or self.cursor >= len(self.entries):
            return None
        return self.entries[self.cursor]

    # ── viewport / pagination ─────────────────────────────────────────────

    def _page_size(self) -> int:
        """Number of rows that fit in the list pane."""
        return max(1, self._visible_rows)

    def _detail_content_height(self) -> int:
        """Detail rows available for content.

        render_detail() reserves the last viewport row for the persistent
        scroll-status line, so scroll math here must reserve it too — otherwise
        the user could ``scroll past`` the renderable content.
        """
        return max(1, self._detail_viewport_height - 1)

    def _scroll_list_into_view(self) -> None:
        """Adjust viewport_top so cursor stays visible in list pane."""
        page = self._page_size()
        total_count = len(self.entries)

        if self.cursor < self.viewport_top:
            self.viewport_top = self.cursor
        elif self.cursor >= self.viewport_top + page:
            self.viewport_top = self.cursor - page + 1

        # Clamp to valid range
        max_top = max(0, total_count - page)
        self.viewport_top = max(0, min(self.viewport_top, max_top))

    def _scroll_detail_into_view(self) -> None:
        """Clamp detail_scroll to valid range based on total_lines."""
        max_scroll = max(0, self._detail_total_lines - self._detail_content_height())
        self.detail_scroll = max(0, min(self.detail_scroll, max_scroll))

    # ── navigation actions ────────────────────────────────────────────────

    def _cursor_up(self) -> None:
        """Move cursor up one row."""
        if self.cursor > 0:
            self.cursor -= 1
            # Reset detail scroll when cursor moves
            self.detail_scroll = 0

    def _cursor_down(self) -> None:
        """Move cursor down one row."""
        total_count = len(self.entries)
        if self.cursor < total_count - 1:
            self.cursor += 1
            # Reset detail scroll when cursor moves
            self.detail_scroll = 0

    def _page_up(self) -> None:
        """Move cursor up one page."""
        self.cursor = max(0, self.cursor - self._page_size())
        self.detail_scroll = 0

    def _page_down(self) -> None:
        """Move cursor down one page."""
        total_count = len(self.entries)
        self.cursor = min(total_count - 1, self.cursor + self._page_size())
        self.detail_scroll = 0

    def _jump_to_start(self) -> None:
        """Jump to first message."""
        self.cursor = 0
        self.detail_scroll = 0

    def _jump_to_end(self) -> None:
        """Jump to last message."""
        total_count = len(self.entries)
        self.cursor = max(0, total_count - 1)
        self.detail_scroll = 0

    def _scroll_detail_up(self) -> None:
        """Scroll detail pane up one line."""
        if self.detail_scroll > 0:
            self.detail_scroll -= 1

    def _scroll_detail_down(self) -> None:
        """Scroll detail pane down one line."""
        max_scroll = max(0, self._detail_total_lines - self._detail_content_height())
        if self.detail_scroll < max_scroll:
            self.detail_scroll += 1

    # ── clipboard ────────────────────────────────────────────────────

    def _format_entry_as_text(self, entry: InspectEntry) -> str:
        """Format an InspectEntry as plain text for clipboard.

        Includes all parts with full content, no truncation.
        """
        lines: list[str] = []

        # Header
        lines.append(f"== Message #{entry.history_index} ==")
        lines.append(f"Kind: {entry.kind} | Role: {entry.role}")

        if entry.timestamp:
            lines.append(f"Timestamp: {entry.timestamp}")
        if entry.model_name:
            lines.append(f"Model: {entry.model_name}")
        if entry.usage:
            lines.append(
                f"Tokens: {entry.usage.input_tokens} in / {entry.usage.output_tokens} out"
            )

        lines.append("")  # blank line before parts

        # Parts
        for i, part in enumerate(entry.parts):
            part_header = f"[Part {i + 1}: {part.part_kind}"
            if part.tool_name:
                part_header += f" ({part.tool_name})"
            part_header += "]"
            lines.append(part_header)

            # Tool args for tool-call parts
            if part.tool_args:
                import json

                try:
                    args_str = json.dumps(part.tool_args, indent=2)
                    lines.append(f"Args: {args_str}")
                except Exception:
                    lines.append(f"Args: {part.tool_args}")

            # Content
            if part.content:
                lines.append(part.content)

            lines.append("")  # blank line between parts

        return "\n".join(lines)

    def _copy_to_clipboard(self, text: str) -> tuple[bool, str]:
        """Copy text to system clipboard.

        Args:
            text: The text to copy

        Returns:
            (success, message) tuple
        """
        system = platform.system()

        # encoding: bytes we feed the clipboard helper on stdin. Windows'
        # clip.exe reads UTF-16-LE, everyone else is happy with UTF-8.
        encoding = "utf-8"

        if system == "Darwin":  # macOS
            cmd = ["pbcopy"]
            tool = "pbcopy"
        elif system == "Linux":
            cmd = ["xclip", "-selection", "clipboard"]
            tool = "xclip"
        elif system == "Windows":
            cmd = ["clip"]
            tool = "clip"
            encoding = "utf-16-le"
        else:
            return False, f"Clipboard not supported on {system}"

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                # Bound the wait so a hung/blocked clipboard helper can never
                # freeze the TUI. 5s is generous for a local pipe write.
                _, stderr = process.communicate(
                    input=text.encode(encoding), timeout=5
                )
            except subprocess.TimeoutExpired:
                # Kill and reap the stuck child so we don't leak a process.
                process.kill()
                try:
                    process.communicate()
                except Exception:
                    pass
                return False, f"{tool} timed out after 5s"

            if process.returncode != 0:
                return False, f"{tool} failed: {stderr.decode()}"
            return True, "Copied to clipboard"

        except FileNotFoundError as e:
            return False, f"{tool} not found: {e}"
        except Exception as e:
            return False, f"Clipboard error: {e}"

    def _do_copy(self) -> None:
        """Copy current message to clipboard."""
        entry = self._current_entry()
        if entry is None:
            emit_warning("No message selected to copy")
            return

        text = self._format_entry_as_text(entry)
        success, message = self._copy_to_clipboard(text)

        if success:
            emit_info(f"Copied message #{entry.history_index} to clipboard")
        else:
            emit_warning(f"Copy failed: {message}")

    # ── rendering ────────────────────────────────────────────────────────

    def _render_footer(self) -> list[tuple[str, str]]:
        """Render navigation keybinding hints in the footer bar."""
        dim = "fg:ansibrightblack"
        key = "fg:ansiwhite"
        sep = (dim, " | ")

        hints: list[tuple[str, str]] = []

        # Navigation
        hints.extend([(key, "j/k"), (dim, ":nav ")])
        hints.extend([(key, "h/l"), (dim, ":scroll ")])
        # j/k move selection (j=up, k=down); h/l scroll detail (h=up, l=down)
        hints.append(sep)

        # Display
        hints.extend([(key, "c"), (dim, ":copy ")])
        hints.append(sep)

        # Exit
        hints.extend([(key, "q"), (dim, ":quit")])

        return hints

    def _update_display(self) -> None:
        """Refresh both panes."""
        self._scroll_list_into_view()

        if self.list_control:
            self.list_control.text = render_list(
                entries=self.entries,
                cursor_index=self.cursor,
                viewport_height=self._visible_rows,
                scroll_offset=self.viewport_top,
                width=self._list_width,
            )

        if self.detail_control:
            entry = self._current_entry()
            formatted_text, total_lines, clamped_scroll = render_detail(
                entry=entry,
                viewport_height=self._detail_viewport_height,
                scroll_offset=self.detail_scroll,
                width=max(20, self._detail_width - 2),
            )
            self._detail_total_lines = total_lines
            self.detail_scroll = clamped_scroll
            self.detail_control.text = formatted_text

        if self.footer_control:
            self.footer_control.text = self._render_footer()

    # ── keybindings ───────────────────────────────────────────────────────

    def _build_keybindings(self) -> KeyBindings:
        """Build all keybindings for the TUI."""
        kb = KeyBindings()

        # Navigation: cursor up/down
        # Directional mental model: left-hand keys (h, j) = UP,
        # right-hand keys (k, l) = DOWN. So j moves selection up,
        # k moves selection down (see inspect-104).
        @kb.add("up")
        @kb.add("c-p")
        @kb.add("j")
        def _up(event: KeyPressEvent) -> None:
            self._cursor_up()
            self._update_display()

        @kb.add("down")
        @kb.add("c-n")
        @kb.add("k")
        def _down(event: KeyPressEvent) -> None:
            self._cursor_down()
            self._update_display()

        # Navigation: fast movement
        @kb.add("pageup")
        def _pageup(event: KeyPressEvent) -> None:
            self._page_up()
            self._update_display()

        @kb.add("pagedown")
        def _pagedown(event: KeyPressEvent) -> None:
            self._page_down()
            self._update_display()

        @kb.add("home")
        @kb.add("g")
        def _home(event: KeyPressEvent) -> None:
            self._jump_to_start()
            self._update_display()

        @kb.add("end")
        @kb.add("G")
        def _end(event: KeyPressEvent) -> None:
            self._jump_to_end()
            self._update_display()

        # Detail pane scrolling: h (up) / l (down), or left/right
        @kb.add("h")
        @kb.add("left")
        def _detail_up(event: KeyPressEvent) -> None:
            self._scroll_detail_up()
            self._update_display()

        @kb.add("l")
        @kb.add("right")
        def _detail_down(event: KeyPressEvent) -> None:
            self._scroll_detail_down()
            self._update_display()

        # Actions
        @kb.add("c")
        def _copy(event: KeyPressEvent) -> None:
            self._do_copy()

        # Exit: q / Ctrl-C / Escape (no confirmation)
        @kb.add("q")
        @kb.add("c-c")
        @kb.add("escape")
        def _quit(event: KeyPressEvent) -> None:
            event.app.exit()

        return kb

    # ── terminal measurement ──────────────────────────────────────────────

    def _measure_terminal(self) -> tuple[int, int]:
        """Return (cols, rows) of the current terminal, with sane fallbacks."""
        try:
            size = shutil.get_terminal_size(fallback=(120, 40))
            return max(60, size.columns), max(15, size.lines)
        except Exception:
            return 120, 40

    # ── main entry ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Launch the TUI and block until user exits."""
        self.list_control = FormattedTextControl(text="")
        self.detail_control = FormattedTextControl(text="")
        self.footer_control = FormattedTextControl(text="")

        # Measure terminal and calculate pane dimensions
        cols, rows = self._measure_terminal()

        # Reserve space for chrome. Two side-by-side Frames in the VSplit cost
        # exactly 4 columns of border (1 per side, per frame). Anything more and
        # the (full_screen=False) app shrinks below the terminal, leaving dead
        # space on the right edge.
        usable_cols = max(40, cols - 4)

        # Left panel: ~35% of width, capped at 45 chars to maximize detail pane.
        # Minimum of 25 ensures index + role badge + some preview remains visible.
        left_cols = min(45, max(25, int(usable_cols * 0.35)))
        right_cols = usable_cols - left_cols

        self._list_width = left_cols
        self._detail_width = right_cols

        # Reserve lines for chrome. The detail pane is what drives total app
        # height: frame top+bottom border (2) + footer bar (1) = 3 rows. Reserve
        # exactly that so the app fills the terminal with no dead space at the
        # bottom edge.
        self._detail_viewport_height = max(5, rows - 3)
        # The list pane renders its own header + separator + scroll indicators +
        # cursor line inside the frame (~6 lines), so it gets a bit less room.
        self._visible_rows = max(5, rows - 9)

        list_width = Dimension(min=20, max=left_cols, preferred=left_cols)
        detail_width = Dimension(min=20, max=right_cols, preferred=right_cols)
        detail_height = Dimension(
            min=5,
            max=self._detail_viewport_height,
            preferred=self._detail_viewport_height,
        )

        list_window = Window(
            content=self.list_control,
            wrap_lines=False,
            width=list_width,
        )
        detail_window = Window(
            content=self.detail_control,
            wrap_lines=False,
            width=detail_width,
            height=detail_height,
        )

        list_frame = Frame(list_window, title="history")
        detail_frame = Frame(detail_window, title="detail")
        main_content = VSplit([list_frame, detail_frame])

        # Footer bar with keybinding hints (always visible)
        footer_bar = Window(
            content=self.footer_control,
            height=1,
            wrap_lines=False,
        )

        # Main layout: split panes above, footer below
        root = HSplit([main_content, footer_bar])

        layout = Layout(root)
        app: Application[None] = Application(
            layout=layout,
            key_bindings=self._build_keybindings(),
            full_screen=False,
            mouse_support=False,
        )

        # Signal to command runner that we're in interactive mode
        try:
            from code_puppy.tools.command_runner import set_awaiting_user_input

            set_awaiting_user_input(True)
        except Exception:
            pass

        # Enter alternate screen buffer for clean rendering
        sys.stdout.write("\033[?1049h")
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        time.sleep(0.05)

        try:
            self._update_display()
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            app.run(in_thread=True)
        finally:
            # Exit alternate screen buffer
            sys.stdout.write("\033[?1049l")
            sys.stdout.flush()
            try:
                import termios

                termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            except Exception:
                pass
            time.sleep(0.1)
            try:
                from code_puppy.tools.command_runner import set_awaiting_user_input

                set_awaiting_user_input(False)
            except Exception:
                pass


__all__ = ["InspectMenu"]
