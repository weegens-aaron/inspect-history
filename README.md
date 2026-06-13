# inspect-history

**A read-only deep inspector for your [Code Puppy](https://github.com/mpfaffenberger/code_puppy) conversation history.**

`inspect-history` is a Code Puppy plugin (native `register_callbacks.py` style). It adds an `/inspect` slash command that opens a split-pane terminal UI showing every message in the current conversation in **full detail with zero truncation** — every part, every tool call (with args), every thinking block, token usage, and provider metadata.

It is strictly **read-only**: `/inspect` never modifies your conversation history, so it's always safe to open mid-task.

---

## Install

This is a Code Puppy plugin. Clone it into your Code Puppy **user plugins** directory:

```bash
git clone https://github.com/weegens-aaron/inspect-history.git \
  ~/.code_puppy/plugins/inspect-history
```

Then restart Code Puppy. The `/inspect` command (alias `/i`) is now available.

To update later:

```bash
cd ~/.code_puppy/plugins/inspect-history && git pull
```

To uninstall, delete the directory and restart Code Puppy:

```bash
rm -rf ~/.code_puppy/plugins/inspect-history
```

> **Note:** the plugin directory must be named `inspect-history` (hyphenated). The
> modules use relative imports, which Code Puppy's plugin loader resolves correctly
> under the hyphenated package name.

---

## Usage

```
/inspect      # open the inspector on the current conversation
/i            # short alias
```

### Keys

| Key | Action |
|-----|--------|
| `j` / `k` | Move the selection up / down the message list |
| `h` / `l` (or arrow keys) | Scroll the detail pane up / down |
| `PageUp` / `PageDown` | Page through the message list |
| `g` / `G` (or Home/End) | Jump to first / last message |
| `c` | Copy the selected message (full detail) to the system clipboard |
| `q` / `Esc` / `Ctrl-C` | Quit (no confirmation — it's read-only) |

---

## What it shows

For every message, fully expanded with no truncation:

- **Message metadata** — history index, role, timestamp, model, provider, finish reason, run id.
- **Token usage** — input/output tokens, plus cache-read/write and audio tokens when present.
- **All parts** — system prompts, user prompts, assistant text, thinking blocks, tool calls (with full args), tool returns, retry prompts, and file parts.
- **Provider details** — any provider-specific metadata carried on the message or its parts.

The system prompt (carried in pydantic-ai `instructions`) is surfaced exactly once as a real system message instead of being repeated on every request, so it never leaks onto tool-return or follow-up messages.

---

## Requirements

- **Code Puppy** with the plugin system (`register_callbacks.py` plugins).
- A clipboard helper for the `c` copy key: `pbcopy` on macOS (built in) or `xclip` on Linux. Copy degrades gracefully with a warning if neither is available, and the helper is bounded with a 5s timeout so a hung clipboard tool can never freeze the UI.

---

## Development

```bash
# from the repo root
python -m pytest -q
```

The test suite registers the package as `inspect_history` via `tests/conftest.py`, so it
runs the same way Code Puppy loads the plugin at runtime.

Project layout:

| File | Responsibility |
|------|----------------|
| `register_callbacks.py` | Registers the `/inspect` command and bridges to the agent history. |
| `inspect_model.py` | Data model + pydantic-ai message introspection (zero truncation). |
| `inspect_render.py` | Pure rendering functions (data -> formatted text). |
| `inspect_menu.py` | The interactive split-pane TUI / state machine. |

---

## License

[MIT](LICENSE) (c) 2026 Aaron Weegens.
