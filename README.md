# inspect-history

**A read-only deep inspector for your [Code Puppy](https://github.com/mpfaffenberger/code_puppy) conversation history.**

`inspect-history` is a Code Puppy plugin (native `register_callbacks.py` style). It adds an `/inspect` slash command that opens a split-pane terminal UI showing every message in the current conversation in **full detail with zero truncation** — every part, every tool call (with args), every thinking block, token usage, and provider metadata.

It is strictly **read-only**: `/inspect` never modifies your conversation history, so it's always safe to open mid-task.

---

## Install

This is a Code Puppy plugin. Plugins live in `~/.code_puppy/plugins/`. Install the
latest release straight into that directory, then restart Code Puppy.

### macOS / Linux

```bash
curl -fsSL https://github.com/weegens-aaron/inspect-history/releases/latest/download/inspect-history.zip -o /tmp/inspect-history.zip && unzip -o /tmp/inspect-history.zip -d ~/.code_puppy/plugins/
```

### Windows (PowerShell)

```powershell
Invoke-WebRequest -Uri https://github.com/weegens-aaron/inspect-history/releases/latest/download/inspect-history.zip -OutFile $env:TEMP\inspect-history.zip; Expand-Archive -Force $env:TEMP\inspect-history.zip -DestinationPath ~\.code_puppy\plugins\
```

### Manual download (any platform, no CLI)

1. Go to the [**Releases** page](https://github.com/weegens-aaron/inspect-history/releases/latest).
2. Download **`inspect-history.zip`** from the latest release's assets.
3. Extract it into `~/.code_puppy/plugins/` (macOS/Linux) or `~\.code_puppy\plugins\` (Windows).

The zip contains a single top-level `inspect_history/` folder, so every path above
results in `…/plugins/inspect_history/…` — extract, don't nest.

After installing, **restart Code Puppy**. The `/inspect` command (alias `/i`) is now available.

### Verify your download (optional but recommended)

Every release publishes an `inspect-history.zip.sha256` asset next to the zip. **macOS / Linux**, after running the install one-liner (which leaves the zip at `/tmp/inspect-history.zip`):

```bash
curl -fsSL https://github.com/weegens-aaron/inspect-history/releases/latest/download/inspect-history.zip.sha256 -o /tmp/inspect-history.zip.sha256
( cd /tmp && shasum -a 256 -c inspect-history.zip.sha256 )   # prints "inspect-history.zip: OK"
```

(`sha256sum -c inspect-history.zip.sha256` works on distros that ship `sha256sum`.)

### Upgrade / Uninstall

| Action | macOS / Linux | Windows |
|--------|---------------|---------|
| **Upgrade** | Re-run the install line — it always pulls the latest release. | Re-run the install line. |
| **Uninstall** | `rm -rf ~/.code_puppy/plugins/inspect_history` | `Remove-Item -Recurse -Force ~\.code_puppy\plugins\inspect_history` |

Every command uses the stable `/releases/latest/download/inspect-history.zip` URL — a
fixed asset name on the latest release — so nothing ever needs version-editing.

> The plugin's modules use relative imports, which Code Puppy's plugin loader resolves
> correctly whether the directory is named `inspect_history` (the release zip) or
> `inspect-history` (a git clone).
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

### Releasing

`scripts/build-release.sh` builds the distributable zip from an explicit allowlist
(runtime `.py` files + `README.md` + `LICENSE`):

```bash
./scripts/build-release.sh
```

It reads `__version__` from `__init__.py` (the single source of truth), writes both a
stable `dist/inspect-history.zip` and a versioned `dist/inspect-history-v<version>.zip`
(each with a `.sha256` sidecar), and self-checks by extracting the zip and importing the
package. To cut a release, bump `__version__`, run the script, then upload every
`dist/` artifact:

```bash
gh release create v<version> dist/inspect-history*.zip dist/inspect-history*.zip.sha256
```

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
