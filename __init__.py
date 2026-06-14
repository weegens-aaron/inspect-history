"""inspect-history: a read-only deep inspector for Code Puppy.

Adds the /inspect slash command, which opens a split-pane TUI showing every
message in the current conversation in full detail with zero truncation.
"""

# Single source of truth for the plugin version. Keep this in sync with the
# "version" field in .plugin/plugin.json on every release (bump both together).
__version__ = "1.0.3"

__all__ = ["__version__"]
