"""Pytest configuration for inspect_history tests.

The inspect_history plugin uses relative imports (e.g., from .inspect_model import ...).
At runtime, code_puppy's plugin loader registers it as a package by adding
~/.code_puppy/plugins/ to sys.path. Under bare pytest we have to do that
ourselves so the modules import cleanly.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLUGINS_ROOT = os.path.dirname(_PLUGIN_DIR)
_PKG = "inspect_history"


def _register_package() -> None:
    """Make ``inspect_history`` importable as a real package."""
    if _PKG in sys.modules:
        return

    # Add the plugins root to sys.path so inspect_history can be imported as a package
    if _PLUGINS_ROOT not in sys.path:
        sys.path.insert(0, _PLUGINS_ROOT)

    spec = importlib.util.spec_from_file_location(
        _PKG,
        os.path.join(_PLUGIN_DIR, "__init__.py"),
        submodule_search_locations=[_PLUGIN_DIR],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG] = module
    spec.loader.exec_module(module)


_register_package()
