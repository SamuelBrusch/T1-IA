"""
Shim module to keep backward compatibility with `from gui import run_gui` when
both a top-level `src/gui.py` file and the package `src/gui/` exist.

This file forwards run_gui from the package implementation at `src/gui/gui.py`.
"""

import importlib.util
import os

_PKG_GUI_PATH = os.path.join(os.path.dirname(__file__), 'gui', 'gui.py')

def _load_pkg_gui_module():
    spec = importlib.util.spec_from_file_location('gui_pkg_gui', _PKG_GUI_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar GUI do pacote em {_PKG_GUI_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_gui():
    """Entry point that lazily loads and runs the package GUI implementation."""
    _load_pkg_gui_module().run_gui()
