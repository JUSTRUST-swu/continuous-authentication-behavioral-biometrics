"""Compatibility wrapper for rne.plots.model_vote_stacked."""

from importlib import import_module as _import_module
import sys as _sys

_mod = _import_module("rne.plots.model_vote_stacked")
_sys.modules[__name__] = _mod

if __name__ == "__main__" and hasattr(_mod, "main"):
    _mod.main()
