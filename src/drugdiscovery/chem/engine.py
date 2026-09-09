"""Facade submodule so both `from . import engine` and `from .chem import <fn>` work.

The RDKit engine lives in the package `__init__`; importing this module lets
callers use the conventional `engine.canonical_smiles(...)` form too.
"""
from . import *  # noqa: F401,F403