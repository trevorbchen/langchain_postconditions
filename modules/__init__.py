"""
Modules Package

This package contains all the refactored modules for the postcondition
generation system.

Submodules:
- pseudocode: C pseudocode generation
- logic: Formal postcondition generation
- z3: Z3 code translation
- pipeline: Unified orchestration
"""

__version__ = '2.0.0'

# Submodules are imported on-demand to avoid circular dependencies
# Use: from modules.pseudocode import PseudocodeGenerator
# Use: from modules.logic import PostconditionGenerator
# Use: from modules.z3 import Z3Translator
# Use: from modules.pipeline import PostconditionPipeline