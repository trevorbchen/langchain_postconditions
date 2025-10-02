"""
Pseudocode Generation Module

This module provides functionality for generating C pseudocode from
natural language specifications.

Main components:
- PseudocodeGenerator: Main class for generating pseudocode
- generate_pseudocode_api: Backward compatible API function
"""

from modules.pseudocode.pseudocode_generator import (
    PseudocodeGenerator,
    generate_pseudocode_api
)

__all__ = [
    'PseudocodeGenerator',
    'generate_pseudocode_api',
]

__version__ = '2.0.0'