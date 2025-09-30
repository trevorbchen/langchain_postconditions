"""
Pseudocode Generation Module

This module provides functionality for generating C pseudocode from
natural language specifications.

Main components:
- PseudocodeGenerator: Main class for generating pseudocode
- generate_pseudocode: Convenience function for quick generation
- generate_pseudocode_batch: Batch processing function
"""

from modules.pseudocode.pseudocode_generator import (
    PseudocodeGenerator,
    generate_pseudocode,
    generate_pseudocode_batch
)

__all__ = [
    'PseudocodeGenerator',
    'generate_pseudocode',
    'generate_pseudocode_batch',
]

__version__ = '2.0.0'