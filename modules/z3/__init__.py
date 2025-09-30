"""
Z3 Translation Module

This module provides functionality for translating formal postconditions
to executable Z3 verification code.

Main components:
- Z3Translator: Main class for Z3 code generation
- translate_postcondition: Convenience function for quick translation
- translate_batch: Batch processing function
"""

from modules.z3.translator import (
    Z3Translator,
    translate_postcondition,
    translate_batch
)

__all__ = [
    'Z3Translator',
    'translate_postcondition',
    'translate_batch',
]

__version__ = '2.0.0'