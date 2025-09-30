"""
Logic Generation Module

This module provides functionality for generating formal postconditions
from function specifications.

Main components:
- PostconditionGenerator: Main class for generating postconditions
- generate_postconditions: Convenience function for quick generation
- generate_postconditions_batch: Batch processing function
"""

from modules.logic.logic_generator import (
    PostconditionGenerator,
    generate_postconditions,
    generate_postconditions_batch
)

__all__ = [
    'PostconditionGenerator',
    'generate_postconditions',
    'generate_postconditions_batch',
]

__version__ = '2.0.0'