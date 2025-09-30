"""
Pipeline Orchestration Module

This module provides the unified pipeline for orchestrating the complete
postcondition generation workflow.

Main components:
- PostconditionPipeline: Main orchestrator class
- process_specification: Convenience function for processing specifications
"""

from modules.pipeline.pipeline import (
    PostconditionPipeline,
    process_specification
)

__all__ = [
    'PostconditionPipeline',
    'process_specification',
]

__version__ = '2.0.0'