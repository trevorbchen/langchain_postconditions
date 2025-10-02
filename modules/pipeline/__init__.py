"""
Pipeline Orchestration Module

This module provides the unified pipeline for orchestrating the complete
postcondition generation workflow.

Main components:
- PostconditionPipeline: Main orchestrator class
- create_pipeline: Convenience function for creating pipeline instances
"""

from modules.pipeline.pipeline import (
    PostconditionPipeline,
    create_pipeline
)

__all__ = [
    'PostconditionPipeline',
    'create_pipeline',
]

__version__ = '2.0.0'