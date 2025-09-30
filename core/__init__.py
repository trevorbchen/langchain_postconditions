"""
Core Package

This package contains the core functionality: models, chains, and agents
for the postcondition generation system.

Components:
- models: Pydantic data models
- chains: LangChain chain implementations
- agents: LangChain agent implementations (future)
"""

__version__ = '2.0.0'

# Core components are imported on-demand to avoid circular dependencies
# Use: from core.models import Function, EnhancedPostcondition
# Use: from core.chains import ChainFactory