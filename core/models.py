"""
Core data models for the postcondition generation system.

Contains all Pydantic models for postconditions, functions, results, and pipeline state.
"""

from typing import List, Optional, Dict, Any, Union  # Added Union for body field fix
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class ProcessingStatus(str, Enum):
    """Status of processing."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


class PostconditionStrength(str, Enum):
    """Strength level of postcondition."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class PostconditionCategory(str, Enum):
    """Category of postcondition."""
    CORE_CORRECTNESS = "core_correctness"
    BOUNDARY_SAFETY = "boundary_safety"
    ERROR_RESILIENCE = "error_resilience"
    PERFORMANCE_CONSTRAINTS = "performance_constraints"
    DOMAIN_COMPLIANCE = "domain_compliance"


# ============================================================================
# CORE MODELS
# ============================================================================

class FunctionParameter(BaseModel):
    """Function parameter (alias for Parameter for compatibility)."""
    name: str
    data_type: str
    description: str = ""


class Parameter(BaseModel):
    """Function parameter."""
    name: str
    data_type: str
    description: str = ""


class ReturnValue(BaseModel):
    """Function return value."""
    condition: str = "success"
    value: str = ""
    description: str = ""
    name: str = "result"


class Function(BaseModel):
    """
    Function representation.
    
    UPDATED: body field now accepts both string and list formats to handle
    different LLM response formats.
    """
    name: str
    signature: str
    description: str = ""
    body: Union[str, List[str]] = ""  # FIXED: Now accepts string OR list
    input_parameters: List[Parameter] = Field(default_factory=list)
    output_parameters: List[Parameter] = Field(default_factory=list)
    return_values: List[ReturnValue] = Field(default_factory=list)
    return_type: str = ""
    edge_cases: List[str] = Field(default_factory=list)
    complexity: str = ""
    memory_usage: str = ""
    dependencies: List[str] = Field(default_factory=list)
    
    @property
    def body_as_string(self) -> str:
        """
        Get body as string, converting from list if needed.
        
        If body is a list (e.g., ["Step 1", "Step 2", "Step 3"]),
        converts it to a numbered string format.
        If body is already a string, returns it as-is.
        
        Returns:
            String representation of the function body
            
        Example:
            >>> func.body = ["Initialize sum", "Loop through array", "Return sum"]
            >>> func.body_as_string
            "1. Initialize sum\\n2. Loop through array\\n3. Return sum"
        """
        if isinstance(self.body, list):
            # Convert list to numbered steps
            return "\n".join(f"{i+1}. {step}" for i, step in enumerate(self.body))
        return self.body
    
    class Config:
        arbitrary_types_allowed = True


class Z3Translation(BaseModel):
    """Z3 code translation with validation results."""
    formal_text: str
    natural_language: str
    z3_code: str = ""
    
    # Validation results
    z3_validation_passed: bool = False
    z3_validation_status: str = "not_validated"
    validation_error: Optional[str] = None
    
    # Analysis metadata
    z3_ast: Optional[Dict[str, Any]] = None
    tokens: Optional[List[tuple]] = None
    custom_functions: Optional[List[str]] = None
    declared_sorts: Optional[List[str]] = None
    declared_variables: Optional[Dict[str, str]] = None
    
    # Translation metrics
    translation_success: bool = False
    translation_time: float = 0.0
    generated_at: str = ""
    
    # Runtime metrics
    solver_created: bool = False
    constraints_added: int = 0
    variables_declared: int = 0
    execution_time: float = 0.0
    runtime_error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class EnhancedPostcondition(BaseModel):
    """
    Enhanced postcondition with comprehensive metadata.
    
    This is the MAIN postcondition model that contains all rich data:
    - Core postcondition text
    - Quality metrics
    - Edge case analysis
    - Mathematical properties
    - Z3 translation (attached by pipeline)
    """
    
    # Core postcondition
    formal_text: str
    natural_language: str
    
    # Rich translations & explanations
    precise_translation: str = ""
    reasoning: str = ""
    
    # Categorization
    strength: str = "standard"
    category: str = "core_correctness"
    
    # Edge case analysis
    edge_cases: List[str] = Field(default_factory=list)
    edge_cases_covered: List[str] = Field(default_factory=list)
    coverage_gaps: List[str] = Field(default_factory=list)
    
    # Quality metrics
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    robustness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    clarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    testability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    mathematical_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Mathematical properties
    mathematical_validity: str = ""
    z3_theory: str = "unknown"
    
    # Organization
    organization_rank: int = 0
    importance_category: str = ""
    selection_reasoning: str = ""
    robustness_assessment: str = ""
    is_primary_in_category: bool = False
    recommended_for_selection: bool = True
    
    # Z3 translation (attached by pipeline during processing)
    z3_translation: Optional[Z3Translation] = None
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# RESULT MODELS
# ============================================================================

class FunctionResult(BaseModel):
    """
    Result for a single function with comprehensive metrics.
    
    Contains:
    - All generated postconditions
    - Aggregated quality metrics
    - Edge case analysis
    - Z3 validation results
    """
    
    # Core identification
    function_name: str
    function_signature: str
    function_description: str
    pseudocode: Optional[Function] = None
    
    # Postconditions
    postconditions: List[EnhancedPostcondition] = Field(default_factory=list)
    postcondition_count: int = 0
    
    # Quality metrics (aggregated from postconditions)
    average_quality_score: float = 0.0
    average_robustness_score: float = 0.0
    average_confidence_score: float = 0.0
    average_clarity_score: float = 0.0
    average_completeness_score: float = 0.0
    
    # Edge case metrics
    total_edge_cases_covered: int = 0
    unique_edge_cases_count: int = 0
    total_coverage_gaps: int = 0
    
    # Mathematical validity
    mathematical_validity_rate: float = 0.0
    
    # Category distribution
    postconditions_by_category: Dict[str, int] = Field(default_factory=dict)
    
    # Z3 metrics
    z3_translations_count: int = 0
    z3_validations_passed: int = 0
    z3_validations_failed: int = 0
    z3_validation_errors: List[str] = Field(default_factory=list)
    
    # Status
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    processing_time: float = 0.0
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


class PseudocodeResult(BaseModel):
    """Result from pseudocode generation."""
    functions: List[Function] = Field(default_factory=list)
    structs: List[Dict[str, Any]] = Field(default_factory=list)
    enums: List[Dict[str, Any]] = Field(default_factory=list)
    global_variables: List[Dict[str, Any]] = Field(default_factory=list)
    includes: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class CompleteEnhancedResult(BaseModel):
    """
    Complete pipeline result with all functions.
    
    This is the top-level result that contains everything from a pipeline run:
    - All function results
    - Session information
    - Aggregate metrics
    - Status and errors
    """
    
    # Session info
    session_id: str
    specification: str
    started_at: str
    completed_at: str = ""
    
    # Pseudocode result (optional)
    pseudocode_result: Optional[PseudocodeResult] = None
    
    # Functions processed
    function_results: List[FunctionResult] = Field(default_factory=list)
    
    # Aggregate metrics
    total_functions: int = 0
    total_postconditions: int = 0
    total_z3_translations: int = 0
    
    average_quality_score: float = 0.0
    average_robustness_score: float = 0.0
    z3_validation_success_rate: float = 0.0
    solver_creation_rate: float = 0.0
    
    # Status
    status: ProcessingStatus = ProcessingStatus.PENDING
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    total_processing_time: float = 0.0
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


# ============================================================================
# EXPORT ALL MODELS
# ============================================================================

__all__ = [
    'ProcessingStatus',
    'PostconditionStrength',
    'PostconditionCategory',
    'FunctionParameter',
    'Parameter',
    'ReturnValue',
    'Function',
    'Z3Translation',
    'EnhancedPostcondition',
    'FunctionResult',
    'PseudocodeResult',
    'CompleteEnhancedResult',
]