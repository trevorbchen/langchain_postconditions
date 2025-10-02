"""
Pydantic Data Models for Postcondition Generation System

STREAMLINED VERSION - Removed:
❌ is_pointer, is_array, is_const from FunctionParameter
❌ preconditions from Function model
❌ All scoring fields from EnhancedPostcondition
❌ All ranking/organization fields from EnhancedPostcondition

KEPT - Phase 4 Complete:
✅ Rich explanations (precise_translation, reasoning)
✅ Edge case analysis (edge_cases_covered, coverage_gaps)
✅ Z3 validation with runtime metrics
✅ All Phase 4 execution tracking fields
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# ENUMS
# ============================================================================

class PostconditionStrength(str, Enum):
    """Strength levels for postconditions."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class PostconditionCategory(str, Enum):
    """Categories of postconditions."""
    RETURN_VALUE = "return_value"
    STATE_CHANGE = "state_change"
    SIDE_EFFECT = "side_effect"
    ERROR_CONDITION = "error_condition"
    MEMORY = "memory"
    CORRECTNESS = "correctness"
    CORE_CORRECTNESS = "core_correctness"
    BOUNDARY_SAFETY = "boundary_safety"
    ERROR_RESILIENCE = "error_resilience"
    PERFORMANCE_CONSTRAINTS = "performance_constraints"
    DOMAIN_COMPLIANCE = "domain_compliance"


class ProcessingStatus(str, Enum):
    """Status of pipeline processing."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


# ============================================================================
# FUNCTION MODELS (STREAMLINED)
# ============================================================================

class FunctionParameter(BaseModel):
    """
    Represents a function parameter.
    
    ❌ REMOVED: is_pointer, is_array, is_const
    """
    name: str
    data_type: str
    description: str = ""
    
    @field_validator('data_type')
    @classmethod
    def validate_data_type(cls, v):
        """Ensure data type is not empty."""
        if not v or not v.strip():
            raise ValueError("data_type cannot be empty")
        return v.strip()


class ReturnValue(BaseModel):
    """Represents a possible return value."""
    condition: str
    value: str
    description: str
    name: Optional[str] = "result"


class Dependency(BaseModel):
    """Represents a function dependency."""
    function: str
    source: str  # "stdlib", "codebase", "generated"
    header: Optional[str] = None


class Function(BaseModel):
    """
    Represents a C function with full specification.
    
    ❌ REMOVED: preconditions field
    """
    name: str
    description: str
    signature: str = ""
    
    input_parameters: List[FunctionParameter] = []
    output_parameters: List[FunctionParameter] = []
    return_values: List[ReturnValue] = []
    return_type: str = "void"
    
    edge_cases: List[str] = []
    
    complexity: str = "O(n)"
    memory_usage: str = "O(1)"
    
    body: str = ""
    dependencies: List[Dependency] = []


class Struct(BaseModel):
    """Represents a C struct."""
    name: str
    fields: List[Dict[str, str]]
    description: str = ""


class Enum(BaseModel):
    """Represents a C enum."""
    name: str
    values: List[str]
    description: str = ""


# ============================================================================
# PSEUDOCODE RESULT
# ============================================================================

class PseudocodeResult(BaseModel):
    """Result from pseudocode generation."""
    functions: List[Function] = []
    structs: List[Struct] = []
    enums: List[Enum] = []
    global_variables: List[Dict[str, Any]] = []
    includes: List[str] = []
    dependencies: List[Dependency] = []
    metadata: Dict[str, Any] = {}
    
    @property
    def function_names(self) -> List[str]:
        """Get list of function names."""
        return [f.name for f in self.functions]


# ============================================================================
# ENHANCED POSTCONDITION MODEL (STREAMLINED)
# ============================================================================

class EnhancedPostcondition(BaseModel):
    """
    Enhanced postcondition with comprehensive metadata.
    
    STREAMLINED VERSION - Removed ALL scoring and ranking fields:
    ❌ confidence_score, robustness_score, clarity_score
    ❌ completeness_score, testability_score, mathematical_quality_score
    ❌ overall_priority_score
    ❌ mathematical_validity
    ❌ organization_rank, importance_category
    ❌ selection_reasoning, robustness_assessment
    ❌ is_primary_in_category, recommended_for_selection
    
    KEPT - Core functionality:
    ✅ formal_text, natural_language
    ✅ precise_translation, reasoning (rich explanations)
    ✅ edge_cases_covered, coverage_gaps (edge case analysis)
    ✅ z3_theory, z3_translation (Z3 integration)
    """
    
    # ========================================================================
    # CORE FIELDS
    # ========================================================================
    formal_text: str = Field(description="Mathematical formal specification")
    natural_language: str = Field(description="Brief natural language explanation")
    
    strength: PostconditionStrength = PostconditionStrength.STANDARD
    category: PostconditionCategory = PostconditionCategory.CORRECTNESS
    
    # ========================================================================
    # RICH EXPLANATIONS (KEPT)
    # ========================================================================
    precise_translation: str = Field(
        default="", 
        description="Detailed natural language translation of formal logic"
    )
    reasoning: str = Field(
        default="", 
        description="WHY this postcondition is necessary and what it ensures"
    )
    
    # ========================================================================
    # EDGE CASE ANALYSIS (KEPT)
    # ========================================================================
    edge_cases: List[str] = Field(
        default=[], 
        description="Edge cases to consider (general)"
    )
    edge_cases_covered: List[str] = Field(
        default=[], 
        description="Specific edge cases explicitly handled"
    )
    coverage_gaps: List[str] = Field(
        default=[], 
        description="Known coverage limitations or scenarios not addressed"
    )
    
    # ========================================================================
    # Z3 INTEGRATION (KEPT)
    # ========================================================================
    z3_theory: str = Field(
        default="unknown", 
        description="Z3 theory category (Arrays, Sequences, Arithmetic, etc.)"
    )
    z3_translation: Optional['Z3Translation'] = None
    
    # ========================================================================
    # WARNINGS (KEPT)
    # ========================================================================
    warnings: List[str] = Field(
        default=[], 
        description="Warnings or caveats about this postcondition"
    )
    
    # ========================================================================
    # COMPUTED PROPERTIES (SIMPLIFIED)
    # ========================================================================
    
    @property
    def has_translations(self) -> bool:
        """Check if detailed translations are available."""
        return bool(self.precise_translation and self.reasoning)
    
    @property
    def edge_case_coverage_ratio(self) -> float:
        """Ratio of covered edge cases to total identified."""
        total = len(self.edge_cases) + len(self.edge_cases_covered)
        if total == 0:
            return 0.0
        return len(self.edge_cases_covered) / total
    
    @property
    def has_z3_translation(self) -> bool:
        """Check if Z3 translation is available."""
        return self.z3_translation is not None


# ============================================================================
# ENHANCED Z3 TRANSLATION MODEL - PHASE 4 COMPLETE (UNCHANGED)
# ============================================================================

class Z3Translation(BaseModel):
    """
    Result of translating a postcondition to Z3.
    
    PHASE 4 COMPLETE: Enhanced with runtime validation fields from Z3CodeValidator.
    
    This model is UNCHANGED - all Phase 4 validation fields are kept.
    """
    
    formal_text: str
    natural_language: str
    
    # ========================================================================
    # Z3 CODE & THEORY
    # ========================================================================
    z3_code: str = ""
    z3_theory_used: str = "unknown"
    
    # ========================================================================
    # TRANSLATION STATUS
    # ========================================================================
    translation_success: bool = False
    translation_time: float = 0.0
    
    # ========================================================================
    # VALIDATION FIELDS (Phase 4 - KEPT)
    # ========================================================================
    z3_validation_passed: bool = Field(
        default=False,
        description="Whether the Z3 code passed all validation checks"
    )
    
    z3_validation_status: str = Field(
        default="not_validated",
        description="Validation status: success, syntax_error, import_error, runtime_error, timeout_error, not_validated"
    )
    
    validation_error: Optional[str] = Field(
        default=None,
        description="Error message if validation failed"
    )
    
    error_type: Optional[str] = Field(
        default=None,
        description="Type of error: SyntaxError, ImportError, NameError, TypeError, TimeoutError, etc."
    )
    
    error_line: Optional[int] = Field(
        default=None,
        description="Line number where error occurred (if applicable)"
    )
    
    validation_warnings: List[str] = Field(
        default=[],
        description="Non-fatal warnings from validation"
    )
    
    # ========================================================================
    # EXECUTION METRICS (Phase 4 - KEPT)
    # ========================================================================
    solver_created: bool = Field(
        default=False,
        description="Whether Solver() was successfully created in the code"
    )
    
    constraints_added: int = Field(
        default=0,
        ge=0,
        description="Number of constraints added to the solver (s.add() calls)"
    )
    
    variables_declared: int = Field(
        default=0,
        ge=0,
        description="Number of Z3 variables declared (Int, Real, Array, etc.)"
    )
    
    execution_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to execute/validate the Z3 code (seconds)"
    )
    
    # ========================================================================
    # ANALYSIS METADATA (KEPT)
    # ========================================================================
    z3_ast: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parsed Z3 abstract syntax tree"
    )
    
    tokens: Optional[List[tuple]] = Field(
        default=None,
        description="Tokenized representation for analysis"
    )
    
    custom_functions: List[str] = Field(
        default=[],
        description="List of custom functions defined in Z3 code"
    )
    
    declared_sorts: List[str] = Field(
        default=[],
        description="Z3 sorts declared (IntSort, ArraySort, etc.)"
    )
    
    declared_variables: Dict[str, str] = Field(
        default={},
        description="Variables declared with their types {var_name: type}"
    )
    
    # ========================================================================
    # METADATA
    # ========================================================================
    warnings: List[str] = Field(
        default=[],
        description="DEPRECATED: Use validation_warnings instead"
    )
    
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp when translation was generated"
    )
    
    # ========================================================================
    # COMPUTED PROPERTIES
    # ========================================================================
    @property
    def is_valid(self) -> bool:
        """Check if translation is valid and passed validation."""
        return self.translation_success and self.z3_validation_passed
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return self.validation_error is not None
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.validation_warnings) > 0 or len(self.warnings) > 0
    
    @property
    def validation_score(self) -> float:
        """
        Calculate validation quality score (0.0 to 1.0).
        
        Based on:
        - Validation passed (50%)
        - Solver created (25%)
        - Has constraints (15%)
        - Has variables (10%)
        """
        score = 0.0
        
        if self.z3_validation_passed:
            score += 0.5
        
        if self.solver_created:
            score += 0.25
        
        if self.constraints_added > 0:
            score += 0.15
        
        if self.variables_declared > 0:
            score += 0.10
        
        return score
    
    class Config:
        json_schema_extra = {
            "example": {
                "formal_text": "∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
                "natural_language": "Array is sorted in ascending order",
                "z3_code": "from z3 import *\n...",
                "z3_theory_used": "arrays",
                "translation_success": True,
                "z3_validation_passed": True,
                "z3_validation_status": "success",
                "solver_created": True,
                "constraints_added": 3,
                "variables_declared": 4,
                "execution_time": 0.023,
                "declared_variables": {"i": "Int", "j": "Int", "arr": "Array", "n": "Int"}
            }
        }


# ============================================================================
# PIPELINE RESULT MODELS (STREAMLINED)
# ============================================================================

class FunctionResult(BaseModel):
    """
    Result for a single function's postcondition generation.
    
    STREAMLINED: Removed quality scoring aggregates, kept Z3 validation metrics.
    """
    
    function_name: str
    function_signature: str
    function_description: str = ""
    pseudocode: Optional[Function] = None
    
    # Postconditions
    postconditions: List[EnhancedPostcondition] = []
    postcondition_count: int = 0
    
    # Z3 translation stats (Phase 4 - useful for validation tracking)
    z3_translations_count: int = 0
    z3_validations_passed: int = 0
    z3_validations_failed: int = 0
    z3_validation_errors: List[Dict[str, str]] = []
    average_solver_creation_rate: float = 0.0
    average_constraints_per_code: float = 0.0
    average_variables_per_code: float = 0.0
    
    # Simple edge case count (useful metric)
    total_edge_cases_covered: int = 0
    
    # Status
    status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    error_message: Optional[str] = None
    processing_time: float = 0.0


class CompleteEnhancedResult(BaseModel):
    """
    Complete result from the pipeline.
    
    STREAMLINED: Removed quality scoring aggregates, kept Z3 validation stats.
    """
    
    session_id: str
    specification: str
    
    # Results
    pseudocode_result: Optional[PseudocodeResult] = None
    function_results: List[FunctionResult] = []
    
    # Statistics
    total_functions: int = 0
    total_postconditions: int = 0
    total_z3_translations: int = 0
    
    # Z3 validation statistics (Phase 4 - useful for tracking)
    z3_validation_success_rate: float = 0.0
    solver_creation_rate: float = 0.0
    
    # Status
    status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    errors: List[str] = []
    warnings: List[str] = []
    
    # Timing
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    total_processing_time: float = 0.0


# ============================================================================
# VALIDATION HELPER
# ============================================================================

def validate_streamlined_migration():
    """Validate that streamlined model has correct fields."""
    print("\nValidating Streamlined Models...")
    
    # Fields that SHOULD exist
    required_fields = [
        'formal_text', 'natural_language', 'precise_translation', 'reasoning',
        'edge_cases_covered', 'coverage_gaps', 'z3_theory'
    ]
    
    # Fields that should NOT exist
    removed_fields = [
        'confidence_score', 'robustness_score', 'clarity_score',
        'completeness_score', 'testability_score', 'mathematical_quality_score',
        'overall_priority_score', 'mathematical_validity',
        'organization_rank', 'importance_category', 'selection_reasoning',
        'robustness_assessment', 'is_primary_in_category'
    ]
    
    postcondition_fields = set(EnhancedPostcondition.model_fields.keys())
    
    missing = [f for f in required_fields if f not in postcondition_fields]
    still_present = [f for f in removed_fields if f in postcondition_fields]
    
    if missing:
        print(f"Missing required fields: {missing}")
        return False
    
    if still_present:
        print(f"Fields should be removed but still present: {still_present}")
        return False
    
    print(f"All {len(required_fields)} required fields present")
    print(f"All {len(removed_fields)} unwanted fields successfully removed")
    
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("STREAMLINED MODELS - Ready for Production")
    print("=" * 80)
    
    validation_passed = validate_streamlined_migration()
    
    if validation_passed:
        print("\nSTREAMLINED MIGRATION SUCCESSFUL!")
        print("\nRemoved:")
        print("  - All scoring fields (7 fields)")
        print("  - All ranking fields (6 fields)")
        print("  - is_pointer, is_array, is_const from FunctionParameter")
        print("  - preconditions from Function")
        print("\nKept:")
        print("  - Rich explanations (precise_translation, reasoning)")
        print("  - Edge case analysis (edge_cases_covered, coverage_gaps)")
        print("  - Z3 integration (z3_theory, z3_translation)")
        print("  - All Phase 4 validation fields")
    else:
        print("\nVALIDATION FAILED - Check output above")
    
    print("=" * 80)