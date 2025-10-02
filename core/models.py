"""
Pydantic Data Models for Postcondition Generation System

🔴 FIXED VERSION - Restored Missing Fields:
✅ Added 7 scoring fields to EnhancedPostcondition
✅ Added 3 aggregate fields to FunctionResult  
✅ Added 2 aggregate fields to CompleteEnhancedResult
✅ Kept all Phase 4 execution tracking fields
✅ Kept rich explanations (precise_translation, reasoning)
✅ Kept edge case analysis (edge_cases_covered, coverage_gaps)
✅ Kept Z3 validation with runtime metrics
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
# FUNCTION MODELS
# ============================================================================

class FunctionParameter(BaseModel):
    """Represents a function parameter."""
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
    """Represents a C function with full specification."""
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
# Z3 TRANSLATION MODEL
# ============================================================================

class Z3Translation(BaseModel):
    """Result from Z3 translation with validation."""
    
    formal_text: str
    natural_language: str
    z3_code: str = ""
    
    # Validation status
    z3_validation_passed: bool = False
    z3_validation_status: str = "not_validated"
    validation_error: Optional[str] = None
    
    # Analysis metadata
    z3_ast: Optional[Dict[str, Any]] = None
    tokens: Optional[List[tuple]] = None
    custom_functions: Optional[List[str]] = None
    declared_sorts: Optional[List[str]] = None
    declared_variables: Optional[Dict[str, str]] = None
    
    # Translation metadata
    translation_success: bool = False
    translation_time: float = 0.0
    generated_at: str = ""
    
    # Phase 4: Runtime validation metrics
    solver_created: bool = False
    constraints_added: int = 0
    variables_declared: int = 0
    execution_time: float = 0.0
    runtime_error: Optional[str] = None


# ============================================================================
# ENHANCED POSTCONDITION MODEL (FIXED - SCORING FIELDS RESTORED)
# ============================================================================

class EnhancedPostcondition(BaseModel):
    """
    Enhanced postcondition with comprehensive metadata.
    
    🔴 FIXED VERSION - Restored missing fields:
    ✅ Added back 7 scoring fields (confidence, robustness, clarity, etc.)
    ✅ Added back mathematical_validity field
    ✅ Kept rich explanations (precise_translation, reasoning)
    ✅ Kept edge case analysis (edge_cases_covered, coverage_gaps)
    ✅ Kept Z3 integration
    """
    
    # ========================================================================
    # CORE FIELDS
    # ========================================================================
    formal_text: str = Field(description="Mathematical formal specification")
    natural_language: str = Field(description="Brief natural language explanation")
    
    strength: PostconditionStrength = PostconditionStrength.STANDARD
    category: PostconditionCategory = PostconditionCategory.CORRECTNESS
    
    # ========================================================================
    # RICH EXPLANATIONS
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
    # EDGE CASE ANALYSIS
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
    # 🔴 SCORING FIELDS (RESTORED)
    # ========================================================================
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in postcondition accuracy (0.0-1.0)"
    )
    robustness_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Robustness against edge cases (0.0-1.0)"
    )
    clarity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Clarity of specification (0.0-1.0)"
    )
    completeness_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Completeness of coverage (0.0-1.0)"
    )
    testability_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ease of testing (0.0-1.0)"
    )
    mathematical_quality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Mathematical rigor (0.0-1.0)"
    )
    overall_quality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall quality score - computed from other scores (0.0-1.0)"
    )
    
    # Mathematical validity notes
    mathematical_validity: str = Field(
        default="",
        description="Notes on mathematical validity and correctness"
    )
    
    # ========================================================================
    # Z3 INTEGRATION
    # ========================================================================
    z3_theory: str = Field(
        default="unknown", 
        description="Z3 theory category (Arrays, Sequences, Arithmetic, etc.)"
    )
    z3_translation: Optional[Z3Translation] = None


# ============================================================================
# FUNCTION RESULT (FIXED - AGGREGATE FIELDS RESTORED)
# ============================================================================

class FunctionResult(BaseModel):
    """
    Result for a single function's postcondition generation.
    
    🔴 FIXED VERSION - Restored missing aggregate fields:
    ✅ Added average_quality_score
    ✅ Added average_robustness_score
    ✅ Added edge_case_coverage_score
    """
    
    function_name: str
    function_signature: str
    function_description: str = ""
    pseudocode: Optional[Function] = None
    
    # Postconditions
    postconditions: List[EnhancedPostcondition] = []
    postcondition_count: int = 0
    
    # ========================================================================
    # 🔴 AGGREGATE QUALITY SCORES (RESTORED)
    # ========================================================================
    average_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average overall quality across all postconditions"
    )
    average_robustness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average robustness across all postconditions"
    )
    edge_case_coverage_score: float = Field(
        default=0.0,
        description="Average number of edge cases covered per postcondition"
    )
    
    # ========================================================================
    # Z3 TRANSLATION STATS
    # ========================================================================
    z3_translations_count: int = 0
    z3_validations_passed: int = 0
    z3_validations_failed: int = 0
    z3_validation_errors: List[Dict[str, str]] = []
    average_solver_creation_rate: float = 0.0
    average_constraints_per_code: float = 0.0
    average_variables_per_code: float = 0.0
    
    # Edge case tracking
    total_edge_cases_covered: int = 0
    
    # Status
    status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    error_message: Optional[str] = None
    processing_time: float = 0.0


# ============================================================================
# COMPLETE RESULT (FIXED - AGGREGATE FIELDS RESTORED)
# ============================================================================

class CompleteEnhancedResult(BaseModel):
    """
    Complete result from the pipeline.
    
    🔴 FIXED VERSION - Restored missing aggregate fields:
    ✅ Added average_quality_score
    ✅ Added average_robustness_score
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
    
    # ========================================================================
    # 🔴 AGGREGATE QUALITY SCORES (RESTORED)
    # ========================================================================
    average_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average quality across all postconditions in pipeline"
    )
    average_robustness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average robustness across all postconditions"
    )
    
    # ========================================================================
    # Z3 VALIDATION STATISTICS
    # ========================================================================
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

def validate_fixed_model():
    """Validate that all required fields are present."""
    
    print("\n" + "=" * 80)
    print("VALIDATING FIXED MODELS")
    print("=" * 80)
    
    # Check EnhancedPostcondition
    print("\n✓ Checking EnhancedPostcondition...")
    pc_fields = set(EnhancedPostcondition.model_fields.keys())
    
    required_scoring_fields = [
        'confidence_score', 'robustness_score', 'clarity_score',
        'completeness_score', 'testability_score', 'mathematical_quality_score',
        'overall_quality_score', 'mathematical_validity'
    ]
    
    missing_pc = [f for f in required_scoring_fields if f not in pc_fields]
    if missing_pc:
        print(f"  ❌ MISSING: {missing_pc}")
        return False
    else:
        print(f"  ✅ All {len(required_scoring_fields)} scoring fields present")
    
    # Check FunctionResult
    print("\n✓ Checking FunctionResult...")
    fr_fields = set(FunctionResult.model_fields.keys())
    
    required_aggregate_fields = [
        'average_quality_score', 'average_robustness_score', 
        'edge_case_coverage_score'
    ]
    
    missing_fr = [f for f in required_aggregate_fields if f not in fr_fields]
    if missing_fr:
        print(f"  ❌ MISSING: {missing_fr}")
        return False
    else:
        print(f"  ✅ All {len(required_aggregate_fields)} aggregate fields present")
    
    # Check CompleteEnhancedResult
    print("\n✓ Checking CompleteEnhancedResult...")
    cr_fields = set(CompleteEnhancedResult.model_fields.keys())
    
    required_result_fields = ['average_quality_score', 'average_robustness_score']
    
    missing_cr = [f for f in required_result_fields if f not in cr_fields]
    if missing_cr:
        print(f"  ❌ MISSING: {missing_cr}")
        return False
    else:
        print(f"  ✅ All {len(required_result_fields)} result fields present")
    
    print("\n" + "=" * 80)
    print("✅ ALL MODELS FIXED - VALIDATION PASSED")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Copy this file to: core/models.py")
    print("2. Test: python -c \"from core.models import EnhancedPostcondition\"")
    print("3. Run pipeline: python main.py")
    
    return True


if __name__ == "__main__":
    validate_fixed_model()