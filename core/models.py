"""
Pydantic Data Models for Postcondition Generation System

PHASE 4 COMPLETE: Enhanced Z3Translation with Runtime Validation Fields
- Added error_type and error_line fields (from validator)
- Added solver_created, constraints_added, variables_declared (execution metrics)
- Added validation_warnings for non-fatal issues
- Enhanced execution_time tracking
- All fields now populated by Z3CodeValidator
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
    is_pointer: bool = False
    is_array: bool = False
    is_const: bool = False
    
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
    
    preconditions: List[str] = []
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
# ENHANCED POSTCONDITION MODEL
# ============================================================================

class EnhancedPostcondition(BaseModel):
    """
    Enhanced postcondition with comprehensive metadata.
    
    MIGRATION COMPLETE: All fields from original system restored.
    """
    
    # Core fields
    formal_text: str = Field(description="Mathematical formal specification")
    natural_language: str = Field(description="Brief natural language explanation")
    
    strength: PostconditionStrength = PostconditionStrength.STANDARD
    category: PostconditionCategory = PostconditionCategory.CORRECTNESS
    
    # Translation fields
    precise_translation: str = Field(default="", description="Detailed natural language translation")
    reasoning: str = Field(default="", description="WHY this postcondition is necessary")
    
    # Edge case analysis
    edge_cases: List[str] = Field(default=[], description="Edge cases to consider")
    edge_cases_covered: List[str] = Field(default=[], description="Edge cases explicitly handled")
    coverage_gaps: List[str] = Field(default=[], description="Known coverage limitations")
    
    # Quality metrics
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    robustness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    clarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    testability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    mathematical_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_priority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Mathematical validation
    mathematical_validity: str = Field(default="", description="Mathematical correctness assessment")
    
    # Organization and ranking
    organization_rank: int = Field(default=0, description="Priority ranking")
    importance_category: str = Field(default="", description="Category of importance")
    selection_reasoning: str = Field(default="", description="Why this was selected")
    robustness_assessment: str = Field(default="", description="Robustness evaluation")
    is_primary_in_category: bool = Field(default=False)
    recommended_for_selection: bool = Field(default=True)
    
    # Z3 information
    z3_theory: str = Field(default="unknown", description="Z3 theory category")
    z3_translation: Optional['Z3Translation'] = None
    
    # Warnings and notes
    warnings: List[str] = Field(default=[], description="Warnings or caveats")
    
    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score."""
        if self.clarity_score == 0 and self.completeness_score == 0:
            return self.confidence_score
        
        scores = [
            (self.confidence_score, 0.3),
            (self.clarity_score, 0.2),
            (self.completeness_score, 0.2),
            (self.testability_score, 0.15),
            (self.robustness_score, 0.15)
        ]
        
        weighted_sum = sum(score * weight for score, weight in scores)
        return min(1.0, weighted_sum)
    
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


# ============================================================================
# ENHANCED Z3 TRANSLATION MODEL - PHASE 4 COMPLETE
# ============================================================================

class Z3Translation(BaseModel):
    """
    Result of translating a postcondition to Z3.
    
    PHASE 4 COMPLETE: Enhanced with runtime validation fields from Z3CodeValidator.
    
    New fields track:
    - Runtime execution results
    - Solver creation and usage
    - Constraint and variable counts
    - Detailed error information with types and line numbers
    - Execution timing
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
    # VALIDATION FIELDS (Enhanced in Phase 4)
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
    
    # 🆕 PHASE 4: Enhanced error information
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
        description="Non-fatal warnings from validation (replaces 'warnings')"
    )
    
    # ========================================================================
    # EXECUTION METRICS (NEW in Phase 4)
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
    # ANALYSIS METADATA (Restored from original system)
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
# PIPELINE RESULT MODELS (Enhanced)
# ============================================================================

class FunctionResult(BaseModel):
    """Result for a single function's postcondition generation."""
    
    function_name: str
    function_signature: str
    function_description: str = ""
    pseudocode: Optional[Function] = None
    
    # Postconditions
    postconditions: List[EnhancedPostcondition] = []
    postcondition_count: int = 0
    
    # Quality metrics (Phase 3 enhancements)
    average_quality_score: float = 0.0
    average_robustness_score: float = 0.0
    edge_case_coverage_score: float = 0.0
    mathematical_validity_rate: float = 0.0
    
    # Z3 translation stats (Phase 4 enhancements)
    z3_translations_count: int = 0
    z3_validations_passed: int = 0
    z3_validations_failed: int = 0
    z3_validation_errors: List[Dict[str, str]] = []
    average_solver_creation_rate: float = 0.0
    average_constraints_per_code: float = 0.0
    average_variables_per_code: float = 0.0
    
    # Status
    status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    error_message: Optional[str] = None
    processing_time: float = 0.0


class CompleteEnhancedResult(BaseModel):
    """Complete result from the pipeline."""
    
    session_id: str
    specification: str
    
    # Results
    pseudocode_result: Optional[PseudocodeResult] = None
    function_results: List[FunctionResult] = []
    
    # Statistics (Phase 3-4 enhancements)
    total_functions: int = 0
    total_postconditions: int = 0
    total_z3_translations: int = 0
    
    average_quality_score: float = 0.0
    average_robustness_score: float = 0.0
    average_validation_score: float = 0.0
    
    # 🆕 PHASE 5: Z3 validation statistics
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
# MODEL VALIDATION HELPERS
# ============================================================================

def validate_migration():
    """
    Validate that all fields from the original system are present.
    
    Checks EnhancedPostcondition and Z3Translation for completeness.
    """
    print("\n🔍 Validating Enhanced Models...")
    
    # Check EnhancedPostcondition fields
    required_postcondition_fields = [
        'formal_text', 'natural_language', 'precise_translation', 'reasoning',
        'edge_cases_covered', 'coverage_gaps', 'robustness_score',
        'mathematical_validity', 'organization_rank', 'importance_category'
    ]
    
    postcondition_fields = set(EnhancedPostcondition.model_fields.keys())
    missing = [f for f in required_postcondition_fields if f not in postcondition_fields]
    
    if missing:
        print(f"❌ Missing EnhancedPostcondition fields: {missing}")
    else:
        print(f"✅ EnhancedPostcondition has all {len(required_postcondition_fields)} required fields")
    
    # Check Z3Translation fields (Phase 4)
    required_z3_fields = [
        'z3_validation_passed', 'z3_validation_status', 'validation_error',
        'error_type', 'error_line', 'solver_created', 'constraints_added',
        'variables_declared', 'execution_time', 'validation_warnings'
    ]
    
    z3_fields = set(Z3Translation.model_fields.keys())
    missing_z3 = [f for f in required_z3_fields if f not in z3_fields]
    
    if missing_z3:
        print(f"❌ Missing Z3Translation fields: {missing_z3}")
    else:
        print(f"✅ Z3Translation has all {len(required_z3_fields)} validation fields (Phase 4)")
    
    return len(missing) == 0 and len(missing_z3) == 0


# ============================================================================
# MAIN - TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 4 COMPLETE - Enhanced Models with Runtime Validation")
    print("=" * 80)
    
    # Validate migration
    print("\n📋 Checking Phase 4 completeness...")
    all_fields_present = validate_migration()
    
    if all_fields_present:
        print("\n✅ All required fields present!")
    else:
        print("\n⚠️ Some fields missing - check output above")
    
    # Example: Create Z3 translation with Phase 4 fields
    print("\n📝 Creating Z3Translation with Phase 4 validation fields...")
    
    translation = Z3Translation(
        formal_text="∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
        natural_language="Array is sorted",
        z3_code="from z3 import *\n...",
        z3_theory_used="arrays",
        translation_success=True,
        z3_validation_passed=True,
        z3_validation_status="success",
        solver_created=True,
        constraints_added=5,
        variables_declared=4,
        execution_time=0.023,
        declared_variables={"i": "Int", "j": "Int", "arr": "Array", "n": "Int"}
    )
    
    print(f"✅ Created Z3Translation")
    print(f"   Valid: {translation.is_valid}")
    print(f"   Validation Score: {translation.validation_score:.2f}")
    print(f"   Solver Created: {translation.solver_created}")
    print(f"   Constraints: {translation.constraints_added}")
    print(f"   Variables: {translation.variables_declared}")
    print(f"   Execution Time: {translation.execution_time}s")
    
    # Show new Phase 4 fields
    print("\n🆕 Phase 4 New Fields:")
    print(f"   error_type: {translation.error_type}")
    print(f"   error_line: {translation.error_line}")
    print(f"   validation_warnings: {translation.validation_warnings}")
    print(f"   solver_created: {translation.solver_created}")
    print(f"   constraints_added: {translation.constraints_added}")
    print(f"   variables_declared: {translation.variables_declared}")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 4 MIGRATION COMPLETE")
    print("=" * 80)
    print("\nCompleted phases:")
    print("1. ✅ Enhanced models with all fields")
    print("2. ✅ Z3 validator created (validator.py)")
    print("3. ✅ Settings updated (z3_validation config)")
    print("4. ✅ Translator integrated with validator")
    print("5. ✅ Models enhanced with validation fields")
    print("\nNext: Update pipeline to preserve all data!")
    print("=" * 80)