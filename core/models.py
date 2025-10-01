"""
Pydantic Data Models for Postcondition Generation System

PHASE 1 MIGRATION: Restored ALL rich fields from original system
- Added 15+ missing fields to EnhancedPostcondition
- Enhanced Z3Translation with validation metadata
- Preserved backward compatibility
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
# ENHANCED POSTCONDITION MODEL - RESTORED ALL FIELDS
# ============================================================================

class EnhancedPostcondition(BaseModel):
    """
    Enhanced postcondition with comprehensive metadata.
    
    MIGRATION PHASE 1: Restored ALL fields from original system:
    - Core formal specification fields (existing)
    - Precise translation and reasoning (restored)
    - Edge case analysis (restored)
    - Quality and robustness metrics (restored)
    - Organization and ranking (restored)
    - Mathematical validation (restored)
    """
    
    # ========================================================================
    # CORE FIELDS (Existing - keep as is)
    # ========================================================================
    formal_text: str = Field(
        description="Mathematical formal specification using proper notation"
    )
    natural_language: str = Field(
        description="Brief natural language explanation"
    )
    
    strength: PostconditionStrength = PostconditionStrength.STANDARD
    category: PostconditionCategory = PostconditionCategory.CORRECTNESS
    
    # ========================================================================
    # TRANSLATION FIELDS (Restored from original system)
    # ========================================================================
    precise_translation: str = Field(
        default="",
        description="Detailed, precise natural language translation of formal text"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of WHY this postcondition is necessary and what it prevents"
    )
    
    # ========================================================================
    # EDGE CASE ANALYSIS (Restored from original system)
    # ========================================================================
    edge_cases: List[str] = Field(
        default=[],
        description="List of edge cases relevant to this postcondition"
    )
    edge_cases_covered: List[str] = Field(
        default=[],
        description="Specific edge cases explicitly addressed by this postcondition"
    )
    coverage_gaps: List[str] = Field(
        default=[],
        description="Known edge cases NOT covered by this postcondition"
    )
    
    # ========================================================================
    # QUALITY SCORES (Existing + Enhanced)
    # ========================================================================
    confidence_score: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0,
        description="Confidence in correctness of this postcondition"
    )
    clarity_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="How clear and understandable the postcondition is"
    )
    completeness_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="How completely this captures the intended property"
    )
    testability_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="How easy it is to test/verify this postcondition"
    )
    
    # ========================================================================
    # ROBUSTNESS METRICS (Restored from original system)
    # ========================================================================
    robustness_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Overall robustness considering edge cases, boundaries, errors"
    )
    mathematical_quality_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Quality of mathematical formulation"
    )
    overall_priority_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Combined priority score for ranking"
    )
    
    # ========================================================================
    # MATHEMATICAL VALIDATION (Restored from original system)
    # ========================================================================
    mathematical_validity: str = Field(
        default="",
        description="Validation notes about mathematical correctness"
    )
    
    # ========================================================================
    # ORGANIZATION & RANKING (Restored from original system)
    # ========================================================================
    organization_rank: int = Field(
        default=0,
        description="Rank within its category (1 = most important)"
    )
    importance_category: str = Field(
        default="",
        description="Importance classification (critical_correctness, essential_boundary, etc.)"
    )
    selection_reasoning: str = Field(
        default="",
        description="Why this postcondition was selected/ranked this way"
    )
    robustness_assessment: str = Field(
        default="",
        description="Detailed assessment of robustness characteristics"
    )
    is_primary_in_category: bool = Field(
        default=False,
        description="Whether this is the primary/most important in its category"
    )
    recommended_for_selection: bool = Field(
        default=True,
        description="Whether this should be included in final selection"
    )
    
    # ========================================================================
    # Z3 THEORY (Existing)
    # ========================================================================
    z3_theory: Optional[str] = Field(
        default=None,
        description="Z3 theory to use for verification (arrays, arithmetic, etc.)"
    )
    
    # ========================================================================
    # WARNINGS & NOTES (Existing)
    # ========================================================================
    warnings: List[str] = Field(
        default=[],
        description="Warnings or caveats about this postcondition"
    )
    
    # ========================================================================
    # COMPUTED PROPERTIES
    # ========================================================================
    @property
    def overall_quality_score(self) -> float:
        """
        Calculate overall quality score.
        
        Weighted average of all quality metrics.
        """
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "formal_text": "∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
                "natural_language": "Array is sorted in ascending order",
                "precise_translation": "For every pair of indices i and j where i comes before j...",
                "reasoning": "This ensures the fundamental sorting property...",
                "strength": "standard",
                "category": "correctness",
                "confidence_score": 0.95,
                "robustness_score": 0.92,
                "edge_cases_covered": ["empty array", "single element"],
                "z3_theory": "arrays"
            }
        }


# ============================================================================
# ENHANCED Z3 TRANSLATION MODEL
# ============================================================================

class Z3Translation(BaseModel):
    """
    Result of translating a postcondition to Z3.
    
    ENHANCED: Added validation metadata and analysis fields.
    """
    formal_text: str
    natural_language: str
    
    # Z3 Code
    z3_code: str = ""
    z3_theory_used: str = "unknown"
    
    # Translation Status
    translation_success: bool = False
    translation_time: float = 0.0
    
    # ========================================================================
    # VALIDATION FIELDS (Enhanced from original system)
    # ========================================================================
    z3_validation_passed: bool = Field(
        default=False,
        description="Whether the Z3 code passed syntax validation"
    )
    z3_validation_status: str = Field(
        default="not_validated",
        description="Validation status: success, syntax_error, runtime_error, not_validated"
    )
    validation_error: Optional[str] = Field(
        default=None,
        description="Error message if validation failed"
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
    custom_functions: Optional[List[str]] = Field(
        default=None,
        description="List of custom functions defined in Z3 code"
    )
    declared_sorts: Optional[List[str]] = Field(
        default=None,
        description="Z3 sorts declared (Int, Array, etc.)"
    )
    declared_variables: Optional[Dict[str, str]] = Field(
        default=None,
        description="Variables declared with their types"
    )
    
    # ========================================================================
    # METADATA
    # ========================================================================
    warnings: List[str] = []
    execution_time: float = 0.0
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# PIPELINE RESULT MODELS (Enhanced)
# ============================================================================

class FunctionResult(BaseModel):
    """Result for a single function's postcondition generation."""
    function_name: str
    function_signature: str = ""
    function_description: str = ""
    
    pseudocode: Optional[Function] = None
    postconditions: List[EnhancedPostcondition] = []
    z3_translations: List[Z3Translation] = []
    
    # Counts
    postcondition_count: int = 0
    z3_success_count: int = 0
    z3_validated_count: int = 0
    
    # ========================================================================
    # ENHANCED METRICS (Restored from original system)
    # ========================================================================
    average_quality_score: float = 0.0
    average_robustness_score: float = Field(
        default=0.0,
        description="Average robustness score across postconditions"
    )
    edge_case_coverage_score: float = 0.0
    mathematical_validity_rate: float = Field(
        default=0.0,
        description="Ratio of mathematically valid postconditions"
    )
    
    processing_time: float = 0.0
    errors: List[str] = []


class CompleteEnhancedResult(BaseModel):
    """Complete result from the entire pipeline."""
    session_id: str
    specification: str
    
    # Status
    overall_status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    
    # Pseudocode stage
    pseudocode_success: bool = False
    pseudocode_raw_output: Optional[PseudocodeResult] = None
    pseudocode_error: Optional[str] = None
    functions_created: List[str] = []
    
    # Postcondition stage
    function_results: List[FunctionResult] = []
    
    # Statistics
    total_postconditions: int = 0
    total_z3_translations: int = 0
    successful_z3_translations: int = 0
    validated_z3_translations: int = 0
    
    # Metadata
    codebase_path: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.now)
    total_processing_time: float = 0.0
    
    errors: List[str] = []
    warnings: List[str] = []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# EDGE CASE MODELS
# ============================================================================

class EdgeCaseAnalysis(BaseModel):
    """Result of edge case analysis."""
    input_edge_cases: List[str] = []
    output_edge_cases: List[str] = []
    algorithmic_edge_cases: List[str] = []
    mathematical_edge_cases: List[str] = []
    boundary_conditions: List[str] = []
    error_conditions: List[str] = []
    performance_edge_cases: List[str] = []
    domain_specific_cases: List[str] = []
    
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_assessment: str = ""
    
    @property
    def all_edge_cases(self) -> List[str]:
        """Get all edge cases combined."""
        return (
            self.input_edge_cases +
            self.output_edge_cases +
            self.algorithmic_edge_cases +
            self.mathematical_edge_cases +
            self.boundary_conditions +
            self.error_conditions +
            self.performance_edge_cases +
            self.domain_specific_cases
        )
    
    @property
    def total_count(self) -> int:
        """Total number of edge cases identified."""
        return len(self.all_edge_cases)


# ============================================================================
# MIGRATION VALIDATION
# ============================================================================

def validate_migration() -> bool:
    """
    Validate that all old system fields are present.
    
    Returns:
        True if migration is complete
    """
    required_fields = [
        # Translation fields
        'precise_translation',
        'reasoning',
        # Edge case fields
        'edge_cases_covered',
        'coverage_gaps',
        # Robustness fields
        'robustness_score',
        'mathematical_quality_score',
        'overall_priority_score',
        # Validation fields
        'mathematical_validity',
        # Organization fields
        'organization_rank',
        'importance_category',
        'selection_reasoning',
        'robustness_assessment',
        'is_primary_in_category',
        'recommended_for_selection',
    ]
    
    postcondition_fields = set(EnhancedPostcondition.model_fields.keys())
    
    missing = [f for f in required_fields if f not in postcondition_fields]
    
    if missing:
        print(f"❌ Missing fields: {missing}")
        return False
    
    print("✅ All required fields present")
    return True


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED MODELS - PHASE 1 MIGRATION VALIDATION")
    print("=" * 70)
    
    # Validate migration
    print("\n📋 Checking migration completeness...")
    validate_migration()
    
    # Example: Create enhanced postcondition
    print("\n📝 Creating enhanced postcondition with all fields...")
    
    postcondition = EnhancedPostcondition(
        formal_text="∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
        natural_language="Array is sorted in ascending order",
        precise_translation="For every pair of indices i and j where i comes before j, the element at position i is less than or equal to the element at position j",
        reasoning="This ensures the fundamental sorting property and prevents out-of-order elements",
        strength=PostconditionStrength.STANDARD,
        category=PostconditionCategory.CORE_CORRECTNESS,
        confidence_score=0.95,
        clarity_score=0.9,
        completeness_score=0.85,
        testability_score=0.9,
        robustness_score=0.92,
        mathematical_quality_score=0.93,
        edge_cases_covered=[
            "Empty array: trivially sorted",
            "Single element: no comparison needed",
            "Duplicate elements: equality handled by ≤"
        ],
        coverage_gaps=[
            "Does not guarantee in-place sorting",
            "Does not specify time complexity"
        ],
        mathematical_validity="Mathematically valid - uses proper universal quantification",
        organization_rank=1,
        importance_category="critical_correctness",
        is_primary_in_category=True,
        z3_theory="arrays"
    )
    
    print(f"✅ Created postcondition: {postcondition.natural_language}")
    print(f"   Overall quality: {postcondition.overall_quality_score:.2f}")
    print(f"   Edge case coverage: {postcondition.edge_case_coverage_ratio:.2f}")
    print(f"   Has translations: {postcondition.has_translations}")
    
    # Show JSON output
    print("\n📄 Sample JSON output (first 500 chars):")
    json_output = postcondition.model_dump_json(indent=2)
    print(json_output[:500] + "...")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 1 MIGRATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. ✅ Enhanced models with all fields")
    print("2. ⏭️  Update prompts to request all fields")
    print("3. ⏭️  Enhance chain parsing to capture all fields")
    print("4. ⏭️  Add translation chain")
    print("5. ⏭️  Enhance Z3 validation")