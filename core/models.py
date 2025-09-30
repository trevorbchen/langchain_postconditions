"""
Pydantic Data Models for Postcondition Generation System

This module defines all data structures used throughout the system.
Using Pydantic provides:
- Automatic validation
- Type safety
- Easy JSON serialization
- Clear documentation
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import field_validator
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "bubble_sort",
                "description": "Sort array using bubble sort algorithm",
                "signature": "void bubble_sort(int* arr, int size)",
                "return_type": "void",
                "complexity": "O(n²)",
                "memory_usage": "O(1)"
            }
        }


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
    
    class Config:
        json_schema_extra = {
            "example": {
                "functions": [
                    {
                        "name": "sort_array",
                        "description": "Sorts an array in ascending order"
                    }
                ],
                "structs": [],
                "includes": ["stdio.h", "stdlib.h"]
            }
        }


# ============================================================================
# POSTCONDITION MODELS
# ============================================================================

class EnhancedPostcondition(BaseModel):
    """Enhanced postcondition with metadata."""
    formal_text: str
    natural_language: str
    
    strength: PostconditionStrength = PostconditionStrength.STANDARD
    category: PostconditionCategory = PostconditionCategory.CORRECTNESS
    
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    clarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    testability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    edge_cases: List[str] = []
    z3_theory: Optional[str] = None
    reasoning: str = ""
    
    warnings: List[str] = []
    
    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score."""
        if self.clarity_score == 0 and self.completeness_score == 0:
            return self.confidence_score
        
        scores = [
            self.confidence_score,
            self.clarity_score,
            self.completeness_score,
            self.testability_score
        ]
        non_zero_scores = [s for s in scores if s > 0]
        
        if not non_zero_scores:
            return 0.0
        
        return sum(non_zero_scores) / len(non_zero_scores)
    
    class Config:
        json_schema_extra = {
            "example": {
                "formal_text": "∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
                "natural_language": "The array is sorted in ascending order",
                "strength": "standard",
                "category": "correctness",
                "confidence_score": 0.95,
                "z3_theory": "arrays"
            }
        }


# ============================================================================
# Z3 TRANSLATION MODELS
# ============================================================================

class Z3Translation(BaseModel):
    """Result of translating a postcondition to Z3."""
    formal_text: str
    natural_language: str
    
    z3_code: str = ""
    z3_theory_used: str = "unknown"
    
    translation_success: bool = False
    z3_validation_passed: bool = False
    z3_validation_status: str = "not_validated"
    
    validation_error: Optional[str] = None
    warnings: List[str] = []
    
    execution_time: float = 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "formal_text": "∀i: arr[i] ≤ arr[i+1]",
                "natural_language": "Array is sorted",
                "z3_code": "from z3 import *\n...",
                "translation_success": True,
                "z3_validation_passed": True
            }
        }


# ============================================================================
# PIPELINE RESULT MODELS
# ============================================================================

class FunctionResult(BaseModel):
    """Result for a single function's postcondition generation."""
    function_name: str
    function_signature: str = ""
    function_description: str = ""
    
    pseudocode: Optional[Function] = None
    postconditions: List[EnhancedPostcondition] = []
    z3_translations: List[Z3Translation] = []
    
    postcondition_count: int = 0
    z3_success_count: int = 0
    z3_validated_count: int = 0
    
    average_quality_score: float = 0.0
    edge_case_coverage_score: float = 0.0
    
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
# HELPER FUNCTIONS
# ============================================================================

def create_function_parameter(
    name: str,
    data_type: str,
    description: str = ""
) -> FunctionParameter:
    """Helper to create a function parameter."""
    return FunctionParameter(
        name=name,
        data_type=data_type,
        description=description
    )


def create_simple_function(
    name: str,
    description: str,
    params: List[tuple] = None
) -> Function:
    """
    Helper to create a simple function.
    
    Args:
        name: Function name
        description: Function description
        params: List of (name, type, description) tuples
    
    Returns:
        Function object
    """
    input_params = []
    if params:
        for param in params:
            if len(param) == 3:
                input_params.append(
                    FunctionParameter(
                        name=param[0],
                        data_type=param[1],
                        description=param[2]
                    )
                )
    
    return Function(
        name=name,
        description=description,
        input_parameters=input_params
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a function
    sort_func = Function(
        name="bubble_sort",
        description="Sort array using bubble sort",
        return_type="void",
        input_parameters=[
            FunctionParameter(
                name="arr",
                data_type="int*",
                description="Array to sort",
                is_pointer=True
            ),
            FunctionParameter(
                name="size",
                data_type="int",
                description="Size of array"
            )
        ],
        complexity="O(n²)",
        memory_usage="O(1)"
    )
    
    print("Function created:", sort_func.name)
    print("JSON output:")
    print(sort_func.model_dump_json(indent=2))
    
    # Example: Create a postcondition
    postcondition = EnhancedPostcondition(
        formal_text="∀i,j: 0 ≤ i < j < size → arr[i] ≤ arr[j]",
        natural_language="Array is sorted in ascending order",
        strength=PostconditionStrength.STANDARD,
        category=PostconditionCategory.CORRECTNESS,
        confidence_score=0.95,
        clarity_score=0.9,
        completeness_score=0.85,
        testability_score=0.9,
        edge_cases=["Empty array", "Single element", "Already sorted"],
        z3_theory="arrays"
    )
    
    print("\nPostcondition created")
    print(f"Quality score: {postcondition.overall_quality_score:.2f}")
    print("JSON output:")
    print(postcondition.model_dump_json(indent=2))