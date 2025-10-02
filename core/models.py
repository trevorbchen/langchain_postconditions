"""
Enhanced Storage Models - Comprehensive Metadata Persistence

This replaces your minimal storage models with rich data structures
that preserve ALL generated information from the pipeline.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


@dataclass
class StoredPostcondition(BaseModel):
    """
    Rich storage model for postconditions.
    Preserves ALL data from EnhancedPostcondition.
    """
    # Core identification
    formal_text: str
    natural_language: str
    strength: str = "standard"
    category: str = "core_correctness"
    
    # Rich translations & explanations
    precise_translation: str = ""
    reasoning: str = ""
    
    # Edge case analysis
    edge_cases: List[str] = []
    edge_cases_covered: List[str] = []
    coverage_gaps: List[str] = []
    
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
    
    # Z3 translation data
    z3_translation: Optional[Dict[str, Any]] = None
    
    # Metadata
    generated_at: str = ""
    generation_session_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "formal_text": self.formal_text,
            "natural_language": self.natural_language,
            "precise_translation": self.precise_translation,
            "reasoning": self.reasoning,
            "strength": self.strength,
            "category": self.category,
            
            # Edge cases
            "edge_cases": self.edge_cases,
            "edge_cases_covered": self.edge_cases_covered,
            "coverage_gaps": self.coverage_gaps,
            
            # Quality scores
            "confidence_score": self.confidence_score,
            "robustness_score": self.robustness_score,
            "clarity_score": self.clarity_score,
            "completeness_score": self.completeness_score,
            "testability_score": self.testability_score,
            "mathematical_quality_score": self.mathematical_quality_score,
            "overall_quality_score": self.overall_quality_score,
            
            # Mathematical
            "mathematical_validity": self.mathematical_validity,
            "z3_theory": self.z3_theory,
            
            # Organization
            "organization_rank": self.organization_rank,
            "importance_category": self.importance_category,
            "selection_reasoning": self.selection_reasoning,
            "robustness_assessment": self.robustness_assessment,
            "is_primary_in_category": self.is_primary_in_category,
            "recommended_for_selection": self.recommended_for_selection,
            
            # Z3
            "z3_translation": self.z3_translation,
            
            # Metadata
            "generated_at": self.generated_at,
            "generation_session_id": self.generation_session_id
        }


@dataclass
class StoredZ3Translation(BaseModel):
    """Rich Z3 translation storage model."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "formal_text": self.formal_text,
            "natural_language": self.natural_language,
            "z3_code": self.z3_code,
            "z3_validation_passed": self.z3_validation_passed,
            "z3_validation_status": self.z3_validation_status,
            "validation_error": self.validation_error,
            "z3_ast": self.z3_ast,
            "tokens": self.tokens,
            "custom_functions": self.custom_functions,
            "declared_sorts": self.declared_sorts,
            "declared_variables": self.declared_variables,
            "translation_success": self.translation_success,
            "translation_time": self.translation_time,
            "generated_at": self.generated_at,
            "solver_created": self.solver_created,
            "constraints_added": self.constraints_added,
            "variables_declared": self.variables_declared,
            "execution_time": self.execution_time,
            "runtime_error": self.runtime_error
        }


@dataclass
class GenerationSnapshot:
    """
    Complete snapshot of a single generation run.
    Preserves EVERYTHING from that specific generation.
    """
    session_id: str
    timestamp: str
    specification: str
    
    # Input context
    function_signature: str
    function_description: str
    function_body: Optional[str] = None
    
    # Generated postconditions (full rich data)
    postconditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregate metrics for this generation
    total_postconditions: int = 0
    average_confidence: float = 0.0
    average_robustness: float = 0.0
    average_quality: float = 0.0
    
    # Edge case analysis
    total_edge_cases_covered: int = 0
    unique_edge_cases: List[str] = field(default_factory=list)
    coverage_gaps_found: List[str] = field(default_factory=list)
    
    # Z3 analysis
    z3_translations_generated: int = 0
    z3_validations_passed: int = 0
    z3_validations_failed: int = 0
    z3_validation_errors: List[str] = field(default_factory=list)
    z3_theories_used: List[str] = field(default_factory=list)
    
    # Performance metrics
    generation_time: float = 0.0
    translation_time: float = 0.0
    validation_time: float = 0.0
    total_time: float = 0.0
    
    # Status
    status: str = "success"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "specification": self.specification,
            "function_signature": self.function_signature,
            "function_description": self.function_description,
            "function_body": self.function_body,
            "postconditions": self.postconditions,
            "total_postconditions": self.total_postconditions,
            "average_confidence": self.average_confidence,
            "average_robustness": self.average_robustness,
            "average_quality": self.average_quality,
            "total_edge_cases_covered": self.total_edge_cases_covered,
            "unique_edge_cases": self.unique_edge_cases,
            "coverage_gaps_found": self.coverage_gaps_found,
            "z3_translations_generated": self.z3_translations_generated,
            "z3_validations_passed": self.z3_validations_passed,
            "z3_validations_failed": self.z3_validations_failed,
            "z3_validation_errors": self.z3_validation_errors,
            "z3_theories_used": self.z3_theories_used,
            "generation_time": self.generation_time,
            "translation_time": self.translation_time,
            "validation_time": self.validation_time,
            "total_time": self.total_time,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings
        }


@dataclass
class ComprehensiveFunctionMetadata:
    """
    THIS IS THE COMPREHENSIVE METADATA YOU WANT!
    
    Replaces your minimal metadata with complete historical tracking
    of all generations, postconditions, and analysis results.
    """
    # Core identification
    function_name: str
    function_signature: str
    function_description: str
    
    # Timestamps
    first_seen: str
    last_updated: str
    
    # Generation history (ALL snapshots preserved)
    total_generations: int = 0
    generations_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregate postcondition data
    total_postconditions_ever_generated: int = 0
    unique_postconditions_count: int = 0
    
    # Current "best" postconditions (from latest or best generation)
    current_postconditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality trends over generations
    quality_trend: List[float] = field(default_factory=list)
    robustness_trend: List[float] = field(default_factory=list)
    confidence_trend: List[float] = field(default_factory=list)
    
    # Aggregate Z3 statistics
    total_z3_translations: int = 0
    total_z3_validations_passed: int = 0
    total_z3_validations_failed: int = 0
    z3_theories_encountered: List[str] = field(default_factory=list)
    
    # Edge case intelligence
    all_edge_cases_covered: List[str] = field(default_factory=list)
    recurring_coverage_gaps: List[str] = field(default_factory=list)
    edge_case_coverage_improvement: float = 0.0
    
    # Performance metrics
    average_generation_time: float = 0.0
    fastest_generation_time: float = 0.0
    slowest_generation_time: float = 0.0
    
    # Best performing generation
    best_generation_session_id: Optional[str] = None
    best_generation_quality_score: float = 0.0
    
    # Status tracking
    status: str = "active"
    last_error: Optional[str] = None
    warnings_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to comprehensive dictionary for storage."""
        return {
            # Core
            "function_name": self.function_name,
            "function_signature": self.function_signature,
            "function_description": self.function_description,
            
            # Timestamps
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
            
            # Generation tracking
            "total_generations": self.total_generations,
            "generations_history": self.generations_history,
            
            # Postconditions
            "total_postconditions_ever_generated": self.total_postconditions_ever_generated,
            "unique_postconditions_count": self.unique_postconditions_count,
            "current_postconditions": self.current_postconditions,
            
            # Trends
            "quality_trend": self.quality_trend,
            "robustness_trend": self.robustness_trend,
            "confidence_trend": self.confidence_trend,
            
            # Z3
            "total_z3_translations": self.total_z3_translations,
            "total_z3_validations_passed": self.total_z3_validations_passed,
            "total_z3_validations_failed": self.total_z3_validations_failed,
            "z3_theories_encountered": self.z3_theories_encountered,
            
            # Edge cases
            "all_edge_cases_covered": self.all_edge_cases_covered,
            "recurring_coverage_gaps": self.recurring_coverage_gaps,
            "edge_case_coverage_improvement": self.edge_case_coverage_improvement,
            
            # Performance
            "average_generation_time": self.average_generation_time,
            "fastest_generation_time": self.fastest_generation_time,
            "slowest_generation_time": self.slowest_generation_time,
            
            # Best generation
            "best_generation_session_id": self.best_generation_session_id,
            "best_generation_quality_score": self.best_generation_quality_score,
            
            # Status
            "status": self.status,
            "last_error": self.last_error,
            "warnings_count": self.warnings_count
        }
    
    def add_generation(self, snapshot: GenerationSnapshot) -> None:
        """Add a new generation snapshot and update aggregate metrics."""
        self.total_generations += 1
        self.generations_history.append(snapshot.to_dict())
        self.last_updated = snapshot.timestamp
        
        # Update postcondition counts
        self.total_postconditions_ever_generated += snapshot.total_postconditions
        
        # Update trends
        self.quality_trend.append(snapshot.average_quality)
        self.robustness_trend.append(snapshot.average_robustness)
        self.confidence_trend.append(snapshot.average_confidence)
        
        # Update Z3 stats
        self.total_z3_translations += snapshot.z3_translations_generated
        self.total_z3_validations_passed += snapshot.z3_validations_passed
        self.total_z3_validations_failed += snapshot.z3_validations_failed
        
        # Update edge cases
        for edge_case in snapshot.unique_edge_cases:
            if edge_case not in self.all_edge_cases_covered:
                self.all_edge_cases_covered.append(edge_case)
        
        # Update performance metrics
        times = [g["total_time"] for g in self.generations_history]
        self.average_generation_time = sum(times) / len(times)
        self.fastest_generation_time = min(times)
        self.slowest_generation_time = max(times)
        
        # Track best generation
        if snapshot.average_quality > self.best_generation_quality_score:
            self.best_generation_quality_score = snapshot.average_quality
            self.best_generation_session_id = snapshot.session_id
        
        # Update current postconditions (use latest)
        self.current_postconditions = snapshot.postconditions


# Conversion utilities
def convert_enhanced_postcondition_to_stored(
    pc: Any,  # EnhancedPostcondition from core/models.py
    session_id: str
) -> StoredPostcondition:
    """Convert EnhancedPostcondition to StoredPostcondition."""
    z3_trans_dict = None
    if hasattr(pc, 'z3_translation') and pc.z3_translation:
        z3_trans_dict = pc.z3_translation.model_dump() if hasattr(pc.z3_translation, 'model_dump') else vars(pc.z3_translation)
    
    return StoredPostcondition(
        formal_text=pc.formal_text,
        natural_language=pc.natural_language,
        precise_translation=getattr(pc, 'precise_translation', ''),
        reasoning=getattr(pc, 'reasoning', ''),
        strength=getattr(pc, 'strength', 'standard'),
        category=getattr(pc, 'category', 'core_correctness'),
        edge_cases=pc.edge_cases if hasattr(pc, 'edge_cases') else [],
        edge_cases_covered=getattr(pc, 'edge_cases_covered', []),
        coverage_gaps=getattr(pc, 'coverage_gaps', []),
        confidence_score=pc.confidence_score,
        robustness_score=getattr(pc, 'robustness_score', 0.0),
        clarity_score=getattr(pc, 'clarity_score', 0.0),
        completeness_score=getattr(pc, 'completeness_score', 0.0),
        testability_score=getattr(pc, 'testability_score', 0.0),
        mathematical_quality_score=getattr(pc, 'mathematical_quality_score', 0.0),
        overall_quality_score=getattr(pc, 'overall_quality_score', 0.0),
        mathematical_validity=getattr(pc, 'mathematical_validity', ''),
        z3_theory=pc.z3_theory if hasattr(pc, 'z3_theory') else 'unknown',
        organization_rank=getattr(pc, 'organization_rank', 0),
        importance_category=getattr(pc, 'importance_category', ''),
        selection_reasoning=getattr(pc, 'selection_reasoning', ''),
        robustness_assessment=getattr(pc, 'robustness_assessment', ''),
        is_primary_in_category=getattr(pc, 'is_primary_in_category', False),
        recommended_for_selection=getattr(pc, 'recommended_for_selection', True),
        z3_translation=z3_trans_dict,
        generated_at=datetime.now().isoformat(),
        generation_session_id=session_id
    )