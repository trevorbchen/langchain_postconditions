"""
Enhanced Database Manager - Comprehensive Data Persistence

Saves ALL rich data from pipeline including full generation history,
quality trends, and detailed analysis results.

This module provides:
- EnhancedDatabaseManager: Main storage class with comprehensive tracking
- ResultStorage: Alias for backward compatibility with pipeline.py
- Helper classes for metadata snapshots and function tracking
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

from core.models import (
    EnhancedPostcondition,
    Z3Translation,
    FunctionResult,
    CompleteEnhancedResult
)

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER CLASSES FOR COMPREHENSIVE METADATA
# ============================================================================

@dataclass
class GenerationSnapshot:
    """Complete snapshot of a single generation run."""
    session_id: str
    timestamp: str
    specification: str
    function_signature: str
    function_description: str
    function_body: Optional[str] = None
    postconditions: List[Dict[str, Any]] = field(default_factory=list)
    total_postconditions: int = 0
    average_confidence: float = 0.0
    average_robustness: float = 0.0
    average_quality: float = 0.0
    total_edge_cases_covered: int = 0
    unique_edge_cases: List[str] = field(default_factory=list)
    coverage_gaps_found: List[str] = field(default_factory=list)
    z3_translations_generated: int = 0
    z3_validations_passed: int = 0
    z3_validations_failed: int = 0
    z3_validation_errors: List[str] = field(default_factory=list)
    z3_theories_used: List[str] = field(default_factory=list)
    generation_time: float = 0.0
    translation_time: float = 0.0
    validation_time: float = 0.0
    total_time: float = 0.0
    status: str = "success"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """Comprehensive metadata tracking all generations for a function."""
    function_name: str
    function_signature: str
    function_description: str
    first_seen: str
    last_updated: str
    total_generations: int = 0
    generations_history: List[Dict[str, Any]] = field(default_factory=list)
    total_postconditions_ever_generated: int = 0
    unique_postconditions_count: int = 0
    current_postconditions: List[Dict[str, Any]] = field(default_factory=list)
    quality_trend: List[float] = field(default_factory=list)
    robustness_trend: List[float] = field(default_factory=list)
    confidence_trend: List[float] = field(default_factory=list)
    total_z3_translations: int = 0
    total_z3_validations_passed: int = 0
    total_z3_validations_failed: int = 0
    z3_theories_encountered: List[str] = field(default_factory=list)
    all_edge_cases_covered: List[str] = field(default_factory=list)
    recurring_coverage_gaps: List[str] = field(default_factory=list)
    edge_case_coverage_improvement: float = 0.0
    average_generation_time: float = 0.0
    fastest_generation_time: float = 0.0
    slowest_generation_time: float = 0.0
    best_generation_session_id: Optional[str] = None
    best_generation_quality_score: float = 0.0
    status: str = "active"
    last_error: Optional[str] = None
    warnings_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "function_signature": self.function_signature,
            "function_description": self.function_description,
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
            "total_generations": self.total_generations,
            "generations_history": self.generations_history,
            "total_postconditions_ever_generated": self.total_postconditions_ever_generated,
            "unique_postconditions_count": self.unique_postconditions_count,
            "current_postconditions": self.current_postconditions,
            "quality_trend": self.quality_trend,
            "robustness_trend": self.robustness_trend,
            "confidence_trend": self.confidence_trend,
            "total_z3_translations": self.total_z3_translations,
            "total_z3_validations_passed": self.total_z3_validations_passed,
            "total_z3_validations_failed": self.total_z3_validations_failed,
            "z3_theories_encountered": self.z3_theories_encountered,
            "all_edge_cases_covered": self.all_edge_cases_covered,
            "recurring_coverage_gaps": self.recurring_coverage_gaps,
            "edge_case_coverage_improvement": self.edge_case_coverage_improvement,
            "average_generation_time": self.average_generation_time,
            "fastest_generation_time": self.fastest_generation_time,
            "slowest_generation_time": self.slowest_generation_time,
            "best_generation_session_id": self.best_generation_session_id,
            "best_generation_quality_score": self.best_generation_quality_score,
            "status": self.status,
            "last_error": self.last_error,
            "warnings_count": self.warnings_count
        }
    
    def add_generation(self, snapshot: GenerationSnapshot) -> None:
        """Add a generation snapshot and update aggregate metrics."""
        self.total_generations += 1
        self.generations_history.append(snapshot.to_dict())
        self.last_updated = snapshot.timestamp
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
        self.fastest_generation_time = min(times) if times else 0.0
        self.slowest_generation_time = max(times) if times else 0.0
        
        # Track best generation
        if snapshot.average_quality > self.best_generation_quality_score:
            self.best_generation_quality_score = snapshot.average_quality
            self.best_generation_session_id = snapshot.session_id
        
        # Update current postconditions
        self.current_postconditions = snapshot.postconditions


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_postcondition_to_dict(pc: Any, session_id: str) -> Dict[str, Any]:
    """
    Convert EnhancedPostcondition to dictionary with all rich fields.
    
    Uses getattr with defaults to handle both old and new postcondition models.
    This ensures backward compatibility while supporting enhanced fields.
    """
    # Handle Z3 translation serialization
    z3_trans_dict = None
    if hasattr(pc, 'z3_translation') and pc.z3_translation:
        if hasattr(pc.z3_translation, 'model_dump'):
            z3_trans_dict = pc.z3_translation.model_dump()
        elif hasattr(pc.z3_translation, 'dict'):
            z3_trans_dict = pc.z3_translation.dict()
        else:
            z3_trans_dict = vars(pc.z3_translation)
    
    return {
        # Core fields
        "formal_text": pc.formal_text,
        "natural_language": pc.natural_language,
        
        # Rich translations & explanations
        "precise_translation": getattr(pc, 'precise_translation', ''),
        "reasoning": getattr(pc, 'reasoning', ''),
        
        # Categorization
        "strength": getattr(pc, 'strength', 'standard'),
        "category": getattr(pc, 'category', 'core_correctness'),
        
        # Edge case analysis
        "edge_cases": getattr(pc, 'edge_cases', []),
        "edge_cases_covered": getattr(pc, 'edge_cases_covered', []),
        "coverage_gaps": getattr(pc, 'coverage_gaps', []),
        
        # Quality metrics
        "confidence_score": pc.confidence_score,
        "robustness_score": getattr(pc, 'robustness_score', 0.0),
        "clarity_score": getattr(pc, 'clarity_score', 0.0),
        "completeness_score": getattr(pc, 'completeness_score', 0.0),
        "testability_score": getattr(pc, 'testability_score', 0.0),
        "mathematical_quality_score": getattr(pc, 'mathematical_quality_score', 0.0),
        "overall_quality_score": getattr(pc, 'overall_quality_score', 0.0),
        
        # Mathematical properties
        "mathematical_validity": getattr(pc, 'mathematical_validity', ''),
        "z3_theory": getattr(pc, 'z3_theory', 'unknown'),
        
        # Organization fields
        "organization_rank": getattr(pc, 'organization_rank', 0),
        "importance_category": getattr(pc, 'importance_category', ''),
        "selection_reasoning": getattr(pc, 'selection_reasoning', ''),
        "robustness_assessment": getattr(pc, 'robustness_assessment', ''),
        "is_primary_in_category": getattr(pc, 'is_primary_in_category', False),
        "recommended_for_selection": getattr(pc, 'recommended_for_selection', True),
        
        # Z3 translation
        "z3_translation": z3_trans_dict,
        
        # Metadata
        "generated_at": getattr(pc, 'generated_at', datetime.now().isoformat()),
        "generation_session_id": session_id
    }


# ============================================================================
# MAIN DATABASE MANAGER
# ============================================================================

class EnhancedDatabaseManager:
    """
    Manages comprehensive persistence of ALL generated data.
    
    Features:
    - Full generation history tracking
    - Rich postcondition metadata with all enhanced fields
    - Quality trends over time
    - Z3 validation tracking
    - Edge case intelligence
    - Performance metrics
    - Per-function and per-session storage
    
    Storage Structure:
        base_dir/
            functions/
                function_name.json  # Comprehensive metadata per function
            sessions/
                session_id.json     # Complete snapshot per pipeline run
    """
    
    def __init__(self, base_dir: Path = None):
        """
        Initialize database manager.
        
        Args:
            base_dir: Base directory for storage (default: "data")
                     Can be string or Path object for compatibility
        """
        # Handle both Path and string inputs for compatibility
        if base_dir is None:
            self.base_dir = Path("data")
        elif isinstance(base_dir, str):
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = base_dir
            
        self.functions_dir = self.base_dir / "functions"
        self.sessions_dir = self.base_dir / "sessions"
        
        # Create directories
        self.functions_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Enhanced database initialized at {self.base_dir}")
    
    def save_pipeline_result(
        self,
        session_id: str,
        specification: str,
        function_results: List[FunctionResult],
        total_time: float
    ) -> None:
        """
        Save complete pipeline results with comprehensive metadata.
        
        This is the main entry point for saving results. It:
        1. Saves a complete session snapshot (all functions, aggregates)
        2. Updates per-function metadata with generation history
        
        Args:
            session_id: Unique session identifier
            specification: Original specification text
            function_results: List of function results from pipeline
            total_time: Total processing time in seconds
        """
        logger.info(f"Saving pipeline result for session {session_id}")
        
        # Save full session data
        self._save_session_snapshot(
            session_id=session_id,
            specification=specification,
            function_results=function_results,
            total_time=total_time
        )
        
        # Save/update per-function metadata
        for func_result in function_results:
            self._save_function_metadata(
                session_id=session_id,
                specification=specification,
                function_result=func_result
            )
        
        logger.info(f"Pipeline result saved successfully (session: {session_id})")
    
    def save_results(self, session_id: str, result: Any) -> None:
        """
        Save complete pipeline results (compatibility wrapper for pipeline.py).
        
        This method provides backward compatibility with pipeline.py which
        calls save_results() instead of save_pipeline_result().
        
        Args:
            session_id: Unique session identifier
            result: CompleteEnhancedResult object from pipeline
        """
        logger.info(f"Saving results via compatibility wrapper for session {session_id}")
        
        try:
            # Extract data from CompleteEnhancedResult object
            specification = getattr(result, 'specification', '')
            function_results = getattr(result, 'function_results', [])
            total_time = getattr(result, 'total_processing_time', 0.0)
            
            # Validate we have the minimum required data
            if not specification and not function_results:
                logger.warning(f"No data to save for session {session_id}")
                return
            
            # Call the main save method
            self.save_pipeline_result(
                session_id=session_id,
                specification=specification,
                function_results=function_results,
                total_time=total_time
            )
            
            logger.info(f"Successfully saved results for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error in save_results compatibility wrapper: {e}")
            logger.exception("Full traceback:")
            raise
    
    def save_function_results(
        self, 
        session_id: str, 
        function_results: List[FunctionResult],
        function_name: Optional[str] = None,  # 🆕 ADDED: Accept this parameter
        specification: Optional[str] = None    # 🆕 ADDED: Accept this parameter
    ) -> None:
        """
        Save function results (compatibility wrapper for pipeline.py).
        
        Another compatibility method that pipeline.py may call.
        
        Args:
            session_id: Unique session identifier
            function_results: List of function results to save
            function_name: Optional function name (for backward compatibility)
            specification: Optional specification (for backward compatibility)
        """
        logger.info(f"Saving {len(function_results)} function results for session {session_id}")
        
        try:
            spec = specification or ""
            
            for func_result in function_results:
                self._save_function_metadata(
                    session_id=session_id,
                    specification=spec,
                    function_result=func_result
                )
            logger.info(f"Successfully saved {len(function_results)} function results")
        except Exception as e:
            logger.error(f"Error saving function results: {e}")
            logger.exception("Full traceback:")
            raise
    
    def _save_session_snapshot(
        self,
        session_id: str,
        specification: str,
        function_results: List[FunctionResult],
        total_time: float
    ) -> None:
        """Save complete session snapshot with all functions and aggregates."""
        session_file = self.sessions_dir / f"{session_id}.json"
        
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "specification": specification,
            "total_functions": len(function_results),
            "total_time": total_time,
            "functions": []
        }
        
        # Add function results
        for func_result in function_results:
            func_data = self._serialize_function_result(func_result, session_id)
            session_data["functions"].append(func_data)
        
        # Calculate session-level aggregates
        session_data["aggregate_metrics"] = self._calculate_session_aggregates(
            function_results
        )
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session snapshot saved: {session_file}")
    
    def _save_function_metadata(
        self,
        session_id: str,
        specification: str,
        function_result: FunctionResult
    ) -> None:
        """Save/update comprehensive function metadata with generation history."""
        func_name = function_result.function_name
        metadata_file = self.functions_dir / f"{func_name}.json"
        
        # Load existing or create new metadata
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = self._dict_to_metadata(data)
        else:
            metadata = ComprehensiveFunctionMetadata(
                function_name=func_name,
                function_signature=function_result.function_signature,
                function_description=function_result.function_description,
                first_seen=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
        
        # Create generation snapshot
        snapshot = self._create_generation_snapshot(
            session_id=session_id,
            specification=specification,
            function_result=function_result
        )
        
        # Add to metadata (updates trends, stats, etc.)
        metadata.add_generation(snapshot)
        
        # Save updated metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Function metadata updated: {metadata_file}")
    
    def _create_generation_snapshot(
        self,
        session_id: str,
        specification: str,
        function_result: FunctionResult
    ) -> GenerationSnapshot:
        """Create comprehensive snapshot of a single generation run."""
        # Convert postconditions with all rich fields
        stored_postconditions = [
            convert_postcondition_to_dict(pc, session_id)
            for pc in function_result.postconditions
        ]
        
        # Calculate quality metrics
        avg_confidence = 0.0
        avg_robustness = 0.0
        avg_quality = 0.0
        
        if function_result.postconditions:
            avg_confidence = sum(
                pc.confidence_score for pc in function_result.postconditions
            ) / len(function_result.postconditions)
            
            avg_robustness = sum(
                getattr(pc, 'robustness_score', 0.0) 
                for pc in function_result.postconditions
            ) / len(function_result.postconditions)
            
            avg_quality = sum(
                getattr(pc, 'overall_quality_score', 0.0)
                for pc in function_result.postconditions
            ) / len(function_result.postconditions)
        
        # Collect edge case intelligence
        unique_edge_cases = set()
        coverage_gaps = set()
        
        for pc in function_result.postconditions:
            if hasattr(pc, 'edge_cases_covered'):
                unique_edge_cases.update(pc.edge_cases_covered)
            if hasattr(pc, 'coverage_gaps'):
                coverage_gaps.update(pc.coverage_gaps)
        
        # Z3 analysis
        z3_theories = set()
        z3_validations_passed = 0
        z3_validations_failed = 0
        z3_errors = []
        
        for pc in function_result.postconditions:
            if hasattr(pc, 'z3_theory') and pc.z3_theory:
                z3_theories.add(pc.z3_theory)
            
            if hasattr(pc, 'z3_translation') and pc.z3_translation:
                if getattr(pc.z3_translation, 'z3_validation_passed', False):
                    z3_validations_passed += 1
                else:
                    z3_validations_failed += 1
                    error = getattr(pc.z3_translation, 'validation_error', None)
                    if error:
                        z3_errors.append(error)
        
        return GenerationSnapshot(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            specification=specification,
            function_signature=function_result.function_signature,
            function_description=function_result.function_description,
            function_body=getattr(function_result.pseudocode, 'body', None) if hasattr(function_result, 'pseudocode') else None,
            postconditions=stored_postconditions,
            total_postconditions=len(function_result.postconditions),
            average_confidence=avg_confidence,
            average_robustness=avg_robustness,
            average_quality=avg_quality,
            total_edge_cases_covered=len(unique_edge_cases),
            unique_edge_cases=list(unique_edge_cases),
            coverage_gaps_found=list(coverage_gaps),
            z3_translations_generated=len(function_result.postconditions),
            z3_validations_passed=z3_validations_passed,
            z3_validations_failed=z3_validations_failed,
            z3_validation_errors=z3_errors,
            z3_theories_used=list(z3_theories),
            generation_time=getattr(function_result, 'processing_time', 0.0),
            total_time=getattr(function_result, 'processing_time', 0.0),
            status=str(getattr(function_result, 'status', 'success'))
        )
    
    def _serialize_function_result(
        self,
        function_result: FunctionResult,
        session_id: str
    ) -> Dict[str, Any]:
        """Serialize function result to dictionary with all postconditions."""
        postconditions_data = [
            convert_postcondition_to_dict(pc, session_id)
            for pc in function_result.postconditions
        ]
        
        return {
            "function_name": function_result.function_name,
            "function_signature": function_result.function_signature,
            "function_description": function_result.function_description,
            "postconditions": postconditions_data,
            "postcondition_count": len(postconditions_data),
            "average_quality_score": getattr(function_result, 'average_quality_score', 0.0),
            "average_robustness_score": getattr(function_result, 'average_robustness_score', 0.0),
            "z3_translations_count": getattr(function_result, 'z3_translations_count', 0),
            "z3_validations_passed": getattr(function_result, 'z3_validations_passed', 0),
            "z3_validations_failed": getattr(function_result, 'z3_validations_failed', 0),
            "status": str(getattr(function_result, 'status', 'success')),
            "processing_time": getattr(function_result, 'processing_time', 0.0)
        }
    
    def _calculate_session_aggregates(
        self,
        function_results: List[FunctionResult]
    ) -> Dict[str, Any]:
        """Calculate session-level aggregate metrics."""
        total_postconditions = sum(len(fr.postconditions) for fr in function_results)
        total_z3_translations = 0
        total_validations_passed = 0
        all_quality_scores = []
        all_robustness_scores = []
        
        for func_result in function_results:
            for pc in func_result.postconditions:
                if hasattr(pc, 'overall_quality_score'):
                    all_quality_scores.append(pc.overall_quality_score)
                if hasattr(pc, 'robustness_score'):
                    all_robustness_scores.append(pc.robustness_score)
                
                if hasattr(pc, 'z3_translation') and pc.z3_translation:
                    total_z3_translations += 1
                    if getattr(pc.z3_translation, 'z3_validation_passed', False):
                        total_validations_passed += 1
        
        return {
            "total_postconditions": total_postconditions,
            "total_z3_translations": total_z3_translations,
            "z3_validation_success_rate": (
                total_validations_passed / total_z3_translations 
                if total_z3_translations > 0 else 0.0
            ),
            "average_quality_score": (
                sum(all_quality_scores) / len(all_quality_scores)
                if all_quality_scores else 0.0
            ),
            "average_robustness_score": (
                sum(all_robustness_scores) / len(all_robustness_scores)
                if all_robustness_scores else 0.0
            )
        }
    
    def _dict_to_metadata(self, data: Dict[str, Any]) -> ComprehensiveFunctionMetadata:
        """Convert dictionary to metadata object."""
        return ComprehensiveFunctionMetadata(
            function_name=data["function_name"],
            function_signature=data["function_signature"],
            function_description=data["function_description"],
            first_seen=data["first_seen"],
            last_updated=data["last_updated"],
            total_generations=data.get("total_generations", 0),
            generations_history=data.get("generations_history", []),
            total_postconditions_ever_generated=data.get("total_postconditions_ever_generated", 0),
            unique_postconditions_count=data.get("unique_postconditions_count", 0),
            current_postconditions=data.get("current_postconditions", []),
            quality_trend=data.get("quality_trend", []),
            robustness_trend=data.get("robustness_trend", []),
            confidence_trend=data.get("confidence_trend", []),
            total_z3_translations=data.get("total_z3_translations", 0),
            total_z3_validations_passed=data.get("total_z3_validations_passed", 0),
            total_z3_validations_failed=data.get("total_z3_validations_failed", 0),
            z3_theories_encountered=data.get("z3_theories_encountered", []),
            all_edge_cases_covered=data.get("all_edge_cases_covered", []),
            recurring_coverage_gaps=data.get("recurring_coverage_gaps", []),
            edge_case_coverage_improvement=data.get("edge_case_coverage_improvement", 0.0),
            average_generation_time=data.get("average_generation_time", 0.0),
            fastest_generation_time=data.get("fastest_generation_time", 0.0),
            slowest_generation_time=data.get("slowest_generation_time", 0.0),
            best_generation_session_id=data.get("best_generation_session_id"),
            best_generation_quality_score=data.get("best_generation_quality_score", 0.0),
            status=data.get("status", "active"),
            last_error=data.get("last_error"),
            warnings_count=data.get("warnings_count", 0)
        )
    
    # ========================================================================
    # QUERY METHODS
    # ========================================================================
    
    def get_function_metadata(self, function_name: str) -> Optional[ComprehensiveFunctionMetadata]:
        """
        Retrieve comprehensive metadata for a function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            ComprehensiveFunctionMetadata or None if not found
        """
        metadata_file = self.functions_dir / f"{function_name}.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return self._dict_to_metadata(data)
    
    def get_function_summary(self, function_name: str) -> Dict[str, Any]:
        """
        Get a summary of function metadata (for pipeline compatibility).
        
        Args:
            function_name: Name of the function
            
        Returns:
            Dictionary with summary information
        """
        metadata = self.get_function_metadata(function_name)
        if not metadata:
            return {
                "function_name": function_name,
                "exists": False
            }
        
        return {
            "function_name": metadata.function_name,
            "exists": True,
            "metadata": {
                "total_generations": metadata.total_generations,
                "last_updated": metadata.last_updated,
                "best_quality_score": metadata.best_generation_quality_score,
                "total_postconditions": metadata.total_postconditions_ever_generated
            }
        }
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full session snapshot.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Complete session data or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_all_functions(self) -> List[str]:
        """List all functions with stored metadata."""
        return [f.stem for f in self.functions_dir.glob("*.json")]
    
    def list_all_sessions(self) -> List[str]:
        """List all stored session IDs."""
        return [f.stem for f in self.sessions_dir.glob("*.json")]
    
    def get_quality_trends(self, function_name: str) -> Optional[Dict[str, List[float]]]:
        """
        Get quality trends for a function over time.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Dictionary with trend data or None if not found
        """
        metadata = self.get_function_metadata(function_name)
        if not metadata:
            return None
        
        return {
            "quality_trend": metadata.quality_trend,
            "robustness_trend": metadata.robustness_trend,
            "confidence_trend": metadata.confidence_trend
        }


# ============================================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================================

# Alias for pipeline.py compatibility
# This allows pipeline.py to import ResultStorage while using EnhancedDatabaseManager
ResultStorage = EnhancedDatabaseManager

# Export main classes
__all__ = [
    'EnhancedDatabaseManager',
    'ResultStorage',  # Alias for backward compatibility
    'GenerationSnapshot',
    'ComprehensiveFunctionMetadata',
    'convert_postcondition_to_dict'
]