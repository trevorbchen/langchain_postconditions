"""
Enhanced Pipeline Orchestrator - Phase 5 Complete

PHASE 5 CHANGES (Validation Tracking):
- Enhanced _translate_all_to_z3 to track detailed validation metrics
- Added validation error collection and reporting
- Track solver creation rates, constraint counts, variable counts
- Calculate average execution times
- Preserve all validation metadata
- Generate validation warnings and recommendations

Previous enhancements:
- Phase 3: Calculate quality metrics (robustness, edge cases, etc.)
- Phase 7: Batch Z3 translation (reduces API calls by 87%)
"""

from typing import Optional, List
from pathlib import Path
from datetime import datetime
import asyncio
import uuid
import sys

# Get the project root directory (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.chains import ChainFactory
from core.models import (
    CompleteEnhancedResult,
    FunctionResult,
    PseudocodeResult,
    Function,
    ProcessingStatus
)
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class PostconditionPipeline:
    """
    Unified pipeline for complete postcondition generation.
    
    PHASE 5 ENHANCEMENTS (Validation Tracking):
    - Tracks comprehensive Z3 validation metrics
    - Collects validation errors with types and line numbers
    - Calculates solver creation and constraint usage rates
    - Measures execution performance
    - Generates validation warnings and recommendations
    
    Previous enhancements:
    - Phase 3: Quality metrics (robustness, edge cases, validity)
    - Phase 7: Batch Z3 translation (87% fewer API calls)
    
    Orchestrates:
    1. Generate pseudocode from specification
    2. Generate postconditions with ALL rich fields
    3. Translate postconditions to Z3 code WITH BATCHING
    4. Track comprehensive validation metrics (NEW Phase 5)
    5. Calculate statistics and generate reports
    6. Save results with UTF-8 encoding
    """
    
    def __init__(
        self,
        codebase_path: Optional[Path] = None,
        validate_z3: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            codebase_path: Optional path to existing codebase for context
            validate_z3: Whether to validate generated Z3 code
        """
        self.factory = ChainFactory()
        self.codebase_path = codebase_path
        self.validate_z3 = validate_z3
    
    async def process(
        self,
        specification: str,
        session_id: Optional[str] = None
    ) -> CompleteEnhancedResult:
        """
        Process a specification through the complete pipeline.
        
        Args:
            specification: Natural language specification
            session_id: Optional session ID for tracking
            
        Returns:
            CompleteEnhancedResult with all generated content
        """
        session_id = session_id or str(uuid.uuid4())
        
        result = CompleteEnhancedResult(
            session_id=session_id,
            specification=specification,
            status=ProcessingStatus.IN_PROGRESS
        )
        
        logger.info(f"Starting pipeline for session {session_id}")
        
        try:
            # Step 1: Generate pseudocode
            logger.info("Step 1: Generating pseudocode...")
            result.pseudocode_result = await self._generate_pseudocode(specification)
            
            if not result.pseudocode_result or not result.pseudocode_result.functions:
                result.status = ProcessingStatus.FAILED
                result.errors.append("No functions generated from pseudocode")
                return result
            
            result.total_functions = len(result.pseudocode_result.functions)
            logger.info(f"Generated {result.total_functions} functions")
            
            # Step 2: Generate postconditions for each function
            logger.info("Step 2: Generating postconditions...")
            for function in result.pseudocode_result.functions:
                func_result = await self._generate_postconditions_for_function(
                    specification, function, result
                )
                result.function_results.append(func_result)
            
            # Step 3: Translate to Z3 with batching (Phase 7)
            if self.validate_z3:
                logger.info("Step 3: Translating to Z3 (with batching)...")
                await self._translate_all_to_z3(result.function_results)
            
            # Step 4: Compute statistics (Phase 3 + Phase 5)
            logger.info("Step 4: Computing statistics...")
            self._compute_statistics(result)
            
            # Step 5: Generate validation report (Phase 5)
            logger.info("Step 5: Generating validation report...")
            self._generate_validation_report(result)
            
            result.status = ProcessingStatus.SUCCESS
            result.completed_at = datetime.now().isoformat()
            
            logger.info(f"Pipeline completed successfully for session {session_id}")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result.status = ProcessingStatus.FAILED
            result.errors.append(str(e))
        
        return result
    
    def process_sync(self, specification: str, session_id: Optional[str] = None) -> CompleteEnhancedResult:
        """Synchronous wrapper for process()."""
        return asyncio.run(self.process(specification, session_id))
    
    async def _generate_pseudocode(self, specification: str) -> PseudocodeResult:
        """Generate C pseudocode from specification."""
        try:
            return await self.factory.pseudocode.agenerate(specification)
        except Exception as e:
            logger.error(f"Pseudocode generation failed: {e}")
            raise
    
    async def _generate_postconditions_for_function(
        self,
        specification: str,
        function: Function,
        result: CompleteEnhancedResult
    ) -> FunctionResult:
        """
        Generate postconditions for a single function with quality metrics.
        
        PHASE 3: Enhanced to calculate quality metrics.
        """
        func_result = FunctionResult(
            function_name=function.name,
            function_signature=function.signature,
            function_description=function.description,
            pseudocode=function
        )
        
        try:
            # Generate postconditions with ALL rich fields
            postconditions = await self.factory.postcondition.agenerate(
                function=function,
                specification=specification
            )
            
            func_result.postconditions = postconditions
            func_result.postcondition_count = len(postconditions)
            
            # PHASE 3: Calculate enriched metrics
            if postconditions:
                # Quality scores
                func_result.average_quality_score = sum(
                    pc.overall_quality_score for pc in postconditions
                ) / len(postconditions)
                
                # Robustness analysis
                func_result.average_robustness_score = sum(
                    pc.robustness_score for pc in postconditions
                ) / len(postconditions)
                
                # Edge case coverage
                func_result.edge_case_coverage_score = sum(
                    len(pc.edge_cases_covered) for pc in postconditions
                ) / len(postconditions)
                
                # Mathematical validity
                valid_count = sum(
                    1 for pc in postconditions 
                    if "valid" in pc.mathematical_validity.lower()
                )
                func_result.mathematical_validity_rate = valid_count / len(postconditions)
                
                logger.info(f"Generated {len(postconditions)} postconditions for {function.name}")
                logger.info(f"  Quality: {func_result.average_quality_score:.2f}")
                logger.info(f"  Robustness: {func_result.average_robustness_score:.2f}")
        
        except Exception as e:
            result.errors.append(f"Error generating postconditions for {function.name}: {e}")
            logger.error(f"Failed to generate postconditions: {e}")
        
        return func_result
    
    async def _translate_all_to_z3(
        self,
        function_results: List[FunctionResult]
    ) -> None:
        """
        Translate all postconditions to Z3 using BATCHING.
        
        PHASE 5 ENHANCEMENTS (Validation Tracking):
        - Track detailed validation metrics per function
        - Collect validation errors with types and line numbers
        - Calculate solver creation rates
        - Track constraint and variable usage
        - Measure execution performance
        - Generate validation warnings
        
        PHASE 7 (Batching):
        - Uses atranslate_batch() for multiple postconditions
        - 8 postconditions = 1 LLM call (87% reduction)
        """
        for func_result in function_results:
            if not func_result.postconditions:
                continue
            
            # Build function context once
            function_context = {
                'name': func_result.function_name,
                'signature': func_result.function_signature,
                'description': func_result.function_description,
                'parameters': [
                    {
                        'name': p.name,
                        'data_type': p.data_type,
                        'description': p.description
                    }
                    for p in func_result.pseudocode.input_parameters
                ] if func_result.pseudocode else []
            }
            
            logger.info(f"\n🔄 Translating Z3 for function: {func_result.function_name}")
            logger.info(f"   Postconditions: {len(func_result.postconditions)}")
            
            # PHASE 7: Batch translate all postconditions for this function
            translations = await self.factory.z3.atranslate_batch(
                postconditions=func_result.postconditions,
                function_context=function_context
            )
            
            # Assign translations to postconditions
            for pc, translation in zip(func_result.postconditions, translations):
                pc.z3_translation = translation
            
            # 🆕 PHASE 5: Track comprehensive validation metrics
            func_result.z3_translations_count = len(translations)
            func_result.z3_validations_passed = sum(
                1 for t in translations if t.z3_validation_passed
            )
            func_result.z3_validations_failed = len(translations) - func_result.z3_validations_passed
            
            # 🆕 PHASE 5: Collect validation errors
            func_result.z3_validation_errors = []
            for pc, translation in zip(func_result.postconditions, translations):
                if not translation.z3_validation_passed:
                    func_result.z3_validation_errors.append({
                        "postcondition": pc.formal_text,
                        "error": translation.validation_error,
                        "error_type": translation.error_type,
                        "error_line": translation.error_line,
                        "status": translation.z3_validation_status
                    })
            
            # 🆕 PHASE 5: Calculate execution metrics
            if translations:
                # Solver creation rate
                solvers_created = sum(1 for t in translations if t.solver_created)
                func_result.average_solver_creation_rate = solvers_created / len(translations)
                
                # Average constraints per code
                func_result.average_constraints_per_code = sum(
                    t.constraints_added for t in translations
                ) / len(translations)
                
                # Average variables per code
                func_result.average_variables_per_code = sum(
                    t.variables_declared for t in translations
                ) / len(translations)
                
                # Average execution time
                avg_exec_time = sum(t.execution_time for t in translations) / len(translations)
                
                logger.info(f"   ✅ Z3 validations: {func_result.z3_validations_passed}/{len(translations)} passed")
                logger.info(f"   🔧 Solver creation rate: {func_result.average_solver_creation_rate:.1%}")
                logger.info(f"   📊 Avg constraints: {func_result.average_constraints_per_code:.1f}")
                logger.info(f"   📊 Avg variables: {func_result.average_variables_per_code:.1f}")
                logger.info(f"   ⏱️  Avg execution time: {avg_exec_time:.3f}s")
                
                # 🆕 PHASE 5: Log validation errors if any
                if func_result.z3_validation_errors:
                    logger.warning(f"   ⚠️  {len(func_result.z3_validation_errors)} validation errors:")
                    for error in func_result.z3_validation_errors[:3]:  # Show first 3
                        logger.warning(f"      • {error['error_type']}: {error['error'][:80]}...")
    
    def _compute_statistics(self, result: CompleteEnhancedResult) -> None:
        """
        Compute overall statistics for the result.
        
        PHASE 5 ENHANCEMENT: Added Z3 validation statistics.
        PHASE 3 ENHANCEMENT: Comprehensive quality metrics.
        """
        # Basic counts
        result.total_postconditions = sum(
            fr.postcondition_count for fr in result.function_results
        )
        
        result.total_z3_translations = sum(
            fr.z3_translations_count for fr in result.function_results
        )
        
        # 🆕 PHASE 5: Z3 validation statistics
        total_passed = sum(
            fr.z3_validations_passed for fr in result.function_results
        )
        total_failed = sum(
            fr.z3_validations_failed for fr in result.function_results
        )
        
        if result.total_z3_translations > 0:
            result.z3_validation_success_rate = total_passed / result.total_z3_translations
        else:
            result.z3_validation_success_rate = 0.0
        
        # 🆕 PHASE 5: Solver creation rate across all functions
        solver_rates = [
            fr.average_solver_creation_rate
            for fr in result.function_results
            if fr.z3_translations_count > 0
        ]
        if solver_rates:
            result.solver_creation_rate = sum(solver_rates) / len(solver_rates)
        else:
            result.solver_creation_rate = 0.0
        
        # PHASE 3: Calculate aggregate quality metrics
        if result.function_results:
            quality_scores = [
                fr.average_quality_score 
                for fr in result.function_results 
                if fr.average_quality_score > 0
            ]
            if quality_scores:
                result.average_quality_score = sum(quality_scores) / len(quality_scores)
            
            robustness_scores = [
                fr.average_robustness_score
                for fr in result.function_results
                if fr.average_robustness_score > 0
            ]
            if robustness_scores:
                result.average_robustness_score = sum(robustness_scores) / len(robustness_scores)
            
            # 🆕 PHASE 5: Average validation score
            validation_scores = []
            for fr in result.function_results:
                for pc in fr.postconditions:
                    if pc.z3_translation:
                        validation_scores.append(pc.z3_translation.validation_score)
            
            if validation_scores:
                result.average_validation_score = sum(validation_scores) / len(validation_scores)
            else:
                result.average_validation_score = 0.0
    
    def _generate_validation_report(self, result: CompleteEnhancedResult) -> None:
        """
        Generate validation report with warnings and recommendations.
        
        NEW in PHASE 5: Comprehensive validation reporting.
        """
        if not settings.z3_validation.generate_reports:
            return
        
        # Collect all validation errors across functions
        all_errors = []
        for fr in result.function_results:
            all_errors.extend(fr.z3_validation_errors)
        
        if not all_errors:
            result.warnings.append("✅ All Z3 validations passed!")
            return
        
        # 🆕 PHASE 5: Generate warnings based on validation results
        
        # High failure rate warning
        if result.z3_validation_success_rate < settings.z3_validation.min_success_rate:
            result.warnings.append(
                f"⚠️  Z3 validation success rate ({result.z3_validation_success_rate:.1%}) "
                f"below threshold ({settings.z3_validation.min_success_rate:.1%})"
            )
        
        # Low solver creation rate warning
        if result.solver_creation_rate < settings.z3_validation.min_solver_creation_rate:
            result.warnings.append(
                f"⚠️  Solver creation rate ({result.solver_creation_rate:.1%}) "
                f"below threshold ({settings.z3_validation.min_solver_creation_rate:.1%})"
            )
        
        # Error type breakdown
        error_types = {}
        for error in all_errors:
            error_type = error.get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if error_types:
            result.warnings.append(f"⚠️  Validation error breakdown:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                result.warnings.append(f"    • {error_type}: {count} occurrences")
        
        # 🆕 PHASE 5: Recommendations
        recommendations = []
        
        if error_types.get('SyntaxError', 0) > 0:
            recommendations.append(
                "🔧 Syntax errors detected. Review Z3 code generation templates for proper Python syntax."
            )
        
        if error_types.get('ImportError', 0) > 0:
            recommendations.append(
                "📦 Import errors found. Ensure 'from z3 import *' is included in all generated code."
            )
        
        if error_types.get('NameError', 0) > 0 or error_types.get('RuntimeError', 0) > 0:
            recommendations.append(
                "⚡ Runtime errors detected. Review variable declarations and Z3 API usage."
            )
        
        if error_types.get('TimeoutError', 0) > 0:
            recommendations.append(
                "⏱️  Timeout errors found. Generated constraints may be too complex or contain infinite loops."
            )
        
        if result.solver_creation_rate < 0.8:
            recommendations.append(
                "🔍 Low solver creation rate. Ensure 'Solver()' is properly instantiated in generated code."
            )
        
        if recommendations:
            result.warnings.append("")
            result.warnings.append("💡 Recommendations:")
            result.warnings.extend(recommendations)
    
    def _build_function_context(self, function: Optional[Function]) -> Optional[dict]:
        """Build function context dictionary for Z3 translation."""
        if not function:
            return None
        
        return {
            'name': function.name,
            'signature': function.signature,
            'parameters': [
                {
                    'name': p.name,
                    'data_type': p.data_type,
                    'description': p.description
                }
                for p in function.input_parameters
            ]
        }
    
    def save_results(
        self,
        result: CompleteEnhancedResult,
        output_dir: Path
    ) -> None:
        """
        Save results to files with proper UTF-8 encoding.
        
        Args:
            result: Complete result to save
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete JSON
        json_path = output_dir / "complete_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(result.model_dump_json(indent=2))
        
        # Save pseudocode if available
        if result.pseudocode_result:
            self._save_pseudocode(result.pseudocode_result, output_dir)
        
        # 🆕 PHASE 5: Save validation report
        if settings.z3_validation.generate_reports:
            self._save_validation_report(result, output_dir)
        
        print(f"\n✅ Results saved to: {output_dir}")
        print(f"   📄 complete_result.json")
        if result.pseudocode_result:
            print(f"   📝 pseudocode_summary.txt")
            print(f"   📝 pseudocode_full.json")
        if settings.z3_validation.generate_reports:
            print(f"   📊 validation_report.txt")
    
    def _save_pseudocode(self, pseudocode: PseudocodeResult, output_dir: Path):
        """Save pseudocode in readable formats."""
        pseudocode_dir = output_dir / "pseudocode"
        pseudocode_dir.mkdir(exist_ok=True)
        
        # Summary text
        summary_path = pseudocode_dir / "pseudocode_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("PSEUDOCODE SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            for func in pseudocode.functions:
                f.write(f"Function: {func.name}\n")
                f.write(f"Signature: {func.signature}\n")
                f.write(f"Description: {func.description}\n")
                f.write("\n")
        
        # Complete JSON
        json_path = pseudocode_dir / "pseudocode_full.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(pseudocode.model_dump_json(indent=2))
    
    def _save_validation_report(self, result: CompleteEnhancedResult, output_dir: Path):
        """
        Save comprehensive validation report.
        
        NEW in PHASE 5: Detailed validation reporting.
        """
        report_path = output_dir / "validation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Z3 VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Session ID: {result.session_id}\n")
            f.write(f"Generated: {result.started_at}\n\n")
            
            f.write("📊 OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Functions:        {result.total_functions}\n")
            f.write(f"Total Postconditions:   {result.total_postconditions}\n")
            f.write(f"Total Z3 Translations:  {result.total_z3_translations}\n\n")
            
            f.write("✅ VALIDATION RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Success Rate:           {result.z3_validation_success_rate:.1%}\n")
            f.write(f"Solver Creation Rate:   {result.solver_creation_rate:.1%}\n")
            f.write(f"Average Quality:        {result.average_quality_score:.2f}\n")
            f.write(f"Average Validation:     {result.average_validation_score:.2f}\n\n")
            
            # Per-function breakdown
            f.write("📋 PER-FUNCTION RESULTS\n")
            f.write("-" * 80 + "\n")
            for fr in result.function_results:
                f.write(f"\nFunction: {fr.function_name}\n")
                f.write(f"  Postconditions:       {fr.postcondition_count}\n")
                f.write(f"  Z3 Translations:      {fr.z3_translations_count}\n")
                f.write(f"  Validations Passed:   {fr.z3_validations_passed}/{fr.z3_translations_count}\n")
                f.write(f"  Solver Creation Rate: {fr.average_solver_creation_rate:.1%}\n")
                f.write(f"  Avg Constraints:      {fr.average_constraints_per_code:.1f}\n")
                f.write(f"  Avg Variables:        {fr.average_variables_per_code:.1f}\n")
                
                if fr.z3_validation_errors:
                    f.write(f"\n  ⚠️ Validation Errors ({len(fr.z3_validation_errors)}):\n")
                    for error in fr.z3_validation_errors[:5]:  # Show first 5
                        f.write(f"    • {error['error_type']}: {error['error'][:100]}\n")
            
            # Warnings and recommendations
            if result.warnings:
                f.write("\n\n⚠️  WARNINGS & RECOMMENDATIONS\n")
                f.write("-" * 80 + "\n")
                for warning in result.warnings:
                    f.write(f"{warning}\n")
            
            f.write("\n" + "=" * 80 + "\n")


# Convenience function
def process_specification(
    specification: str,
    codebase_path: Optional[Path] = None
) -> CompleteEnhancedResult:
    """Process a specification with validation tracking."""
    pipeline = PostconditionPipeline(codebase_path=codebase_path)
    return pipeline.process_sync(specification)


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 5 COMPLETE - Pipeline with Validation Tracking")
    print("=" * 80)
    print("\n✅ New Features:")
    print("  • Track detailed validation metrics per function")
    print("  • Collect validation errors with types and line numbers")
    print("  • Calculate solver creation and constraint usage rates")
    print("  • Measure execution performance")
    print("  • Generate validation warnings and recommendations")
    print("  • Save comprehensive validation reports")
    print("\n📊 Metrics Tracked:")
    print("  • Z3 validation success rate")
    print("  • Solver creation rate")
    print("  • Average constraints per code")
    print("  • Average variables per code")
    print("  • Execution times")
    print("  • Error type breakdown")
    print("\n💡 Reports Generated:")
    print("  • validation_report.txt with detailed breakdown")
    print("  • Per-function validation statistics")
    print("  • Warnings and recommendations")
    print("=" * 80)