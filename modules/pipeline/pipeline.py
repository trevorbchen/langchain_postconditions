"""
Enhanced Pipeline Orchestrator

🔴 FIXED VERSION - Added Missing Methods:
✅ Added _save_pseudocode()
✅ Added _save_postconditions_detailed()
✅ Added _save_z3_code()
✅ Added _save_validation_report()
✅ Added _save_session_metadata()
✅ Updated _compute_statistics() to calculate quality aggregates
"""

from typing import Optional, List
from pathlib import Path
from datetime import datetime
import asyncio
import uuid
import sys
import os
import json

# Get the project root directory
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
    """Unified pipeline for complete postcondition generation."""
    
    def __init__(
        self,
        codebase_path: Optional[Path] = None,
        validate_z3: bool = True
    ):
        self.factory = ChainFactory()
        self.codebase_path = codebase_path
        self.validate_z3 = validate_z3
    
    async def process(
        self,
        specification: str,
        session_id: Optional[str] = None
    ) -> CompleteEnhancedResult:
        """Process a specification through the complete pipeline."""
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
            
            # Step 2: Generate postconditions
            logger.info("Step 2: Generating postconditions...")
            for function in result.pseudocode_result.functions:
                func_result = await self._generate_postconditions_for_function(
                    specification, function, result
                )
                result.function_results.append(func_result)
            
            # Step 3: Translate to Z3
            if self.validate_z3:
                logger.info("Step 3: Translating to Z3...")
                await self._translate_all_to_z3(result.function_results)
            
            # Step 4: Compute statistics (including quality aggregates)
            logger.info("Step 4: Computing statistics...")
            self._compute_statistics(result)
            
            # Step 5: Generate validation report
            logger.info("Step 5: Generating validation report...")
            self._generate_validation_report(result)
            
            result.status = ProcessingStatus.SUCCESS
            result.completed_at = datetime.now().isoformat()
            
            logger.info(f"Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result.status = ProcessingStatus.FAILED
            result.errors.append(str(e))
        
        return result
    
    def process_sync(self, specification: str, session_id: Optional[str] = None) -> CompleteEnhancedResult:
        """Synchronous wrapper for process()."""
        return asyncio.run(self.process(specification, session_id))
    
    async def _generate_pseudocode(self, specification: str) -> PseudocodeResult:
        """Generate pseudocode from specification."""
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
        """Generate postconditions for a single function."""
        func_result = FunctionResult(
            function_name=function.name,
            function_signature=function.signature,
            function_description=function.description,
            pseudocode=function
        )
        
        try:
            postconditions = await self.factory.postcondition.agenerate(
                function=function,
                specification=specification
            )
            
            func_result.postconditions = postconditions
            func_result.postcondition_count = len(postconditions)
            func_result.status = ProcessingStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Failed to generate postconditions for {function.name}: {e}")
            func_result.status = ProcessingStatus.FAILED
            func_result.error_message = str(e)
            result.errors.append(f"Error generating postconditions for {function.name}: {e}")
        
        return func_result
    
    async def _translate_all_to_z3(self, function_results: List[FunctionResult]) -> None:
        """Translate all postconditions to Z3 code."""
        for func_result in function_results:
            if not func_result.postconditions:
                continue
            
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
            
            translations = await self.factory.z3.atranslate_batch(
                postconditions=func_result.postconditions,
                function_context=function_context
            )
            
            for pc, translation in zip(func_result.postconditions, translations):
                pc.z3_translation = translation
            
            # Track metrics
            func_result.z3_translations_count = len(translations)
            func_result.z3_validations_passed = sum(
                1 for t in translations if t.z3_validation_passed
            )
            func_result.z3_validations_failed = len(translations) - func_result.z3_validations_passed
            
            if translations:
                solvers_created = sum(1 for t in translations if t.solver_created)
                func_result.average_solver_creation_rate = solvers_created / len(translations)
                
                func_result.average_constraints_per_code = sum(
                    t.constraints_added for t in translations
                ) / len(translations)
                
                func_result.average_variables_per_code = sum(
                    t.variables_declared for t in translations
                ) / len(translations)
                
                logger.info(f"   ✅ Z3 validations: {func_result.z3_validations_passed}/{len(translations)} passed")
    
    def _compute_statistics(self, result: CompleteEnhancedResult) -> None:
        """
        Compute overall statistics including quality aggregates.
        
        🔴 FIXED: Now calculates quality scores that were causing AttributeErrors
        """
        result.total_functions = len(result.function_results)
        result.total_postconditions = sum(
            fr.postcondition_count for fr in result.function_results
        )
        result.total_z3_translations = sum(
            fr.z3_translations_count for fr in result.function_results
        )
        
        # ====================================================================
        # 🔴 FIX: Compute function-level quality aggregates
        # ====================================================================
        for fr in result.function_results:
            if fr.postconditions:
                # Calculate average quality score
                fr.average_quality_score = sum(
                    pc.overall_quality_score for pc in fr.postconditions
                ) / len(fr.postconditions)
                
                # Calculate average robustness
                fr.average_robustness_score = sum(
                    pc.robustness_score for pc in fr.postconditions
                ) / len(fr.postconditions)
                
                # Calculate edge case coverage
                fr.edge_case_coverage_score = sum(
                    len(pc.edge_cases_covered) for pc in fr.postconditions
                ) / len(fr.postconditions)
                
                # Count total edge cases
                fr.total_edge_cases_covered = sum(
                    len(pc.edge_cases_covered) for pc in fr.postconditions
                )
        
        # ====================================================================
        # 🔴 FIX: Compute pipeline-level quality aggregates
        # ====================================================================
        if result.total_postconditions > 0:
            # Average quality across all postconditions
            all_quality_scores = [
                pc.overall_quality_score 
                for fr in result.function_results 
                for pc in fr.postconditions
            ]
            result.average_quality_score = sum(all_quality_scores) / len(all_quality_scores)
            
            # Average robustness across all postconditions
            all_robustness_scores = [
                pc.robustness_score
                for fr in result.function_results
                for pc in fr.postconditions
            ]
            result.average_robustness_score = sum(all_robustness_scores) / len(all_robustness_scores)
        
        # Z3 validation statistics
        if result.total_z3_translations > 0:
            successful_validations = sum(
                fr.z3_validations_passed for fr in result.function_results
            )
            result.z3_validation_success_rate = successful_validations / result.total_z3_translations
            
            total_solvers = sum(
                fr.average_solver_creation_rate * fr.z3_translations_count
                for fr in result.function_results
                if fr.z3_translations_count > 0
            )
            result.solver_creation_rate = total_solvers / result.total_z3_translations
    
    def _generate_validation_report(self, result: CompleteEnhancedResult) -> None:
        """Generate validation warnings."""
        if not settings.z3_validation.generate_reports:
            return
        
        if result.z3_validation_success_rate < settings.z3_validation.min_success_rate:
            result.warnings.append(
                f"⚠️  Z3 validation success rate ({result.z3_validation_success_rate:.1%}) "
                f"below threshold ({settings.z3_validation.min_success_rate:.1%})"
            )
        
        if result.solver_creation_rate < settings.z3_validation.min_solver_creation_rate:
            result.warnings.append(
                f"⚠️  Solver creation rate ({result.solver_creation_rate:.1%}) "
                f"below threshold ({settings.z3_validation.min_solver_creation_rate:.1%})"
            )
    
    # ========================================================================
    # 🔴 MISSING SAVE METHODS - ADDED BELOW
    # ========================================================================
    
    def _save_pseudocode(self, pseudocode_result: PseudocodeResult, output_dir: Path) -> None:
        """
        Save pseudocode to files.
        
        🔴 FIXED: This method was missing, causing AttributeError
        """
        pseudocode_dir = output_dir / "pseudocode"
        pseudocode_dir.mkdir(exist_ok=True)
        
        # Save full JSON
        full_path = pseudocode_dir / "pseudocode_full.json"
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(pseudocode_result.model_dump(), f, indent=2)
        
        # Save human-readable summary
        summary_path = pseudocode_dir / "pseudocode_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GENERATED FUNCTIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for func in pseudocode_result.functions:
                f.write(f"Function: {func.name}\n")
                f.write(f"Signature: {func.signature}\n")
                f.write(f"Description: {func.description}\n")
                f.write(f"Return Type: {func.return_type}\n")
                f.write(f"Complexity: {func.complexity}\n")
                f.write(f"Memory: {func.memory_usage}\n")
                
                if func.input_parameters:
                    f.write(f"\nInput Parameters:\n")
                    for param in func.input_parameters:
                        f.write(f"  - {param.name}: {param.data_type}\n")
                        if param.description:
                            f.write(f"    {param.description}\n")
                
                if func.edge_cases:
                    f.write(f"\nEdge Cases:\n")
                    for edge in func.edge_cases:
                        f.write(f"  - {edge}\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
    
    def _save_postconditions_detailed(self, result: CompleteEnhancedResult, output_dir: Path) -> None:
        """
        Save detailed postcondition information for each function.
        
        🔴 FIXED: This method was missing, causing AttributeError
        """
        postcond_dir = output_dir / "postconditions"
        postcond_dir.mkdir(exist_ok=True)
        
        for func_result in result.function_results:
            if not func_result.postconditions:
                continue
            
            # Create filename
            safe_name = "".join(c if c.isalnum() else "_" for c in func_result.function_name)
            filename = f"{safe_name}_postconditions.json"
            filepath = postcond_dir / filename
            
            # Prepare data
            postconditions_data = []
            for i, pc in enumerate(func_result.postconditions, 1):
                pc_dict = pc.model_dump()
                pc_dict['id'] = i
                postconditions_data.append(pc_dict)
            
            # Save JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "function": func_result.function_name,
                    "signature": func_result.function_signature,
                    "description": func_result.function_description,
                    "postcondition_count": len(postconditions_data),
                    "average_quality": func_result.average_quality_score,
                    "average_robustness": func_result.average_robustness_score,
                    "postconditions": postconditions_data
                }, f, indent=2)
            
            # Also save human-readable version
            txt_filename = f"{safe_name}_postconditions.txt"
            txt_filepath = postcond_dir / txt_filename
            
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"POSTCONDITIONS: {func_result.function_name}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Signature: {func_result.function_signature}\n")
                f.write(f"Total: {len(func_result.postconditions)}\n")
                f.write(f"Average Quality: {func_result.average_quality_score:.2f}\n")
                f.write(f"Average Robustness: {func_result.average_robustness_score:.2f}\n\n")
                
                for i, pc in enumerate(func_result.postconditions, 1):
                    f.write(f"Postcondition #{i}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Formal: {pc.formal_text}\n")
                    f.write(f"Natural: {pc.natural_language}\n")
                    
                    if pc.precise_translation:
                        f.write(f"Translation: {pc.precise_translation}\n")
                    
                    if pc.reasoning:
                        f.write(f"Reasoning: {pc.reasoning}\n")
                    
                    f.write(f"Quality: {pc.overall_quality_score:.2f}\n")
                    f.write(f"Robustness: {pc.robustness_score:.2f}\n")
                    f.write(f"Z3 Theory: {pc.z3_theory}\n")
                    
                    if pc.edge_cases_covered:
                        f.write(f"Edge Cases: {', '.join(pc.edge_cases_covered)}\n")
                    
                    f.write("\n")
    
    def _save_z3_code(self, result: CompleteEnhancedResult, output_dir: Path) -> None:
        """
        Save Z3 verification code for each postcondition.
        
        🔴 FIXED: This method was missing, causing AttributeError
        """
        z3_dir = output_dir / "z3_code"
        z3_dir.mkdir(exist_ok=True)
        
        for func_result in result.function_results:
            if not func_result.postconditions:
                continue
            
            for i, pc in enumerate(func_result.postconditions, 1):
                if not pc.z3_translation or not pc.z3_translation.z3_code:
                    continue
                
                # Create filename
                safe_name = "".join(c if c.isalnum() else "_" for c in func_result.function_name)
                filename = f"{safe_name}_pc{i}.py"
                filepath = z3_dir / filename
                
                # Save Z3 code
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Z3 Verification Code\n")
                    f.write(f"# Function: {func_result.function_name}\n")
                    f.write(f"# Postcondition {i}: {pc.natural_language}\n")
                    f.write(f"# Formal: {pc.formal_text}\n")
                    f.write(f"#\n")
                    f.write(f"# Validation Status: {'✓ PASSED' if pc.z3_translation.z3_validation_passed else '✗ FAILED'}\n")
                    if pc.z3_translation.validation_error:
                        f.write(f"# Error: {pc.z3_translation.validation_error}\n")
                    f.write(f"#\n\n")
                    
                    f.write(pc.z3_translation.z3_code)
    
    def _save_validation_report(self, result: CompleteEnhancedResult, output_dir: Path) -> None:
        """
        Save Z3 validation report.
        
        🔴 FIXED: This method was missing, causing AttributeError
        """
        report_path = output_dir / "validation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Z3 VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Session ID: {result.session_id}\n")
            f.write(f"Specification: {result.specification}\n")
            f.write(f"Generated: {result.started_at}\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Functions: {result.total_functions}\n")
            f.write(f"Total Postconditions: {result.total_postconditions}\n")
            f.write(f"Total Z3 Translations: {result.total_z3_translations}\n")
            f.write(f"Validation Success Rate: {result.z3_validation_success_rate:.1%}\n")
            f.write(f"Solver Creation Rate: {result.solver_creation_rate:.1%}\n")
            f.write(f"Average Quality Score: {result.average_quality_score:.2f}\n")
            f.write(f"Average Robustness Score: {result.average_robustness_score:.2f}\n\n")
            
            f.write("PER-FUNCTION DETAILS\n")
            f.write("=" * 80 + "\n\n")
            
            for func_result in result.function_results:
                f.write(f"Function: {func_result.function_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Signature: {func_result.function_signature}\n")
                f.write(f"Postconditions: {func_result.postcondition_count}\n")
                f.write(f"Z3 Translations: {func_result.z3_translations_count}\n")
                f.write(f"Validations Passed: {func_result.z3_validations_passed}\n")
                f.write(f"Validations Failed: {func_result.z3_validations_failed}\n")
                f.write(f"Average Quality: {func_result.average_quality_score:.2f}\n")
                f.write(f"Average Robustness: {func_result.average_robustness_score:.2f}\n")
                f.write(f"Edge Cases Covered: {func_result.total_edge_cases_covered}\n\n")
            
            if result.warnings:
                f.write("WARNINGS\n")
                f.write("-" * 80 + "\n")
                for warning in result.warnings:
                    f.write(f"{warning}\n")
                f.write("\n")
            
            if result.errors:
                f.write("ERRORS\n")
                f.write("-" * 80 + "\n")
                for error in result.errors:
                    f.write(f"{error}\n")
    
    def _save_session_metadata(self, result: CompleteEnhancedResult, output_dir: Path, session_name: str) -> None:
        """
        Save session metadata.
        
        🔴 FIXED: This method was missing, causing AttributeError
        """
        metadata_path = output_dir / "session_metadata.txt"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SESSION METADATA\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Session Name: {session_name}\n")
            f.write(f"Session ID: {result.session_id}\n")
            f.write(f"Specification: {result.specification}\n\n")
            
            f.write("TIMESTAMPS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Started: {result.started_at}\n")
            f.write(f"Completed: {result.completed_at}\n")
            f.write(f"Processing Time: {result.total_processing_time:.2f}s\n\n")
            
            f.write("STATUS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Overall Status: {result.status.value}\n\n")
            
            f.write("STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Functions: {result.total_functions}\n")
            f.write(f"Postconditions: {result.total_postconditions}\n")
            f.write(f"Z3 Translations: {result.total_z3_translations}\n")
            f.write(f"Validation Rate: {result.z3_validation_success_rate:.1%}\n")
            f.write(f"Quality Score: {result.average_quality_score:.2f}\n")
            f.write(f"Robustness Score: {result.average_robustness_score:.2f}\n")
    
    # ========================================================================
    # save_results() METHOD (existing, modified to use the new save methods)
    # ========================================================================
    
    def save_results(self, result: CompleteEnhancedResult, output_dir: Path) -> Path:
        """
        Save results to timestamped folder with all files.
        
        Args:
            result: CompleteEnhancedResult to save
            output_dir: Base output directory
            
        Returns:
            Path to the created session directory
        """
        # Ensure absolute path
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        spec_words = result.specification.lower().split()[:3]
        spec_name = "_".join("".join(c for c in word if c.isalnum()) for word in spec_words)
        folder_name = f"{timestamp}_{spec_name}"
        session_dir = output_dir / folder_name
        
        logger.info(f"📁 Creating new session folder: {folder_name}")
        
        # Create directory
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Session directory created: {session_dir}")
        
        files_created = []
        
        # 1. Save complete JSON result
        try:
            json_path = session_dir / "complete_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(result.model_dump_json(indent=2))
                f.flush()
                os.fsync(f.fileno())
            
            if json_path.exists():
                size = json_path.stat().st_size
                logger.info(f"✅ Saved: complete_result.json ({size:,} bytes)")
                files_created.append(("complete_result.json", size))
        except Exception as e:
            logger.error(f"❌ Failed to save JSON: {e}")
        
        # 2. Save pseudocode (using new method)
        if result.pseudocode_result:
            try:
                self._save_pseudocode(result.pseudocode_result, session_dir)
            except Exception as e:
                logger.error(f"❌ Failed to save pseudocode: {e}")
        
        # 3. Save postconditions (using new method)
        try:
            self._save_postconditions_detailed(result, session_dir)
        except Exception as e:
            logger.error(f"❌ Failed to save postconditions: {e}")
        
        # 4. Save Z3 code (using new method)
        try:
            self._save_z3_code(result, session_dir)
        except Exception as e:
            logger.error(f"❌ Failed to save Z3 code: {e}")
        
        # 5. Save validation report (using new method)
        try:
            self._save_validation_report(result, session_dir)
        except Exception as e:
            logger.error(f"❌ Failed to save validation report: {e}")
        
        # 6. Save session metadata (using new method)
        try:
            self._save_session_metadata(result, session_dir, folder_name)
        except Exception as e:
            logger.error(f"❌ Failed to save session metadata: {e}")
        
        # Count all created files
        all_files = list(session_dir.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        
        logger.info(f"\n📊 Session Summary:")
        logger.info(f"   Folder: {folder_name}")
        logger.info(f"   Total files: {len(all_files)}")
        
        print(f"\n✅ Results saved to new session: {session_dir}")
        print(f"   📁 Session: {folder_name}")
        print(f"   📄 Files: {len(all_files)}")
        
        return session_dir


def process_specification(specification: str, codebase_path: Optional[Path] = None) -> CompleteEnhancedResult:
    """Convenience function to process a specification."""
    pipeline = PostconditionPipeline(codebase_path=codebase_path)
    return pipeline.process_sync(specification)