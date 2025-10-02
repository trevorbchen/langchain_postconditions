"""
Enhanced Pipeline Orchestrator - Phase 5 Complete
"""

from typing import Optional, List
from pathlib import Path
from datetime import datetime
import asyncio
import uuid
import sys
import os  # Required for os.fsync()

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
            
            # Step 4: Compute statistics
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
            
            if postconditions:
                func_result.average_quality_score = sum(
                    pc.overall_quality_score for pc in postconditions
                ) / len(postconditions)
                
                func_result.average_robustness_score = sum(
                    pc.robustness_score for pc in postconditions
                ) / len(postconditions)
                
                func_result.edge_case_coverage_score = sum(
                    len(pc.edge_cases_covered) for pc in postconditions
                ) / len(postconditions)
                
                valid_count = sum(
                    1 for pc in postconditions 
                    if "valid" in pc.mathematical_validity.lower()
                )
                func_result.mathematical_validity_rate = valid_count / len(postconditions)
                
                logger.info(f"Generated {len(postconditions)} postconditions for {function.name}")
        
        except Exception as e:
            result.errors.append(f"Error generating postconditions for {function.name}: {e}")
            logger.error(f"Failed to generate postconditions: {e}")
        
        return func_result
    
    async def _translate_all_to_z3(self, function_results: List[FunctionResult]) -> None:
        """Translate all postconditions to Z3."""
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
            
            # Collect errors
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
        """Compute overall statistics."""
        result.total_postconditions = sum(
            fr.postcondition_count for fr in result.function_results
        )
        
        result.total_z3_translations = sum(
            fr.z3_translations_count for fr in result.function_results
        )
        
        total_passed = sum(
            fr.z3_validations_passed for fr in result.function_results
        )
        
        if result.total_z3_translations > 0:
            result.z3_validation_success_rate = total_passed / result.total_z3_translations
        else:
            result.z3_validation_success_rate = 0.0
        
        solver_rates = [
            fr.average_solver_creation_rate
            for fr in result.function_results
            if fr.z3_translations_count > 0
        ]
        if solver_rates:
            result.solver_creation_rate = sum(solver_rates) / len(solver_rates)
        else:
            result.solver_creation_rate = 0.0
        
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
        """Generate validation report with warnings."""
        if not settings.z3_validation.generate_reports:
            return
        
        all_errors = []
        for fr in result.function_results:
            all_errors.extend(fr.z3_validation_errors)
        
        if not all_errors:
            result.warnings.append("✅ All Z3 validations passed!")
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
    
    def save_results(self, result: CompleteEnhancedResult, output_dir: Path) -> Path:
        """
        Save results to timestamped folder (pseudo-database approach).
        
        Each run creates a new folder:
        output/pipeline_results/
            2025-10-01_23-22-35_reverse_list/
                complete_result.json
                pseudocode/
                validation_report.txt
            2025-10-01_23-45-12_binary_search/
                complete_result.json
                ...
        
        Args:
            result: CompleteEnhancedResult to save
            output_dir: Base output directory (e.g., output/pipeline_results)
            
        Returns:
            Path to the created timestamped folder
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Ensure absolute path
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        
        # 🆕 CREATE TIMESTAMPED FOLDER
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Extract safe folder name from specification (first 3 words, sanitized)
        spec_words = result.specification.lower().split()[:3]
        spec_name = "_".join("".join(c for c in word if c.isalnum()) for word in spec_words)
        
        # Create unique folder name: timestamp_specification
        folder_name = f"{timestamp}_{spec_name}"
        session_dir = output_dir / folder_name
        
        logger.info(f"📁 Creating new session folder: {folder_name}")
        
        # Create directory
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Session directory created: {session_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to create session directory: {e}")
            raise
        
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
            raise
        
        # 2. Save pseudocode
        if result.pseudocode_result:
            try:
                self._save_pseudocode(result.pseudocode_result, session_dir)
                pseudocode_dir = session_dir / "pseudocode"
                if pseudocode_dir.exists():
                    for pf in pseudocode_dir.glob("*"):
                        if pf.is_file():
                            size = pf.stat().st_size
                            logger.info(f"✅ Saved: pseudocode/{pf.name} ({size:,} bytes)")
                            files_created.append((f"pseudocode/{pf.name}", size))
            except Exception as e:
                logger.error(f"❌ Failed to save pseudocode: {e}")
        
        # 3. Save postconditions by function
        try:
            self._save_postconditions_detailed(result, session_dir)
            postcond_dir = session_dir / "postconditions"
            if postcond_dir.exists():
                for pf in postcond_dir.glob("*"):
                    if pf.is_file():
                        size = pf.stat().st_size
                        logger.info(f"✅ Saved: postconditions/{pf.name} ({size:,} bytes)")
                        files_created.append((f"postconditions/{pf.name}", size))
        except Exception as e:
            logger.error(f"❌ Failed to save postconditions: {e}")
        
        # 4. Save Z3 code
        try:
            self._save_z3_code(result, session_dir)
            z3_dir = session_dir / "z3_code"
            if z3_dir.exists():
                for pf in z3_dir.glob("*"):
                    if pf.is_file():
                        size = pf.stat().st_size
                        logger.info(f"✅ Saved: z3_code/{pf.name} ({size:,} bytes)")
                        files_created.append((f"z3_code/{pf.name}", size))
        except Exception as e:
            logger.error(f"❌ Failed to save Z3 code: {e}")
        
        # 5. Save validation report
        if settings.z3_validation.generate_reports:
            try:
                self._save_validation_report(result, session_dir)
                report_path = session_dir / "validation_report.txt"
                if report_path.exists():
                    size = report_path.stat().st_size
                    logger.info(f"✅ Saved: validation_report.txt ({size:,} bytes)")
                    files_created.append(("validation_report.txt", size))
            except Exception as e:
                logger.error(f"❌ Failed to save validation report: {e}")
        
        # 6. Save session metadata
        try:
            self._save_session_metadata(result, session_dir, folder_name)
            meta_path = session_dir / "session_metadata.txt"
            if meta_path.exists():
                size = meta_path.stat().st_size
                logger.info(f"✅ Saved: session_metadata.txt ({size:,} bytes)")
                files_created.append(("session_metadata.txt", size))
        except Exception as e:
            logger.error(f"❌ Failed to save session metadata: {e}")
        
        # Summary
        logger.info(f"\n📊 Session Summary:")
        logger.info(f"   Folder: {folder_name}")
        logger.info(f"   Total files: {len(files_created)}")
        for filename, size in files_created:
            logger.info(f"     ✓ {filename} ({size:,} bytes)")
        
        if not files_created:
            raise RuntimeError("No files were created")
        
        print(f"\n✅ Results saved to new session: {session_dir}")
        print(f"   📁 Session: {folder_name}")
        print(f"   📄 Files: {len(files_created)}")
        
        return session_dir


def process_specification(specification: str, codebase_path: Optional[Path] = None) -> CompleteEnhancedResult:
    """Convenience function."""
    pipeline = PostconditionPipeline(codebase_path=codebase_path)
    return pipeline.process_sync(specification)