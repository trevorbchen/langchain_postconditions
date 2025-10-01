"""
Enhanced Pipeline Orchestrator - Phase 7 Complete

PHASE 3 CHANGES:
- Enhanced _generate_postconditions_for_function to calculate ALL rich metrics
- Updated _compute_statistics to aggregate quality data properly
- Added comprehensive logging for enriched statistics
- All metrics now show real values (not 0.0)

PHASE 7 CHANGES (BATCHING):
- Updated _translate_all_to_z3 to use batch translation
- Now translates all postconditions per function in ONE LLM call
- Reduced Z3 translation calls by ~87%
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
import logging

logger = logging.getLogger(__name__)


class PostconditionPipeline:
    """
    Unified pipeline for complete postcondition generation.
    
    ENHANCED in Phase 3:
    - Calculates ALL enriched metrics from rich fields
    - Tracks quality scores, robustness, edge case coverage
    - Computes mathematical validity rates
    - Enhanced result statistics with real data
    
    ENHANCED in Phase 7 (BATCHING):
    - Uses batch Z3 translation (atranslate_batch)
    - Reduces LLM calls by ~87% for Z3 translation
    
    Orchestrates:
    1. Generate pseudocode from specification
    2. Generate postconditions with ALL rich fields (Phase 1-2)
    3. Translate postconditions to Z3 code WITH BATCHING (Phase 7)
    4. Calculate comprehensive statistics (Phase 3)
    5. Save results with UTF-8 encoding
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
            session_id: Optional session identifier
            
        Returns:
            CompleteEnhancedResult with all generated content and enriched metadata
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        start_time = datetime.now()
        
        # Initialize result
        result = CompleteEnhancedResult(
            session_id=session_id,
            specification=specification,
            codebase_path=str(self.codebase_path) if self.codebase_path else None,
            generated_at=start_time,
            overall_status=ProcessingStatus.IN_PROGRESS
        )
        
        try:
            # Step 1: Generate pseudocode
            print(f"📝 Step 1/3: Generating pseudocode...")
            pseudocode_result = await self._generate_pseudocode(specification, result)
            
            if not pseudocode_result or not pseudocode_result.functions:
                result.overall_status = ProcessingStatus.FAILED
                result.errors.append("Pseudocode generation failed")
                return result
            
            # Step 2: Generate postconditions (parallel) with ALL enriched fields
            print(f"🔍 Step 2/3: Generating postconditions for {len(pseudocode_result.functions)} functions...")
            function_results = await self._generate_all_postconditions(
                specification,
                pseudocode_result.functions,
                result
            )
            
            # Step 3: Translate to Z3 WITH BATCHING (Phase 7)
            print(f"⚡ Step 3/3: Translating to Z3 (with batching)...")
            await self._translate_all_to_z3(function_results)
            
            # Finalize result with comprehensive statistics
            result.function_results = function_results
            self._compute_statistics(result)
            result.total_processing_time = (datetime.now() - start_time).total_seconds()
            
            # Determine overall status
            if result.successful_z3_translations == result.total_postconditions:
                result.overall_status = ProcessingStatus.SUCCESS
            elif result.successful_z3_translations > 0:
                result.overall_status = ProcessingStatus.PARTIAL
            else:
                result.overall_status = ProcessingStatus.FAILED
            
            # Print summary with enriched metrics
            self._print_summary(result)
            
            return result
        
        except Exception as e:
            result.overall_status = ProcessingStatus.FAILED
            result.errors.append(f"Pipeline error: {str(e)}")
            return result
    
    def process_sync(
        self,
        specification: str,
        session_id: Optional[str] = None
    ) -> CompleteEnhancedResult:
        """Synchronous version of process()."""
        return asyncio.run(self.process(specification, session_id))
    
    async def _generate_pseudocode(
        self,
        specification: str,
        result: CompleteEnhancedResult
    ) -> Optional[PseudocodeResult]:
        """Generate pseudocode from specification."""
        try:
            pseudocode_result = await self.factory.pseudocode.agenerate(
                specification=specification
            )
            
            result.pseudocode_success = True
            result.pseudocode_raw_output = pseudocode_result
            result.functions_created = [f.name for f in pseudocode_result.functions]
            
            return pseudocode_result
        
        except Exception as e:
            result.pseudocode_success = False
            result.pseudocode_error = str(e)
            result.errors.append(f"Pseudocode generation failed: {e}")
            return None
    
    async def _generate_all_postconditions(
        self,
        specification: str,
        functions: List[Function],
        result: CompleteEnhancedResult
    ) -> List[FunctionResult]:
        """Generate postconditions for all functions in parallel."""
        tasks = [
            self._generate_postconditions_for_function(specification, func, result)
            for func in functions
        ]
        return await asyncio.gather(*tasks)
    
    async def _generate_postconditions_for_function(
        self,
        specification: str,
        function: Function,
        result: CompleteEnhancedResult
    ) -> FunctionResult:
        """
        Generate postconditions for a single function.
        
        PHASE 3 ENHANCEMENT: Calculates ALL enriched metrics from rich fields.
        """
        func_result = FunctionResult(
            function_name=function.name,
            function_signature=function.signature,
            function_description=function.description,
            pseudocode=function
        )
        
        try:
            # Generate postconditions (now with ALL enriched fields from Phase 1-2!)
            postconditions = await self.factory.postcondition.agenerate(
                function=function,
                specification=specification
            )
            
            func_result.postconditions = postconditions
            func_result.postcondition_count = len(postconditions)
            
            # PHASE 3: Calculate enriched metrics from ALL rich fields
            if postconditions:
                print(f"\n   📊 Function: {function.name}")
                print(f"      Postconditions generated: {len(postconditions)}")
                
                func_result.average_quality_score = sum(
                    pc.overall_priority_score for pc in postconditions
                ) / len(postconditions)
                print(f"      Avg quality score: {func_result.average_quality_score:.2f}")
                
                func_result.average_robustness_score = sum(
                    pc.robustness_score for pc in postconditions
                ) / len(postconditions)
                print(f"      Avg robustness: {func_result.average_robustness_score:.2f}")
                
                total_edge_cases = sum(
                    len(pc.edge_cases_covered) for pc in postconditions
                )
                func_result.edge_case_coverage_score = total_edge_cases / len(postconditions)
                print(f"      Edge cases per postcondition: {func_result.edge_case_coverage_score:.1f}")
                
                valid_count = sum(
                    1 for pc in postconditions 
                    if pc.mathematical_validity and "valid" in pc.mathematical_validity.lower()
                )
                func_result.mathematical_validity_rate = valid_count / len(postconditions)
                print(f"      Mathematical validity: {func_result.mathematical_validity_rate:.0%}")
                
                with_translation = sum(1 for pc in postconditions if pc.precise_translation)
                with_reasoning = sum(1 for pc in postconditions if pc.reasoning)
                
                print(f"      With translations: {with_translation}/{len(postconditions)}")
                print(f"      With reasoning: {with_reasoning}/{len(postconditions)}")
                
                avg_clarity = sum(pc.clarity_score for pc in postconditions) / len(postconditions)
                avg_completeness = sum(pc.completeness_score for pc in postconditions) / len(postconditions)
                
                print(f"      Avg clarity: {avg_clarity:.2f}")
                print(f"      Avg completeness: {avg_completeness:.2f}")
        
        except Exception as e:
            result.errors.append(f"Error generating postconditions for {function.name}: {e}")
            func_result.postcondition_count = 0
            logger.error(f"Failed to generate postconditions: {e}")
        
        return func_result
    
    async def _translate_all_to_z3(
        self,
        function_results: List[FunctionResult]
    ) -> None:
        """
        Translate all postconditions to Z3 using BATCHING (PHASE 7).
        
        NEW: Uses atranslate_batch() to translate multiple postconditions
        per function in a single LLM call instead of one call per postcondition.
        
        Savings: 
        - Before: 8 postconditions = 8 LLM calls
        - After:  8 postconditions = 1 LLM call (87% reduction)
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
            
            # ✨ NEW IN PHASE 7: Batch translate all postconditions for this function
            # (typically 6-10 postconditions → 1 LLM call)
            translations = await self.factory.z3.atranslate_batch(
                postconditions=func_result.postconditions,
                function_context=function_context
            )
            
            # Store translations
            func_result.z3_translations = translations
            
            # Update stats
            func_result.z3_success_count = sum(
                1 for t in translations if t.translation_success
            )
            func_result.z3_validated_count = sum(
                1 for t in translations if t.z3_validation_passed
            )
            
            logger.info(f"   ✅ Z3 translations: {func_result.z3_success_count}/{len(translations)} succeeded")
            logger.info(f"   ✅ Z3 validated: {func_result.z3_validated_count}/{len(translations)} passed validation")
    
    def _compute_statistics(self, result: CompleteEnhancedResult) -> None:
        """
        Compute overall statistics for the result.
        
        PHASE 3 ENHANCEMENT: Comprehensive statistics including quality metrics.
        """
        # Basic counts
        result.total_postconditions = sum(
            fr.postcondition_count for fr in result.function_results
        )
        
        result.total_z3_translations = sum(
            len(fr.z3_translations) for fr in result.function_results
        )
        
        result.successful_z3_translations = sum(
            fr.z3_success_count for fr in result.function_results
        )
        
        result.validated_z3_translations = sum(
            fr.z3_validated_count for fr in result.function_results
        )
        
        # PHASE 3: Calculate aggregate quality metrics
        if result.function_results:
            quality_scores = [
                fr.average_quality_score 
                for fr in result.function_results 
                if fr.average_quality_score > 0
            ]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                result.warnings.append(
                    f"Average quality score across all functions: {avg_quality:.2f}"
                )
            
            robustness_scores = [
                fr.average_robustness_score
                for fr in result.function_results
                if fr.average_robustness_score > 0
            ]
            if robustness_scores:
                avg_robustness = sum(robustness_scores) / len(robustness_scores)
                result.warnings.append(
                    f"Average robustness score: {avg_robustness:.2f}"
                )
            
            if result.total_postconditions > 0:
                total_edge_cases = sum(
                    fr.edge_case_coverage_score * fr.postcondition_count
                    for fr in result.function_results
                )
                avg_edge_cases = total_edge_cases / result.total_postconditions
                result.warnings.append(
                    f"Average edge cases per postcondition: {avg_edge_cases:.1f}"
                )
            
            validity_rates = [
                fr.mathematical_validity_rate
                for fr in result.function_results
                if fr.postcondition_count > 0
            ]
            if validity_rates:
                avg_validity = sum(validity_rates) / len(validity_rates)
                result.warnings.append(
                    f"Mathematical validity rate: {avg_validity:.1%}"
                )
    
    def _build_function_context(self, function: Optional[Function]) -> Optional[dict]:
        """Build function context dictionary for Z3 translation."""
        if not function:
            return None
        
        return {
            "name": function.name,
            "signature": function.signature,
            "parameters": [
                {
                    "name": p.name,
                    "data_type": p.data_type,
                    "description": p.description
                }
                for p in function.input_parameters
            ],
            "return_type": function.return_type,
            "description": function.description
        }
    
    def _print_summary(self, result: CompleteEnhancedResult) -> None:
        """
        Print enhanced summary with quality metrics.
        
        PHASE 3: Shows enriched statistics with real values.
        """
        print(f"\n{'='*70}")
        print(f"✅ Pipeline complete!")
        print(f"{'='*70}")
        print(f"Session ID: {result.session_id}")
        print(f"Status: {result.overall_status.value}")
        print(f"\nGeneration Statistics:")
        print(f"  Functions: {len(result.function_results)}")
        print(f"  Postconditions: {result.total_postconditions}")
        print(f"  Z3 translations: {result.successful_z3_translations}/{result.total_z3_translations}")
        print(f"  Z3 validated: {result.validated_z3_translations}/{result.total_z3_translations}")
        print(f"  Processing time: {result.total_processing_time:.1f}s")
        
        # PHASE 3: Quality metrics summary (now with real values!)
        if result.function_results:
            print(f"\n📊 Quality Metrics:")
            for fr in result.function_results:
                if fr.postcondition_count > 0:
                    print(f"  {fr.function_name}:")
                    print(f"    Quality: {fr.average_quality_score:.2f}")
                    print(f"    Robustness: {fr.average_robustness_score:.2f}")
                    print(f"    Edge cases/pc: {fr.edge_case_coverage_score:.1f}")
                    print(f"    Math validity: {fr.mathematical_validity_rate:.0%}")
        
        print(f"{'='*70}\n")
    
    def save_results(
        self,
        result: CompleteEnhancedResult,
        output_dir: Path
    ) -> Path:
        """
        Save complete results to directory.
        
        Uses UTF-8 encoding for mathematical symbols in rich fields.
        
        Args:
            result: Result to save
            output_dir: Output directory
            
        Returns:
            Path to saved directory
        """
        output_dir = Path(output_dir)
        session_dir = output_dir / result.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main result with UTF-8 encoding (for mathematical symbols)
        result_path = session_dir / "result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(result.model_dump_json(indent=2))
        
        # Save pseudocode files
        self._save_pseudocode_files(result, session_dir)
        
        # Save individual Z3 files with UTF-8 encoding
        z3_dir = session_dir / "z3_code"
        z3_dir.mkdir(exist_ok=True)
        
        for func_result in result.function_results:
            for i, translation in enumerate(func_result.z3_translations):
                if translation.z3_code:
                    z3_path = z3_dir / f"{func_result.function_name}_pc{i+1}.py"
                    with open(z3_path, 'w', encoding='utf-8') as f:
                        f.write(f"# Z3 verification for {func_result.function_name}\n")
                        f.write(f"# Postcondition: {translation.natural_language}\n\n")
                        f.write(translation.z3_code)
        
        # Save enriched postconditions summary
        summary_path = session_dir / "postconditions_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Postconditions Summary\n")
            f.write(f"Session: {result.session_id}\n")
            f.write(f"Generated: {result.generated_at}\n")
            f.write(f"{'='*70}\n\n")
            
            for fr in result.function_results:
                f.write(f"Function: {fr.function_name}\n")
                f.write(f"{'='*70}\n\n")
                
                for i, pc in enumerate(fr.postconditions, 1):
                    f.write(f"Postcondition {i}:\n")
                    f.write(f"  Formal: {pc.formal_text}\n")
                    f.write(f"  Natural: {pc.natural_language}\n")
                    
                    if pc.precise_translation:
                        f.write(f"  Translation: {pc.precise_translation}\n")
                    
                    if pc.reasoning:
                        f.write(f"  Reasoning: {pc.reasoning}\n")
                    
                    if pc.edge_cases_covered:
                        f.write(f"  Edge cases: {len(pc.edge_cases_covered)}\n")
                        for ec in pc.edge_cases_covered[:3]:
                            f.write(f"    - {ec}\n")
                    
                    f.write(f"  Quality: {pc.overall_priority_score:.2f}\n")
                    f.write(f"  Robustness: {pc.robustness_score:.2f}\n")
                    f.write(f"\n")
                
                f.write(f"\n")
        
        print(f"💾 Results saved to: {session_dir}")
        return session_dir
    
    def _save_pseudocode_files(self, result: CompleteEnhancedResult, session_dir: Path) -> None:
        """Save pseudocode in multiple formats."""
        if not result.pseudocode_raw_output or not result.pseudocode_raw_output.functions:
            return
        
        pseudocode_dir = session_dir / "pseudocode"
        pseudocode_dir.mkdir(exist_ok=True)
        
        pseudocode = result.pseudocode_raw_output
        
        # 1. Human-readable summary
        summary_path = pseudocode_dir / "pseudocode_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("PSEUDOCODE SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Session: {result.session_id}\n")
            f.write(f"Specification: {result.specification}\n")
            f.write(f"Functions: {len(pseudocode.functions)}\n")
            f.write("=" * 70 + "\n\n")
            
            for func in pseudocode.functions:
                f.write(f"Function: {func.name}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Signature: {func.signature}\n")
                f.write(f"Description: {func.description}\n\n")
                
                if func.input_parameters:
                    f.write("Input Parameters:\n")
                    for param in func.input_parameters:
                        f.write(f"  • {param.name} ({param.data_type}): {param.description}\n")
                    f.write("\n")
                
                f.write(f"Complexity: {func.complexity}\n")
                f.write(f"Memory Usage: {func.memory_usage}\n\n")
                f.write("=" * 70 + "\n\n")
        
        # 2. Complete JSON
        json_path = pseudocode_dir / "pseudocode_full.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(pseudocode.model_dump_json(indent=2))
        
        print(f"   📝 Pseudocode saved:")
        print(f"      • pseudocode_summary.txt")
        print(f"      • pseudocode_full.json")


# Convenience function
def process_specification(
    specification: str,
    codebase_path: Optional[Path] = None
) -> CompleteEnhancedResult:
    """Process a specification with batching enabled."""
    pipeline = PostconditionPipeline(codebase_path=codebase_path)
    return pipeline.process_sync(specification)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 7 COMPLETE - Pipeline with Batch Z3 Translation")
    print("=" * 70)
    print("\n✅ Changes:")
    print("  - Updated _translate_all_to_z3 to use atranslate_batch()")
    print("  - Reduces Z3 calls by ~87%")
    print("\n📊 Expected Savings:")
    print("  - 8 postconditions: 8 calls → 1 call")
    print("  - 24 postconditions: 24 calls → 3 calls")