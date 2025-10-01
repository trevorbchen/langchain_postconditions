"""
Enhanced Pipeline Orchestrator - Phase 3 Complete

PHASE 3 CHANGES:
- Enhanced _generate_postconditions_for_function to calculate ALL rich metrics
- Updated _compute_statistics to aggregate quality data properly
- Added comprehensive logging for enriched statistics
- All metrics now show real values (not 0.0)
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
    
    Orchestrates:
    1. Generate pseudocode from specification
    2. Generate postconditions with ALL rich fields (Phase 1-2)
    3. Translate postconditions to Z3 code (with validation)
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
            
            # Step 3: Translate to Z3 (parallel) with validation
            print(f"⚡ Step 3/3: Translating to Z3...")
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
            
            # 🆕 PHASE 3: Calculate enriched metrics from ALL rich fields
            if postconditions:
                print(f"\n   📊 Function: {function.name}")
                print(f"      Postconditions generated: {len(postconditions)}")
                
                # ═══════════════════════════════════════════════════════════
                # QUALITY SCORES (from Phase 1-2 fields)
                # ═══════════════════════════════════════════════════════════
                func_result.average_quality_score = sum(
                    pc.overall_priority_score for pc in postconditions
                ) / len(postconditions)
                print(f"      Avg quality score: {func_result.average_quality_score:.2f}")
                
                # ═══════════════════════════════════════════════════════════
                # ROBUSTNESS ANALYSIS (Phase 1 field)
                # ═══════════════════════════════════════════════════════════
                func_result.average_robustness_score = sum(
                    pc.robustness_score for pc in postconditions
                ) / len(postconditions)
                print(f"      Avg robustness: {func_result.average_robustness_score:.2f}")
                
                # ═══════════════════════════════════════════════════════════
                # EDGE CASE COVERAGE (Phase 1 field)
                # ═══════════════════════════════════════════════════════════
                total_edge_cases = sum(
                    len(pc.edge_cases_covered) for pc in postconditions
                )
                func_result.edge_case_coverage_score = total_edge_cases / len(postconditions)
                print(f"      Edge cases per postcondition: {func_result.edge_case_coverage_score:.1f}")
                
                # ═══════════════════════════════════════════════════════════
                # MATHEMATICAL VALIDITY RATE (Phase 1 field)
                # ═══════════════════════════════════════════════════════════
                valid_count = sum(
                    1 for pc in postconditions 
                    if pc.mathematical_validity and "valid" in pc.mathematical_validity.lower()
                )
                func_result.mathematical_validity_rate = valid_count / len(postconditions)
                print(f"      Mathematical validity: {func_result.mathematical_validity_rate:.0%}")
                
                # ═══════════════════════════════════════════════════════════
                # RICH FIELD COMPLETENESS (Phase 1 fields)
                # ═══════════════════════════════════════════════════════════
                with_translation = sum(1 for pc in postconditions if pc.precise_translation)
                with_reasoning = sum(1 for pc in postconditions if pc.reasoning)
                
                print(f"      With translations: {with_translation}/{len(postconditions)}")
                print(f"      With reasoning: {with_reasoning}/{len(postconditions)}")
                
                # ═══════════════════════════════════════════════════════════
                # ADDITIONAL QUALITY METRICS
                # ═══════════════════════════════════════════════════════════
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
        Translate all postconditions to Z3 in parallel.
        
        Preserves all validation metadata from enhanced Z3 translator.
        """
        translation_tasks = []
        
        for func_result in function_results:
            for postcondition in func_result.postconditions:
                task = self.factory.z3.atranslate(
                    postcondition=postcondition,
                    function_context=self._build_function_context(func_result.pseudocode)
                )
                translation_tasks.append((func_result, task))
        
        # Execute all translations in parallel
        if translation_tasks:
            results = await asyncio.gather(*[task for _, task in translation_tasks])
            
            # Assign translations back to function results
            for (func_result, _), translation in zip(translation_tasks, results):
                func_result.z3_translations.append(translation)
                
                if translation.translation_success:
                    func_result.z3_success_count += 1
                
                if translation.z3_validation_passed:
                    func_result.z3_validated_count += 1
    
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
        
        # 🆕 PHASE 3: Calculate aggregate quality metrics
        if result.function_results:
            # Average quality across all functions
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
            
            # Average robustness across all functions
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
            
            # Total edge cases covered
            if result.total_postconditions > 0:
                total_edge_cases = sum(
                    fr.edge_case_coverage_score * fr.postcondition_count
                    for fr in result.function_results
                )
                avg_edge_cases = total_edge_cases / result.total_postconditions
                result.warnings.append(
                    f"Average edge cases per postcondition: {avg_edge_cases:.1f}"
                )
            
            # Mathematical validity rate
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
        
        # 🆕 PHASE 3: Quality metrics summary (now with real values!)
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
        
        # 🆕 SAVE PSEUDOCODE FILES
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
        """
        Save pseudocode in multiple formats.
        
        Creates:
        - pseudocode/ directory
        - pseudocode_summary.txt (human-readable)
        - pseudocode_full.json (complete JSON)
        - Individual .c files for each function (C-style pseudocode)
        """
        if not result.pseudocode_raw_output or not result.pseudocode_raw_output.functions:
            return
        
        pseudocode_dir = session_dir / "pseudocode"
        pseudocode_dir.mkdir(exist_ok=True)
        
        pseudocode = result.pseudocode_raw_output
        
        # ═══════════════════════════════════════════════════════════════
        # 1. HUMAN-READABLE SUMMARY
        # ═══════════════════════════════════════════════════════════════
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
                
                if func.output_parameters:
                    f.write("Output Parameters:\n")
                    for param in func.output_parameters:
                        f.write(f"  • {param.name} ({param.data_type}): {param.description}\n")
                    f.write("\n")
                
                if func.return_values:
                    f.write("Return Values:\n")
                    for ret in func.return_values:
                        f.write(f"  • {ret.condition} → {ret.value}: {ret.description}\n")
                    f.write("\n")
                
                if func.preconditions:
                    f.write("Preconditions:\n")
                    for pre in func.preconditions:
                        f.write(f"  • {pre}\n")
                    f.write("\n")
                
                if func.edge_cases:
                    f.write("Edge Cases:\n")
                    for edge in func.edge_cases:
                        f.write(f"  • {edge}\n")
                    f.write("\n")
                
                f.write(f"Complexity: {func.complexity}\n")
                f.write(f"Memory Usage: {func.memory_usage}\n\n")
                
                if func.body:
                    f.write("Algorithm:\n")
                    f.write(f"{func.body}\n\n")
                
                f.write("=" * 70 + "\n\n")
        
        # ═══════════════════════════════════════════════════════════════
        # 2. COMPLETE JSON
        # ═══════════════════════════════════════════════════════════════
        json_path = pseudocode_dir / "pseudocode_full.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(pseudocode.model_dump_json(indent=2))
        
        # ═══════════════════════════════════════════════════════════════
        # 3. INDIVIDUAL C-STYLE FILES
        # ═══════════════════════════════════════════════════════════════
        for func in pseudocode.functions:
            c_path = pseudocode_dir / f"{func.name}.c"
            with open(c_path, 'w', encoding='utf-8') as f:
                # Header comment
                f.write("/*\n")
                f.write(f" * Function: {func.name}\n")
                f.write(f" * Description: {func.description}\n")
                f.write(f" * Complexity: {func.complexity}\n")
                f.write(f" * Memory: {func.memory_usage}\n")
                f.write(" */\n\n")
                
                # Includes
                if pseudocode.includes:
                    for include in pseudocode.includes:
                        f.write(f"#include <{include}>\n")
                    f.write("\n")
                
                # Structs (if any)
                if pseudocode.structs:
                    for struct in pseudocode.structs:
                        f.write(f"// {struct.description}\n" if struct.description else "")
                        f.write(f"struct {struct.name} {{\n")
                        for field in struct.fields:
                            f.write(f"    {field.get('data_type', 'int')} {field.get('name', 'field')};\n")
                        f.write("};\n\n")
                
                # Function signature
                f.write(f"{func.signature} {{\n")
                
                # Preconditions as comments
                if func.preconditions:
                    f.write("    // Preconditions:\n")
                    for pre in func.preconditions:
                        f.write(f"    // - {pre}\n")
                    f.write("\n")
                
                # Edge cases as comments
                if func.edge_cases:
                    f.write("    // Edge Cases:\n")
                    for edge in func.edge_cases:
                        f.write(f"    // - {edge}\n")
                    f.write("\n")
                
                # Body
                if func.body:
                    for line in func.body.split('\n'):
                        if line.strip():
                            f.write(f"    {line}\n")
                else:
                    f.write("    // TODO: Implement function\n")
                
                f.write("}\n")
        
        # ═══════════════════════════════════════════════════════════════
        # 4. MAKEFILE (if multiple functions)
        # ═══════════════════════════════════════════════════════════════
        if len(pseudocode.functions) > 1:
            makefile_path = pseudocode_dir / "Makefile"
            with open(makefile_path, 'w', encoding='utf-8') as f:
                f.write("# Makefile for pseudocode functions\n\n")
                f.write("CC = gcc\n")
                f.write("CFLAGS = -Wall -Wextra -std=c11\n\n")
                
                func_names = [func.name for func in pseudocode.functions]
                f.write(f"TARGETS = {' '.join(func_names)}\n\n")
                
                f.write("all: $(TARGETS)\n\n")
                
                for func_name in func_names:
                    f.write(f"{func_name}: {func_name}.c\n")
                    f.write(f"\t$(CC) $(CFLAGS) -o {func_name} {func_name}.c\n\n")
                
                f.write("clean:\n")
                f.write("\trm -f $(TARGETS)\n")
        
        print(f"   📝 Pseudocode saved:")
        print(f"      • pseudocode_summary.txt (human-readable)")
        print(f"      • pseudocode_full.json (complete JSON)")
        print(f"      • {len(pseudocode.functions)} .c files (C-style)")
        if len(pseudocode.functions) > 1:
            print(f"      • Makefile (build script)")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def process_specification(
    specification: str,
    codebase_path: Optional[Path] = None
) -> CompleteEnhancedResult:
    """
    Convenience function to process a specification.
    
    Args:
        specification: What to implement
        codebase_path: Optional path to existing codebase
        
    Returns:
        CompleteEnhancedResult with all enriched data
        
    Example:
        >>> result = process_specification("Sort an array using bubble sort")
        >>> print(f"Generated {result.total_postconditions} postconditions")
        >>> print(f"Average quality: {result.function_results[0].average_quality_score:.2f}")
    """
    pipeline = PostconditionPipeline(codebase_path=codebase_path)
    return pipeline.process_sync(specification)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3 COMPLETE - Enhanced Pipeline with Rich Statistics")
    print("=" * 70)
    
    print("\n✅ Changes Made:")
    print("  1. Enhanced _generate_postconditions_for_function")
    print("     - Calculates quality scores from rich fields")
    print("     - Tracks robustness, edge case coverage")
    print("     - Computes mathematical validity rate")
    print("  2. Updated _compute_statistics")
    print("     - Aggregates quality metrics across functions")
    print("     - Real values (not 0.0 anymore!)")
    print("  3. Enhanced logging and summary")
    print("     - Shows all enriched statistics")
    print("  4. Improved save_results")
    print("     - Includes rich field summary")
    
    print("\n⏭️  Next: Phase 4 - Test the complete system")
    print("\n🧪 Test: python main.py --spec 'Reverse an array'")