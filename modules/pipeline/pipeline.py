"""
Pipeline for orchestrating postcondition generation workflow.

Handles the complete pipeline from specification to postconditions and Z3 translations,
with dual storage (request-centric and function-centric).
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Core models - ALL imports needed
from core.models import (
    CompleteEnhancedResult,
    FunctionResult,
    EnhancedPostcondition,
    Z3Translation,
    Function,
    ProcessingStatus
)

# Chain factory
from core.chains import ChainFactory

# Module components
from modules.pseudocode.pseudocode_generator import PseudocodeGenerator
from modules.z3.translator import Z3Translator

# Storage
from storage.database import ResultStorage

# Config
from config.settings import Settings

logger = logging.getLogger(__name__)


class PostconditionPipeline:
    """
    Main pipeline for generating postconditions and Z3 translations.
    
    Workflow:
    1. Parse specification into functions
    2. Generate postconditions for each function
    3. Translate postconditions to Z3 code
    4. Save results to both request-centric and function-centric storage
    """
    
    def __init__(
        self,
        storage: Optional[ResultStorage] = None,
        settings: Optional[Settings] = None,
        streaming: bool = False,
        codebase_path: Optional[Path] = None,
        validate_z3: bool = True
    ):
        """
        Initialize pipeline with dependencies.
        
        Args:
            storage: Storage backend (creates default if None)
            settings: Configuration settings (loads default if None)
            streaming: Enable streaming responses from LLM
            codebase_path: Optional path to existing codebase (for context)
            validate_z3: Whether to validate Z3 translations
        """
        self.settings = settings or Settings()
        default_output_dir = Path("outputs")
        self.storage = storage or ResultStorage(default_output_dir)
        self.streaming = streaming
        self.codebase_path = codebase_path
        self.validate_z3 = validate_z3
        
        # Initialize chain factory
        self.factory = ChainFactory()
        
        # Initialize components
        self.pseudocode_generator = PseudocodeGenerator()
        self.z3_translator = Z3Translator()
        
        logger.info("Pipeline initialized")
    
    async def run(
        self,
        specification: str,
        functions: Optional[List[Function]] = None,
        edge_cases: Optional[List[str]] = None,
        generate_z3: bool = True,
        session_id: Optional[str] = None
    ) -> CompleteEnhancedResult:
        """
        Run the complete postcondition generation pipeline.
        
        Args:
            specification: Natural language specification
            functions: Pre-parsed functions (will parse from spec if None)
            edge_cases: Optional list of edge cases to consider
            generate_z3: Whether to generate Z3 translations
            session_id: Optional session identifier
            
        Returns:
            CompleteEnhancedResult containing all generated postconditions and translations
        """
        logger.info("Starting pipeline execution")
        start_time = datetime.now()
        
        # Generate unique request ID
        request_id = session_id or self._generate_request_id()
        
        # Initialize result
        result = CompleteEnhancedResult(
            session_id=request_id,
            specification=specification,
            started_at=start_time.isoformat(),
            status=ProcessingStatus.IN_PROGRESS
        )
        
        try:
            # Step 1: Parse or validate functions
            if functions is None:
                logger.info("Parsing specification into functions")
                functions = await self._parse_specification(specification)
                
                if not functions:
                    result.status = ProcessingStatus.FAILED
                    result.errors.append("No functions found in specification")
                    return result
            
            logger.info(f"Processing {len(functions)} functions")
            
            # Step 2: Generate postconditions for each function
            for function in functions:
                func_result = await self._process_function(
                    specification=specification,
                    function=function,
                    edge_cases=edge_cases,
                    generate_z3=generate_z3
                )
                result.function_results.append(func_result)
            
            # Step 3: Save results to storage (dual write)
            await self._save_results(request_id, result)
            
            result.status = ProcessingStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            result.status = ProcessingStatus.FAILED
            result.errors.append(f"Pipeline error: {str(e)}")
        
        # Calculate execution time
        end_time = datetime.now()
        result.total_processing_time = (end_time - start_time).total_seconds()
        result.completed_at = end_time.isoformat()
        
        logger.info(
            f"Pipeline completed in {result.total_processing_time:.2f}s "
            f"({len(result.function_results)} functions processed)"
        )
        
        return result
    
    async def _parse_specification(
        self,
        specification: str
    ) -> List[Function]:
        """
        Parse specification into pseudocode functions.
        
        Args:
            specification: Natural language specification
            
        Returns:
            List of parsed functions
        """
        try:
            # Use chain to extract functions
            result = await self.factory.pseudocode.agenerate(specification)
            
            # Handle different return types from pseudocode chain
            if hasattr(result, 'functions'):
                return result.functions
            elif isinstance(result, tuple):
                functions_data = result[0] if result else []
            elif isinstance(result, dict):
                functions_data = result.get('functions', [])
            elif isinstance(result, list):
                functions_data = result
            else:
                logger.error(f"Unexpected pseudocode result type: {type(result)}")
                return []
            
            # Ensure we have a list
            if not isinstance(functions_data, list):
                functions_data = [functions_data]
            
            # Convert to Function objects
            functions = []
            for func_data in functions_data:
                if isinstance(func_data, dict):
                    function = Function(
                        name=func_data.get("name", "unknown"),
                        signature=func_data.get("signature", ""),
                        description=func_data.get("description", ""),
                        body=func_data.get("body", ""),
                        input_parameters=func_data.get("input_parameters", []),
                        return_type=func_data.get("return_type", "")
                    )
                    functions.append(function)
                elif hasattr(func_data, 'name'):
                    functions.append(func_data)
            
            return functions
            
        except Exception as e:
            logger.error(f"Failed to parse specification: {e}", exc_info=True)
            return []
    
    async def _process_function(
        self,
        specification: str,
        function: Function,
        edge_cases: Optional[List[str]],
        generate_z3: bool
    ) -> FunctionResult:
        """
        Process a single function through the pipeline.
        
        Args:
            specification: Original specification
            function: Function to process
            edge_cases: Optional edge cases
            generate_z3: Whether to generate Z3 code
            
        Returns:
            FunctionResult with postconditions and Z3 translations
        """
        logger.info(f"Processing function: {function.name}")
        
        func_result = FunctionResult(
            function_name=function.name,
            function_signature=function.signature,
            function_description=function.description,
            pseudocode=function,
            status=ProcessingStatus.IN_PROGRESS
        )
        
        try:
            # Generate postconditions
            postconditions = await self._generate_postconditions(
                specification=specification,
                function=function,
                edge_cases=edge_cases
            )
            
            func_result.postconditions = postconditions
            func_result.postcondition_count = len(postconditions)
            
            # Generate Z3 translations if requested
            if generate_z3 and postconditions:
                z3_translations = await self._generate_z3_translations(
                    postconditions=postconditions,
                    function=function
                )
                func_result.z3_translations_count = len(z3_translations) if z3_translations else 0
                
                # Track Z3 validation metrics
                func_result.z3_validations_passed = sum(
                    1 for pc in postconditions
                    if hasattr(pc, 'z3_translation') and 
                       getattr(pc.z3_translation, 'z3_validation_passed', False)
                )
                func_result.z3_validations_failed = (
                    func_result.z3_translations_count - func_result.z3_validations_passed
                )
                
                # Collect Z3 errors
                func_result.z3_validation_errors = [
                    getattr(pc.z3_translation, 'validation_error', '')
                    for pc in postconditions
                    if hasattr(pc, 'z3_translation') and 
                       getattr(pc.z3_translation, 'validation_error', None)
                ]
            
            # Calculate quality metrics (COMPREHENSIVE)
            if postconditions:
                self._calculate_function_metrics(func_result)
            
            logger.info(
                f"Completed {function.name}: "
                f"{len(postconditions)} postconditions, "
                f"{func_result.z3_translations_count} Z3 translations, "
                f"Quality={getattr(func_result, 'average_quality_score', 0.0):.2f}"
            )
            
            func_result.status = ProcessingStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Error processing function {function.name}: {e}")
            func_result.status = ProcessingStatus.FAILED
            func_result.error_message = f"Processing error: {str(e)}"
        
        return func_result
    
    async def _generate_postconditions(
        self,
        specification: str,
        function: Function,
        edge_cases: Optional[List[str]]
    ) -> List[EnhancedPostcondition]:
        """
        Generate postconditions for a function.
        
        Args:
            specification: Original specification
            function: Function to generate postconditions for
            edge_cases: Optional edge cases to consider
            
        Returns:
            List of generated postconditions
        """
        try:
            # Build context for generation
            context = self._build_generation_context(specification, function)
            
            # Generate postconditions using chain
            postconditions = await self.factory.postcondition.agenerate(
                function=function,
                specification=specification,
                context=context,
                edge_cases=edge_cases
            )
            
            return postconditions
            
        except Exception as e:
            logger.error(
                f"Failed to generate postconditions for {function.name}: {e}"
            )
            return []
    
    async def _generate_z3_translations(
        self,
        postconditions: List[EnhancedPostcondition],
        function: Function
    ) -> List[Z3Translation]:
        """
        Generate Z3 translations for postconditions.
        CRITICAL: Attaches Z3Translation to each postcondition.
        
        Args:
            postconditions: Postconditions to translate
            function: Function context
            
        Returns:
            List of Z3 translations
        """
        try:
            translations = []
            z3_validations_passed = 0
            z3_validations_failed = 0
            z3_errors = []
            
            # Translate each postcondition
            for postcondition in postconditions:
                try:
                    translation = self.z3_translator.translate(
                        postcondition=postcondition,
                        function_context={
                            "name": function.name,
                            "signature": function.signature,
                            "input_parameters": function.input_parameters,
                            "return_type": function.return_type
                        }
                    )
                    
                    # CRITICAL: Attach Z3 translation to postcondition
                    postcondition.z3_translation = translation
                    translations.append(translation)
                    
                    # Track validation results
                    if getattr(translation, 'z3_validation_passed', False):
                        z3_validations_passed += 1
                    else:
                        z3_validations_failed += 1
                        error = getattr(translation, 'validation_error', None)
                        if error:
                            z3_errors.append(error)
                    
                except Exception as e:
                    logger.error(f"Z3 translation failed for postcondition: {e}")
                    z3_validations_failed += 1
                    z3_errors.append(str(e))
            
            logger.info(
                f"Z3 translations: {len(translations)} generated, "
                f"{z3_validations_passed} passed validation, "
                f"{z3_validations_failed} failed"
            )
            
            return translations
            
        except Exception as e:
            logger.error(f"Failed to generate Z3 translations: {e}")
            return []
    
    def _build_generation_context(
        self,
        specification: str,
        function: Function
    ) -> str:
        """
        Build context string for postcondition generation.
        
        Args:
            specification: Original specification
            function: Function being processed
            
        Returns:
            Context string
        """
        context_parts = []
        
        if function.description:
            context_parts.append(f"Function Purpose: {function.description}")
        
        if function.input_parameters:
            params_str = ", ".join(
                f"{p.name}: {p.data_type}"
                for p in function.input_parameters
            )
            context_parts.append(f"Parameters: {params_str}")
        
        if function.return_type:
            context_parts.append(f"Return Type: {function.return_type}")
        
        if function.body:
            context_parts.append(f"Function Body:\n{function.body}")
        
        return "\n\n".join(context_parts)
    
    def _calculate_function_metrics(self, func_result: FunctionResult):
        """
        Calculate COMPREHENSIVE quality metrics for a function result.
        This captures ALL rich data from postconditions and aggregates it.
        """
        postconditions = func_result.postconditions
        
        if not postconditions:
            return
        
        # Quality score aggregation
        quality_scores = [
            getattr(pc, 'overall_quality_score', 0.0)
            for pc in postconditions
        ]
        func_result.average_quality_score = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )
        
        # Robustness score aggregation
        robustness_scores = [
            getattr(pc, 'robustness_score', 0.0)
            for pc in postconditions
        ]
        func_result.average_robustness_score = (
            sum(robustness_scores) / len(robustness_scores) if robustness_scores else 0.0
        )
        
        # Confidence score aggregation
        confidence_scores = [pc.confidence_score for pc in postconditions]
        func_result.average_confidence_score = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        )
        
        # Clarity score aggregation
        clarity_scores = [
            getattr(pc, 'clarity_score', 0.0)
            for pc in postconditions
        ]
        func_result.average_clarity_score = (
            sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
        )
        
        # Completeness score aggregation
        completeness_scores = [
            getattr(pc, 'completeness_score', 0.0)
            for pc in postconditions
        ]
        func_result.average_completeness_score = (
            sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        )
        
        # Edge case coverage
        func_result.total_edge_cases_covered = sum(
            len(getattr(pc, 'edge_cases_covered', []))
            for pc in postconditions
        )
        
        # Unique edge cases
        unique_edge_cases = set()
        for pc in postconditions:
            if hasattr(pc, 'edge_cases_covered'):
                unique_edge_cases.update(pc.edge_cases_covered)
        func_result.unique_edge_cases_count = len(unique_edge_cases)
        
        # Coverage gaps
        all_gaps = set()
        for pc in postconditions:
            if hasattr(pc, 'coverage_gaps'):
                all_gaps.update(pc.coverage_gaps)
        func_result.total_coverage_gaps = len(all_gaps)
        
        # Mathematical validity rate
        valid_count = sum(
            1 for pc in postconditions
            if "valid" in getattr(pc, 'mathematical_validity', '').lower()
        )
        func_result.mathematical_validity_rate = (
            valid_count / len(postconditions) if postconditions else 0.0
        )
        
        # Category distribution
        categories: Dict[str, int] = {}
        for pc in postconditions:
            cat = getattr(pc, 'category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        func_result.postconditions_by_category = categories
        
        logger.info(
            f"Metrics for {func_result.function_name}: "
            f"Quality={func_result.average_quality_score:.2f}, "
            f"Robustness={func_result.average_robustness_score:.2f}, "
            f"EdgeCases={func_result.total_edge_cases_covered}"
        )
    
    async def _save_results(self, request_id: str, result: CompleteEnhancedResult):
        """
        Save results to both request-centric and function-centric storage.
        
        Args:
            request_id: Unique request identifier
            result: Pipeline result to save
        """
        try:
            # 1. Save to request-centric storage
            self.storage.save_results(request_id, result)
            logger.info(f"Saved results to request storage: {request_id}")
            
            # 2. Save to function-centric storage with COMPREHENSIVE data
            for func_result in result.function_results:
                # Extract Z3 translations from postconditions
                z3_translations = [
                    pc.z3_translation 
                    for pc in func_result.postconditions 
                    if hasattr(pc, 'z3_translation') and pc.z3_translation
                ]
                
                self.storage.save_function_results(
                    function_name=func_result.function_name,
                    request_id=request_id,
                    postconditions=func_result.postconditions,
                    z3_translations=z3_translations,
                    function_signature=func_result.function_signature,
                    function_description=func_result.function_description,
                    metrics={
                        "average_quality_score": getattr(func_result, 'average_quality_score', 0.0),
                        "average_robustness_score": getattr(func_result, 'average_robustness_score', 0.0),
                        "average_confidence_score": getattr(func_result, 'average_confidence_score', 0.0),
                        "total_edge_cases_covered": func_result.total_edge_cases_covered,
                        "unique_edge_cases_count": getattr(func_result, 'unique_edge_cases_count', 0),
                        "mathematical_validity_rate": getattr(func_result, 'mathematical_validity_rate', 0.0),
                        "z3_validations_passed": getattr(func_result, 'z3_validations_passed', 0),
                        "z3_validations_failed": getattr(func_result, 'z3_validations_failed', 0),
                        "postconditions_by_category": getattr(func_result, 'postconditions_by_category', {})
                    }
                )
                logger.info(
                    f"Saved comprehensive function results: {func_result.function_name}"
                )
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            result.errors.append(f"Storage error: {str(e)}")
    
    @staticmethod
    def _generate_request_id() -> str:
        """Generate a unique request identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"req_{timestamp}"
    
    # Compatibility methods
    def process_sync(self, specification: str, session_id: Optional[str] = None) -> CompleteEnhancedResult:
        """Synchronous wrapper for async run()."""
        return asyncio.run(self.run(specification, session_id=session_id))
    
    def save_results(self, result: CompleteEnhancedResult, output_dir: Path):
        """Save pipeline results to specified directory."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = output_dir / "complete_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(result.model_dump(), f, indent=2, default=str)
            
            logger.info(f"Saved complete result to {result_file}")
            
            if hasattr(result, 'session_id'):
                self.storage.save_results(result.session_id, result)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    # Query methods
    def get_function_history(self, function_name: str) -> Dict[str, Any]:
        """Get complete history of postconditions for a function."""
        try:
            summary = self.storage.get_function_summary(function_name)
            postconditions_by_request = self.storage.load_function_postconditions(function_name)
            z3_by_request = self.storage.load_function_z3_translations(function_name)
            
            timeline = []
            for request_id in summary.get("request_ids", []):
                postconditions = postconditions_by_request.get(request_id, [])
                z3_translations = z3_by_request.get(request_id, [])
                
                timeline.append({
                    "request_id": request_id,
                    "postcondition_count": len(postconditions),
                    "z3_translation_count": len(z3_translations),
                    "postconditions": [
                        {
                            "formal_text": pc.formal_text,
                            "natural_language": pc.natural_language,
                        }
                        for pc in postconditions
                    ]
                })
            
            return {
                "function_name": function_name,
                "summary": summary,
                "timeline": timeline
            }
            
        except Exception as e:
            logger.error(f"Failed to get function history: {e}")
            return {"error": str(e)}
    
    def list_all_functions(self) -> List[Dict[str, Any]]:
        """List all functions with summary information."""
        try:
            function_names = self.storage.list_all_functions()
            
            summaries = []
            for func_name in function_names:
                summary = self.storage.get_function_summary(func_name)
                summaries.append(summary)
            
            summaries.sort(
                key=lambda x: x.get("metadata", {}).get("last_updated", ""),
                reverse=True
            )
            
            return summaries
            
        except Exception as e:
            logger.error(f"Failed to list functions: {e}")
            return []


def create_pipeline(output_dir: str = "outputs", streaming: bool = False) -> PostconditionPipeline:
    """Create a PostconditionPipeline instance."""
    storage = ResultStorage(Path(output_dir))
    return PostconditionPipeline(storage=storage, streaming=streaming)