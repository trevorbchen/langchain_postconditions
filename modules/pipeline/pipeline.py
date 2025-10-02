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

from core.models import (
    CompleteEnhancedResult,
    FunctionResult,
    EnhancedPostcondition,
    Z3Translation,
    Function,
    ProcessingStatus
)
from core.chains import ChainFactory
from modules.pseudocode.pseudocode_generator import PseudocodeGenerator
from modules.z3.translator import Z3Translator
from storage.database import ResultStorage
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
        # Default output directory if not specified
        default_output_dir = Path("outputs")
        self.storage = storage or ResultStorage(default_output_dir)
        self.streaming = streaming
        self.codebase_path = codebase_path
        self.validate_z3 = validate_z3
        
        # Initialize chain factory (no streaming argument)
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
                # PseudocodeResult object with functions attribute
                return result.functions
            elif isinstance(result, tuple):
                # If it's a tuple, take the first element (usually the data)
                functions_data = result[0] if result else []
            elif isinstance(result, dict):
                # If it's a dict with 'functions' key
                functions_data = result.get('functions', [])
            elif isinstance(result, list):
                # If it's already a list
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
                # Handle both dict and object types
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
                    # Already a Function object
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
            
            # Calculate quality metrics
            if postconditions:
                self._calculate_function_metrics(func_result)
            
            # Generate Z3 translations if requested
            if generate_z3 and postconditions:
                z3_translations = await self._generate_z3_translations(
                    postconditions=postconditions,
                    function=function
                )
                func_result.z3_translations_count = len(z3_translations) if z3_translations else 0
            
            logger.info(
                f"Completed {function.name}: "
                f"{len(postconditions)} postconditions, "
                f"{func_result.z3_translations_count} Z3 translations"
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
        
        Args:
            postconditions: Postconditions to translate
            function: Function context
            
        Returns:
            List of Z3 translations
        """
        try:
            translations = []
            
            # Translate each postcondition
            for postcondition in postconditions:
                translation = self.z3_translator.translate(
                    postcondition=postcondition,
                    function_context={
                        "name": function.name,
                        "signature": function.signature,
                        "input_parameters": function.input_parameters,
                        "return_type": function.return_type
                    }
                )
                translations.append(translation)
            
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
        
        # Add function details
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
        Calculate quality metrics for a function result.
        
        Args:
            func_result: Function result to calculate metrics for
        """
        postconditions = func_result.postconditions
        
        if not postconditions:
            return
        
        # Edge case coverage
        func_result.total_edge_cases_covered = sum(
            len(pc.edge_cases_covered) for pc in postconditions
        )
    
    async def _save_results(self, request_id: str, result: CompleteEnhancedResult):
        """
        Save results to both request-centric and function-centric storage.
        
        Args:
            request_id: Unique request identifier
            result: Pipeline result to save
        """
        try:
            # 1. Save to request-centric storage (backward compatible)
            self.storage.save_results(request_id, result)
            logger.info(f"Saved results to request storage: {request_id}")
            
            # 2. Save to function-centric storage (NEW)
            for func_result in result.function_results:
                self.storage.save_function_results(
                    function_name=func_result.function_name,
                    request_id=request_id,
                    postconditions=func_result.postconditions,
                    z3_translations=None,  # Z3 translations not stored separately in this version
                    function_signature=func_result.function_signature,
                    function_description=func_result.function_description
                )
                logger.info(
                    f"Saved function results: {func_result.function_name} "
                    f"(request: {request_id})"
                )
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            result.errors.append(f"Storage error: {str(e)}")
    
    @staticmethod
    def _generate_request_id() -> str:
        """
        Generate a unique request identifier.
        
        Returns:
            Request ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"req_{timestamp}"
    
    # =========================================================================
    # COMPATIBILITY METHODS (for main.py)
    # =========================================================================
    
    def process_sync(self, specification: str, session_id: Optional[str] = None) -> CompleteEnhancedResult:
        """
        Synchronous wrapper for async run().
        
        Args:
            specification: Natural language specification
            session_id: Optional session identifier
            
        Returns:
            CompleteEnhancedResult containing all generated postconditions and translations
        """
        return asyncio.run(self.run(specification, session_id=session_id))
    
    def save_results(self, result: CompleteEnhancedResult, output_dir: Path):
        """
        Save pipeline results to specified directory.
        
        Args:
            result: Pipeline result to save
            output_dir: Directory to save results to
        """
        try:
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save complete result as JSON
            result_file = output_dir / "complete_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(result.model_dump(), f, indent=2, default=str)
            
            logger.info(f"Saved complete result to {result_file}")
            
            # Also save using storage backend (dual write)
            if hasattr(result, 'session_id'):
                self.storage.save_results(result.session_id, result)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    # =========================================================================
    # QUERY METHODS (NEW - for UI integration)
    # =========================================================================
    
    def get_function_history(
        self,
        function_name: str
    ) -> Dict[str, Any]:
        """
        Get complete history of postconditions for a function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Dictionary with function history and statistics
        """
        try:
            # Get summary statistics
            summary = self.storage.get_function_summary(function_name)
            
            # Load all generations
            postconditions_by_request = self.storage.load_function_postconditions(
                function_name
            )
            z3_by_request = self.storage.load_function_z3_translations(
                function_name
            )
            
            # Build timeline
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
    
    def compare_generations(
        self,
        function_name: str,
        request_id_1: str,
        request_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two generations of postconditions for a function.
        
        Args:
            function_name: Name of the function
            request_id_1: First request ID
            request_id_2: Second request ID
            
        Returns:
            Comparison results
        """
        try:
            # Load both generations
            all_pcs = self.storage.load_function_postconditions(function_name)
            
            gen1_pcs = all_pcs.get(request_id_1, [])
            gen2_pcs = all_pcs.get(request_id_2, [])
            
            return {
                "function_name": function_name,
                "generation_1": {
                    "request_id": request_id_1,
                    "count": len(gen1_pcs),
                    "postconditions": [pc.model_dump() for pc in gen1_pcs]
                },
                "generation_2": {
                    "request_id": request_id_2,
                    "count": len(gen2_pcs),
                    "postconditions": [pc.model_dump() for pc in gen2_pcs]
                },
                "comparison": {
                    "count_diff": len(gen2_pcs) - len(gen1_pcs)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to compare generations: {e}")
            return {"error": str(e)}
    
    def list_all_functions(self) -> List[Dict[str, Any]]:
        """
        List all functions with summary information.
        
        Returns:
            List of function summaries
        """
        try:
            function_names = self.storage.list_all_functions()
            
            summaries = []
            for func_name in function_names:
                summary = self.storage.get_function_summary(func_name)
                summaries.append(summary)
            
            # Sort by last updated (most recent first)
            summaries.sort(
                key=lambda x: x.get("metadata", {}).get("last_updated", ""),
                reverse=True
            )
            
            return summaries
            
        except Exception as e:
            logger.error(f"Failed to list functions: {e}")
            return []


# Convenience function for creating pipeline
def create_pipeline(
    output_dir: str = "outputs",
    streaming: bool = False
) -> PostconditionPipeline:
    """
    Create a PostconditionPipeline instance.
    
    Args:
        output_dir: Directory for storing results
        streaming: Enable streaming responses
        
    Returns:
        Configured pipeline instance
    """
    storage = ResultStorage(Path(output_dir))
    return PostconditionPipeline(storage=storage, streaming=streaming)