"""
Storage module for postcondition generation results.

Provides dual storage:
1. Request-centric: outputs/requests/{request_id}/
2. Function-centric: outputs/functions/{function_name}/
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from core.models import (
    CompleteEnhancedResult,
    FunctionResult,
    EnhancedPostcondition,
    Z3Translation,
    Function
)

logger = logging.getLogger(__name__)


class ResultStorage:
    """Handles storage and retrieval of postcondition generation results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize storage with output directory.
        
        Args:
            output_dir: Base directory for storing results
        """
        self.output_dir = Path(output_dir)
        self.requests_dir = self.output_dir / "requests"
        self.functions_dir = self.output_dir / "functions"  # 🆕 NEW
        
        # Create directories
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.functions_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Storage initialized: {self.output_dir}")
    
    # =========================================================================
    # REQUEST-CENTRIC STORAGE (Existing - Backward Compatible)
    # =========================================================================
    
    def save_results(self, request_id: str, result: CompleteEnhancedResult) -> Path:
        """
        Save results organized by request ID (backward compatible).
        
        Args:
            request_id: Unique request identifier
            result: Pipeline result containing all function results
            
        Returns:
            Path to the saved request directory
        """
        request_dir = self.requests_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "request_id": request_id,
            "session_id": result.session_id,
            "timestamp": result.started_at,
            "specification": result.specification,
            "function_count": len(result.function_results),
            "total_postconditions": result.total_postconditions,
            "status": result.status.value,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        
        metadata_file = request_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Save each function's results
        for func_result in result.function_results:
            self._save_function_result_to_request(request_dir, func_result)
        
        logger.info(f"Saved results to {request_dir}")
        return request_dir
    
    def _save_function_result_to_request(
        self,
        request_dir: Path,
        func_result: FunctionResult
    ):
        """Save individual function result to request directory."""
        func_dir = request_dir / func_result.function_name
        func_dir.mkdir(parents=True, exist_ok=True)
        
        # Save postconditions
        if func_result.postconditions:
            postconditions_file = func_dir / "postconditions.json"
            postconditions_data = [
                pc.model_dump() for pc in func_result.postconditions
            ]
            with open(postconditions_file, 'w', encoding='utf-8') as f:
                json.dump(postconditions_data, f, indent=2)
        
        # Save function metadata
        func_metadata = {
            "function_name": func_result.function_name,
            "function_signature": func_result.function_signature,
            "function_description": func_result.function_description,
            "postcondition_count": func_result.postcondition_count,
            "z3_translations_count": func_result.z3_translations_count,
            "status": func_result.status.value,
        }
        
        metadata_file = func_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(func_metadata, f, indent=2)
    
    def load_results(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Load results for a specific request ID.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Dictionary containing request results or None if not found
        """
        request_dir = self.requests_dir / request_id
        
        if not request_dir.exists():
            logger.warning(f"Request directory not found: {request_dir}")
            return None
        
        # Load metadata
        metadata_file = request_dir / "metadata.json"
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load function results
        function_results = {}
        for func_dir in request_dir.iterdir():
            if func_dir.is_dir():
                func_data = self._load_function_result_from_request(func_dir)
                if func_data:
                    function_results[func_dir.name] = func_data
        
        metadata['function_results'] = function_results
        return metadata
    
    def _load_function_result_from_request(
        self,
        func_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Load individual function result from request directory."""
        result = {}
        
        # Load metadata
        metadata_file = func_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                result['metadata'] = json.load(f)
        
        # Load postconditions
        postconditions_file = func_dir / "postconditions.json"
        if postconditions_file.exists():
            with open(postconditions_file, 'r', encoding='utf-8') as f:
                pc_data = json.load(f)
                result['postconditions'] = [
                    EnhancedPostcondition(**pc) for pc in pc_data
                ]
        
        return result if result else None
    
    def list_requests(self) -> List[str]:
        """
        List all stored request IDs.
        
        Returns:
            List of request ID strings
        """
        if not self.requests_dir.exists():
            return []
        
        return [
            d.name for d in self.requests_dir.iterdir()
            if d.is_dir()
        ]
    
    # =========================================================================
    # FUNCTION-CENTRIC STORAGE (🆕 NEW)
    # =========================================================================
    
    def save_function_results(
        self,
        function_name: str,
        request_id: str,
        postconditions: List[EnhancedPostcondition],
        z3_translations: Optional[List[Z3Translation]] = None,
        function_signature: str = "",
        function_description: str = ""
    ) -> Path:
        """
        Save results organized by function name.
        
        Args:
            function_name: Name of the function
            request_id: Request that generated these results
            postconditions: List of generated postconditions
            z3_translations: Optional list of Z3 translations
            function_signature: Function signature for metadata
            function_description: Function description for metadata
            
        Returns:
            Path to the function directory
        """
        # Sanitize function name for filesystem
        safe_function_name = self._sanitize_filename(function_name)
        func_dir = self.functions_dir / safe_function_name
        func_dir.mkdir(parents=True, exist_ok=True)
        
        # Update function metadata
        self._update_function_metadata(
            func_dir,
            function_name,
            function_signature,
            function_description
        )
        
        # Save postconditions
        if postconditions:
            pc_dir = func_dir / "postconditions"
            pc_dir.mkdir(exist_ok=True)
            
            pc_file = pc_dir / f"{request_id}.json"
            pc_data = [pc.model_dump() for pc in postconditions]
            
            with open(pc_file, 'w', encoding='utf-8') as f:
                json.dump(pc_data, f, indent=2)
            
            logger.info(
                f"Saved {len(postconditions)} postconditions for "
                f"{function_name} (request: {request_id})"
            )
        
        # Save Z3 translations
        if z3_translations:
            z3_dir = func_dir / "z3_translations"
            z3_dir.mkdir(exist_ok=True)
            
            z3_file = z3_dir / f"{request_id}.json"
            z3_data = [z3.model_dump() for z3 in z3_translations]
            
            with open(z3_file, 'w', encoding='utf-8') as f:
                json.dump(z3_data, f, indent=2)
            
            logger.info(
                f"Saved {len(z3_translations)} Z3 translations for "
                f"{function_name} (request: {request_id})"
            )
        
        return func_dir
    
    def _update_function_metadata(
        self,
        func_dir: Path,
        function_name: str,
        function_signature: str,
        function_description: str
    ):
        """Update or create function metadata file with comprehensive stats."""
        metadata_file = func_dir / "metadata.json"
        
        # Load existing metadata or create new
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "function_name": function_name,
                "function_signature": function_signature,
                "function_description": function_description,
                "first_seen": datetime.now().isoformat(),
                "total_generations": 0,
                "total_postconditions": 0,
                "total_z3_translations": 0,
                "generations_history": []
            }
        
        # Update basic metadata
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["total_generations"] = metadata.get("total_generations", 0) + 1
        
        # Update signature/description if provided
        if function_signature:
            metadata["function_signature"] = function_signature
        if function_description:
            metadata["function_description"] = function_description
        
        # Calculate statistics from stored files
        pc_dir = func_dir / "postconditions"
        z3_dir = func_dir / "z3_translations"
        
        total_pcs = 0
        total_z3 = 0
        history = []
        
        if pc_dir.exists():
            for pc_file in pc_dir.glob("*.json"):
                request_id = pc_file.stem
                try:
                    with open(pc_file, 'r', encoding='utf-8') as f:
                        pcs = json.load(f)
                        pc_count = len(pcs)
                        total_pcs += pc_count
                        
                        # Get Z3 count for this request
                        z3_file = z3_dir / f"{request_id}.json" if z3_dir.exists() else None
                        z3_count = 0
                        if z3_file and z3_file.exists():
                            with open(z3_file, 'r', encoding='utf-8') as zf:
                                z3s = json.load(zf)
                                z3_count = len(z3s)
                                total_z3 += z3_count
                        
                        # Add to history
                        history.append({
                            "request_id": request_id,
                            "postcondition_count": pc_count,
                            "z3_translation_count": z3_count,
                            "generated_at": pc_file.stat().st_mtime
                        })
                except Exception as e:
                    logger.error(f"Error reading {pc_file}: {e}")
        
        # Update statistics
        metadata["total_postconditions"] = total_pcs
        metadata["total_z3_translations"] = total_z3
        metadata["generations_history"] = sorted(history, key=lambda x: x["generated_at"], reverse=True)
        
        # Save updated metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def load_function_postconditions(
        self,
        function_name: str,
        request_id: Optional[str] = None
    ) -> Dict[str, List[EnhancedPostcondition]]:
        """
        Load postconditions for a function.
        
        Args:
            function_name: Name of the function
            request_id: Optional specific request ID to load
            
        Returns:
            Dict mapping request_id -> List[EnhancedPostcondition]
            If request_id specified, returns single-item dict
        """
        safe_function_name = self._sanitize_filename(function_name)
        pc_dir = self.functions_dir / safe_function_name / "postconditions"
        
        if not pc_dir.exists():
            logger.warning(f"No postconditions found for function: {function_name}")
            return {}
        
        results = {}
        
        # Load specific request or all requests
        if request_id:
            json_files = [pc_dir / f"{request_id}.json"]
        else:
            json_files = list(pc_dir.glob("*.json"))
        
        for json_file in json_files:
            if not json_file.exists():
                continue
            
            req_id = json_file.stem
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                postconditions = [
                    EnhancedPostcondition(**pc) for pc in data
                ]
                results[req_id] = postconditions
                
            except Exception as e:
                logger.error(
                    f"Failed to load postconditions from {json_file}: {e}"
                )
        
        return results
    
    def load_function_z3_translations(
        self,
        function_name: str,
        request_id: Optional[str] = None
    ) -> Dict[str, List[Z3Translation]]:
        """
        Load Z3 translations for a function.
        
        Args:
            function_name: Name of the function
            request_id: Optional specific request ID to load
            
        Returns:
            Dict mapping request_id -> List[Z3Translation]
            If request_id specified, returns single-item dict
        """
        safe_function_name = self._sanitize_filename(function_name)
        z3_dir = self.functions_dir / safe_function_name / "z3_translations"
        
        if not z3_dir.exists():
            logger.warning(
                f"No Z3 translations found for function: {function_name}"
            )
            return {}
        
        results = {}
        
        # Load specific request or all requests
        if request_id:
            json_files = [z3_dir / f"{request_id}.json"]
        else:
            json_files = list(z3_dir.glob("*.json"))
        
        for json_file in json_files:
            if not json_file.exists():
                continue
            
            req_id = json_file.stem
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                translations = [
                    Z3Translation(**z3) for z3 in data
                ]
                results[req_id] = translations
                
            except Exception as e:
                logger.error(
                    f"Failed to load Z3 translations from {json_file}: {e}"
                )
        
        return results
    
    def load_function_metadata(self, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a specific function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Dictionary containing function metadata or None if not found
        """
        safe_function_name = self._sanitize_filename(function_name)
        metadata_file = self.functions_dir / safe_function_name / "metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"No metadata found for function: {function_name}")
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata for {function_name}: {e}")
            return None
    
    def list_all_functions(self) -> List[str]:
        """
        Get list of all function names with stored results.
        
        Returns:
            List of function names
        """
        if not self.functions_dir.exists():
            return []
        
        functions = []
        for func_dir in self.functions_dir.iterdir():
            if func_dir.is_dir() and not func_dir.name.startswith('.'):
                # Try to get original function name from metadata
                metadata = self.load_function_metadata(func_dir.name)
                if metadata and 'function_name' in metadata:
                    functions.append(metadata['function_name'])
                else:
                    functions.append(func_dir.name)
        
        return sorted(functions)
    
    def get_function_summary(self, function_name: str) -> Dict[str, Any]:
        """
        Get summary statistics for a function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Dictionary with summary statistics
        """
        metadata = self.load_function_metadata(function_name)
        postconditions_by_request = self.load_function_postconditions(function_name)
        z3_by_request = self.load_function_z3_translations(function_name)
        
        # Calculate statistics
        total_postconditions = sum(
            len(pcs) for pcs in postconditions_by_request.values()
        )
        
        return {
            "function_name": function_name,
            "metadata": metadata,
            "total_generations": len(postconditions_by_request),
            "total_postconditions": total_postconditions,
            "total_z3_translations": sum(
                len(z3s) for z3s in z3_by_request.values()
            ),
            "request_ids": list(postconditions_by_request.keys()),
        }
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Sanitize function name for use as directory name.
        
        Args:
            name: Original function name
            
        Returns:
            Sanitized filename-safe string
        """
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        sanitized = name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')
        
        # Limit length
        max_length = 200
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get overall storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        return {
            "output_directory": str(self.output_dir),
            "total_requests": len(self.list_requests()),
            "total_functions": len(self.list_all_functions()),
            "storage_size_bytes": self._get_directory_size(self.output_dir),
        }
    
    @staticmethod
    def _get_directory_size(directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total = 0
        try:
            for entry in directory.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating directory size: {e}")
        return total


# Convenience function for creating storage instance
def create_storage(output_dir: str = "outputs") -> ResultStorage:
    """
    Create a ResultStorage instance.
    
    Args:
        output_dir: Base directory for storing results
        
    Returns:
        Configured ResultStorage instance
    """
    return ResultStorage(Path(output_dir))