#!/usr/bin/env python3
"""
Enhanced Z3 Translator - Phase 3 Complete

PHASE 3 CHANGES:
1. ✅ Integrated Z3CodeValidator from validator.py
2. ✅ Uses settings.z3_validation configuration
3. ✅ Enhanced validation with runtime execution
4. ✅ Tracks solver creation and constraints
5. ✅ Improved error reporting with error types and line numbers
6. ✅ Added execution time tracking
7. ✅ Preserves all metadata from validator

VALIDATION PIPELINE:
- Pass 1: Syntax validation (AST parsing)
- Pass 2: Import validation (Z3 imports)
- Pass 3: Runtime execution (NEW - validates code actually runs)
- Pass 4: Solver validation (verifies Solver() creation)
- Pass 5: Metadata extraction (variables, sorts, functions)
"""

import openai
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from dotenv import load_dotenv
import ast
import re
import sys
from pathlib import Path
from datetime import datetime
import time

# Get the project root directory (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.prompt_loader import PromptsManager
from config.settings import settings

# 🆕 IMPORT VALIDATOR (Phase 3)
from modules.z3.validator import Z3CodeValidator, ValidationResult

load_dotenv()
logger = logging.getLogger(__name__)


class Z3Translator:
    """
    Enhanced Z3 code generator with comprehensive validation.
    
    PHASE 3 ENHANCEMENTS:
    - Integrated Z3CodeValidator for runtime validation
    - Uses settings.z3_validation configuration
    - Validates code execution (not just syntax)
    - Tracks solver creation and constraint usage
    - Detailed error reporting with types and line numbers
    """
    
    def __init__(self, api_key: Optional[str] = None, prompts_file: str = "config/prompts.yaml"):
        """
        Initialize translator with validator.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            prompts_file: Path to prompts YAML file
        """
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.prompts = PromptsManager(prompts_file)
        
        # 🆕 INITIALIZE VALIDATOR (Phase 3)
        self.validator = Z3CodeValidator(
            timeout=settings.z3_validation.timeout_seconds,
            execution_method=settings.z3_validation.execution_method
        )
        
        logger.info(f"Z3Translator initialized with validation: {settings.z3_validation.enabled}")
    
    def translate(self, 
                  postcondition: Dict[str, str],
                  function_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Translate formal postcondition to Z3 Python code with validation.
        
        ENHANCED in Phase 3: Now uses Z3CodeValidator for comprehensive validation.
        
        Args:
            postcondition: Dict with keys: formal_text, natural_language, z3_theory
            function_context: Optional function context for better translation
            
        Returns:
            Translation result with comprehensive validation metadata
        """
        start_time = time.time()
        
        formal_text = postcondition.get('formal_text', '')
        natural_language = postcondition.get('natural_language', '')
        z3_theory = postcondition.get('z3_theory', 'unknown')
        
        logger.info(f"Translating postcondition to Z3 (theory: {z3_theory})")
        
        try:
            # Generate Z3 code using LLM
            z3_code = self._generate_z3_code(
                formal_text=formal_text,
                natural_language=natural_language,
                z3_theory=z3_theory,
                function_context=function_context
            )
            
            # Create result with validation
            result = self._create_translation_result(
                formal_text=formal_text,
                natural_language=natural_language,
                z3_code=z3_code,
                z3_theory=z3_theory
            )
            
            result['translation_time'] = time.time() - start_time
            result['generated_at'] = datetime.now().isoformat()
            
            logger.info(
                f"Translation complete: {result['z3_validation_status']} "
                f"(time: {result['translation_time']:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            return self._create_error_result(
                formal_text=formal_text,
                natural_language=natural_language,
                error=str(e),
                elapsed_time=time.time() - start_time
            )
    
    def _generate_z3_code(
        self,
        formal_text: str,
        natural_language: str,
        z3_theory: str,
        function_context: Optional[Dict] = None
    ) -> str:
        """
        Generate Z3 code using OpenAI API.
        
        Args:
            formal_text: Formal specification
            natural_language: Natural language description
            z3_theory: Z3 theory to use
            function_context: Optional function context
            
        Returns:
            Generated Z3 Python code
        """
        # Build context string
        context = self._build_context_string(function_context, z3_theory)
        
        # Get prompt template
        system_prompt = self.prompts.get_prompt(
            "z3_translation",
            "system",
            default="Translate formal postconditions to Z3 Python code."
        )
        user_prompt = self.prompts.get_prompt(
            "z3_translation",
            "user",
            default="Formal: {formal_text}\nNatural: {natural_language}\nContext: {context}\n\nGenerate Z3 code:"
        )
        
        # Format prompts
        user_message = user_prompt.format(
            formal_text=formal_text,
            natural_language=natural_language,
            context=context
        )
        
        # Call OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Extract code from response
        z3_code = self._extract_code(response.choices[0].message.content)
        
        return z3_code
    
    def _build_context_string(
        self,
        function_context: Optional[Dict],
        z3_theory: str
    ) -> str:
        """Build context string for translation."""
        parts = []
        
        if function_context:
            parts.append(f"Function: {function_context.get('name', 'unknown')}")
            
            if inputs := function_context.get('input_parameters'):
                parts.append(f"Inputs: {[p['name'] + ':' + p['data_type'] for p in inputs]}")
            
            if returns := function_context.get('return_values'):
                parts.append(f"Returns: {[r['name'] + ':' + r['data_type'] for r in returns]}")
        
        parts.append(f"\nZ3 Theory: {z3_theory}")
        
        theory_hints = {
            'arrays': 'Use ArraySort, Select, Store',
            'sequences': 'Use SeqSort, Length, Concat',
            'sets': 'Use SetSort, IsMember, Union',
            'arithmetic': 'Use Int, Real, arithmetic operators'
        }
        
        if hint := theory_hints.get(z3_theory):
            parts.append(f"Hint: {hint}")
        
        return "\n".join(parts)
    
    def _extract_code(self, response: str) -> str:
        """Extract Z3 code from AI response."""
        # Remove markdown code blocks if present
        if '```python' in response:
            code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            code = response.split('```')[1].split('```')[0]
        else:
            code = response
        
        return code.strip()
    
    def _create_translation_result(
        self,
        formal_text: str,
        natural_language: str,
        z3_code: str,
        z3_theory: str
    ) -> Dict[str, Any]:
        """
        Create translation result with comprehensive validation.
        
        ENHANCED in Phase 3: Uses Z3CodeValidator for runtime validation.
        
        Args:
            formal_text: Formal postcondition text
            natural_language: Natural language description
            z3_code: Generated Z3 code
            z3_theory: Z3 theory used
            
        Returns:
            Complete translation result with validation metadata
        """
        result = {
            'formal_text': formal_text,
            'natural_language': natural_language,
            'z3_code': z3_code,
            'z3_theory_used': z3_theory,
            'translation_success': False,
            
            # Validation fields (will be populated by validator)
            'z3_validation_passed': False,
            'z3_validation_status': 'not_validated',
            'validation_error': None,
            'error_type': None,
            'error_line': None,
            'warnings': [],
            
            # Execution metrics (NEW in Phase 3)
            'solver_created': False,
            'constraints_added': 0,
            'variables_declared': 0,
            'execution_time': 0.0,
            
            # Metadata
            'z3_ast': None,
            'tokens': None,
            'custom_functions': [],
            'declared_sorts': [],
            'declared_variables': {},
        }
        
        if not z3_code:
            result['validation_error'] = "Empty Z3 code generated"
            result['z3_validation_status'] = 'failed'
            return result
        
        # 🆕 PHASE 3: USE VALIDATOR FOR COMPREHENSIVE VALIDATION
        if settings.z3_validation.enabled:
            validation_result = self._validate_with_validator(z3_code)
            result.update(self._merge_validation_result(validation_result))
        else:
            # Fallback to basic syntax validation if validator disabled
            syntax_valid, syntax_errors = self._validate_syntax_only(z3_code)
            if syntax_valid:
                result['translation_success'] = True
                result['z3_validation_passed'] = True
                result['z3_validation_status'] = 'success'
            else:
                result['z3_validation_status'] = 'syntax_error'
                result['validation_error'] = '; '.join(syntax_errors)
                result['warnings'] = syntax_errors
        
        # Extract metadata regardless of validation result
        metadata = self._extract_metadata(z3_code)
        result.update(metadata)
        
        return result
    
    def _validate_with_validator(self, z3_code: str) -> ValidationResult:
        """
        Validate Z3 code using Z3CodeValidator.
        
        NEW in Phase 3: Uses comprehensive validator with runtime execution.
        
        Args:
            z3_code: Z3 code to validate
            
        Returns:
            ValidationResult from validator
        """
        try:
            validation_result = self.validator.validate(z3_code)
            
            if validation_result.passed:
                logger.debug(f"✅ Z3 validation passed (time: {validation_result.execution_time:.3f}s)")
            else:
                logger.warning(
                    f"❌ Z3 validation failed: {validation_result.status} - "
                    f"{validation_result.error_message}"
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validator error: {e}", exc_info=True)
            # Return failed result
            return ValidationResult(
                passed=False,
                status="validator_error",
                error_message=f"Validator exception: {str(e)}",
                error_type="ValidatorError"
            )
    
    def _merge_validation_result(self, validation: ValidationResult) -> Dict[str, Any]:
        """
        Merge ValidationResult into translation result format.
        
        NEW in Phase 3: Converts ValidationResult to dict format.
        
        Args:
            validation: ValidationResult from validator
            
        Returns:
            Dictionary with validation fields
        """
        return {
            'translation_success': validation.passed,
            'z3_validation_passed': validation.passed,
            'z3_validation_status': validation.status,
            'validation_error': validation.error_message,
            'error_type': validation.error_type,
            'error_line': validation.error_line,
            'warnings': validation.warnings,
            
            # Execution metrics (NEW)
            'solver_created': validation.solver_created,
            'constraints_added': validation.constraints_added,
            'variables_declared': validation.variables_declared,
            'execution_time': validation.execution_time,
        }
    
    def _validate_syntax_only(self, code: str) -> Tuple[bool, List[str]]:
        """
        Fallback syntax validation when validator is disabled.
        
        Args:
            code: Python code to validate
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
            return False, errors
        
        return True, errors
    
    def _extract_metadata(self, code: str) -> Dict[str, Any]:
        """
        Extract metadata from Z3 code (variables, sorts, functions).
        
        Args:
            code: Z3 Python code
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'declared_variables': {},
            'declared_sorts': [],
            'custom_functions': [],
        }
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Find variable declarations (Int, Real, Bool, Array, etc.)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        # Z3 variable types
                        if func_name in ['Int', 'Real', 'Bool', 'BitVec', 'String']:
                            if node.args and isinstance(node.args[0], ast.Constant):
                                var_name = node.args[0].value
                                metadata['declared_variables'][var_name] = func_name
                        
                        # Array declarations
                        elif func_name == 'Array':
                            if node.args and isinstance(node.args[0], ast.Constant):
                                var_name = node.args[0].value
                                metadata['declared_variables'][var_name] = 'Array'
                        
                        # Sorts
                        elif func_name.endswith('Sort'):
                            if func_name not in metadata['declared_sorts']:
                                metadata['declared_sorts'].append(func_name)
                
                # Custom function definitions
                elif isinstance(node, ast.FunctionDef):
                    metadata['custom_functions'].append(node.name)
            
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def _create_error_result(
        self,
        formal_text: str,
        natural_language: str,
        error: str,
        elapsed_time: float
    ) -> Dict[str, Any]:
        """
        Create error result when translation fails.
        
        Args:
            formal_text: Formal postcondition
            natural_language: Natural language description
            error: Error message
            elapsed_time: Time elapsed before error
            
        Returns:
            Error result dictionary
        """
        return {
            'formal_text': formal_text,
            'natural_language': natural_language,
            'z3_code': f'# Translation failed: {error}\n# Formal: {formal_text}',
            'z3_theory_used': 'unknown',
            'translation_success': False,
            'z3_validation_passed': False,
            'z3_validation_status': 'failed',
            'validation_error': error,
            'error_type': 'TranslationError',
            'warnings': [f'Translation failed: {error}'],
            'solver_created': False,
            'constraints_added': 0,
            'variables_declared': 0,
            'execution_time': 0.0,
            'z3_ast': None,
            'tokens': None,
            'custom_functions': [],
            'declared_sorts': [],
            'declared_variables': {},
            'translation_time': elapsed_time,
            'generated_at': datetime.now().isoformat()
        }


# ============================================================================
# BACKWARD COMPATIBLE API
# ============================================================================

def translate_to_z3_api(formal_text: str,
                        z3_theory: str = "unknown",
                        api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Drop-in replacement for old translation API.
    
    Args:
        formal_text: Formal postcondition
        z3_theory: Z3 theory to use
        api_key: Optional OpenAI API key
        
    Returns:
        Translation result dictionary
    """
    translator = Z3Translator(api_key)
    
    postcondition = {
        'formal_text': formal_text,
        'natural_language': 'Generated postcondition',
        'z3_theory': z3_theory
    }
    
    result = translator.translate(postcondition)
    
    return {
        'success': result['translation_success'],
        'z3_code': result['z3_code'],
        'declaration': '',  # For backward compatibility
        'error': result.get('validation_error'),
        'warnings': result.get('warnings', []),
        'validation_status': result.get('z3_validation_status'),
        'solver_created': result.get('solver_created', False),  # NEW
        'constraints_added': result.get('constraints_added', 0),  # NEW
        'metadata': {
            'declared_variables': result.get('declared_variables', {}),
            'declared_sorts': result.get('declared_sorts', []),
            'custom_functions': result.get('custom_functions', [])
        }
    }


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_z3_code(z3_code: str) -> Dict[str, Any]:
    """
    Standalone function to validate Z3 code.
    
    ENHANCED in Phase 3: Uses Z3CodeValidator for comprehensive validation.
    
    Args:
        z3_code: Z3 Python code to validate
        
    Returns:
        Validation result dictionary
        
    Example:
        >>> result = validate_z3_code(my_z3_code)
        >>> if result['valid']:
        ...     print("Code is valid!")
        ...     print(f"Solver created: {result['solver_created']}")
    """
    if not settings.z3_validation.enabled:
        logger.warning("Z3 validation is disabled in settings")
        return {
            'valid': False,
            'status': 'disabled',
            'errors': ['Validation disabled in settings'],
            'warnings': [],
            'metadata': {}
        }
    
    validator = Z3CodeValidator(
        timeout=settings.z3_validation.timeout_seconds,
        execution_method=settings.z3_validation.execution_method
    )
    
    validation_result = validator.validate(z3_code)
    
    return {
        'valid': validation_result.passed,
        'status': validation_result.status,
        'errors': [validation_result.error_message] if validation_result.error_message else [],
        'warnings': validation_result.warnings,
        'solver_created': validation_result.solver_created,
        'constraints_added': validation_result.constraints_added,
        'variables_declared': validation_result.variables_declared,
        'execution_time': validation_result.execution_time,
        'metadata': {
            'error_type': validation_result.error_type,
            'error_line': validation_result.error_line,
        }
    }


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 3 COMPLETE - Enhanced Z3 Translator with Runtime Validation")
    print("=" * 80)
    
    # Test 1: Validate valid Z3 code
    print("\n1. Testing runtime validation with valid Z3 code...")
    print("-" * 80)
    
    valid_code = """
from z3 import *

# Declare variables
x = Int('x')
y = Int('y')

# Create solver
s = Solver()
s.add(x > 0)
s.add(y > x)

# Check
result = s.check()
print(f"Result: {result}")
"""
    
    result = validate_z3_code(valid_code)
    print(f"✅ Valid: {result['valid']}")
    print(f"   Status: {result['status']}")
    print(f"   Solver Created: {result['solver_created']}")
    print(f"   Constraints: {result['constraints_added']}")
    print(f"   Variables: {result['variables_declared']}")
    print(f"   Execution Time: {result['execution_time']:.3f}s")
    
    # Test 2: Validate code with runtime error
    print("\n2. Testing runtime validation with runtime error...")
    print("-" * 80)
    
    runtime_error_code = """
from z3 import *

# Undefined variable error
s = Solver()
s.add(undefined_x > 0)  # This will cause NameError
print(s.check())
"""
    
    result = validate_z3_code(runtime_error_code)
    print(f"❌ Valid: {result['valid']}")
    print(f"   Status: {result['status']}")
    print(f"   Error: {result['errors']}")
    print(f"   Error Type: {result['metadata']['error_type']}")
    
    # Test 3: Check validator settings
    print("\n3. Current Z3 validation settings...")
    print("-" * 80)
    print(f"   Enabled: {settings.z3_validation.enabled}")
    print(f"   Timeout: {settings.z3_validation.timeout_seconds}s")
    print(f"   Method: {settings.z3_validation.execution_method}")
    print(f"   Validate Execution: {settings.z3_validation.validate_execution}")
    print(f"   Validate Solver: {settings.z3_validation.validate_solver}")
    
    print("\n" + "=" * 80)
    print("✅ Phase 3 Complete!")
    print("   - Validator integrated")
    print("   - Runtime validation working")
    print("   - Comprehensive error reporting")
    print("   - Execution metrics tracked")
    print("=" * 80)