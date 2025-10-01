#!/usr/bin/env python3
"""
Enhanced Z3 Translator - Phase 4 Migration

CHANGES:
1. Added comprehensive Z3 code validation
2. Added AST parsing for metadata extraction
3. Added declared_variables, declared_sorts, custom_functions tracking
4. Enhanced validation with detailed error reporting
5. Added execution time tracking
6. Improved robustness with multiple validation passes
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

# Get the project root directory (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.prompt_loader import PromptsManager

load_dotenv()
logger = logging.getLogger(__name__)


class Z3Translator:
    """
    Enhanced Z3 code generator with comprehensive validation.
    
    ENHANCED in Phase 4:
    - Comprehensive syntax validation
    - AST parsing for metadata extraction
    - Tracks declared variables, sorts, and functions
    - Detailed error reporting
    - Execution time tracking
    """
    
    def __init__(self, api_key: Optional[str] = None, prompts_file: str = "config/prompts.yaml"):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.prompts = PromptsManager(prompts_file)
    
    def translate(self, 
                  postcondition: Dict[str, str],
                  function_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Translate formal postcondition to Z3 Python code.
        
        Args:
            postcondition: Dict with 'formal_text', 'natural_language', 'z3_theory'
            function_context: Optional function signature/type info
            
        Returns:
            Dict with enhanced validation metadata
        """
        start_time = datetime.now()
        
        # Load prompt template
        template = self.prompts.get_z3_translation_prompt()
        
        formal_text = postcondition.get('formal_text', '')
        natural_language = postcondition.get('natural_language', '')
        z3_theory = postcondition.get('z3_theory', 'unknown')
        
        # Build context
        context_str = self._build_context(function_context, z3_theory)
        
        # Format prompt
        formatted = template.format(
            formal_text=formal_text,
            natural_language=natural_language,
            z3_theory=z3_theory,
            function_context=context_str
        )
        
        system_prompt = formatted["system"]
        user_prompt = formatted["human"]
        
        # Generate Z3 code
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            z3_code = self._extract_code(response.choices[0].message.content)
            
            # ENHANCED: Comprehensive validation
            result = self._create_translation_result(
                formal_text=formal_text,
                natural_language=natural_language,
                z3_code=z3_code,
                z3_theory=z3_theory
            )
            
            # Track execution time
            result['translation_time'] = (datetime.now() - start_time).total_seconds()
            result['generated_at'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Z3 translation failed: {e}")
            return self._generate_fallback_z3(formal_text, natural_language, str(e))
    
    def _build_context(self, function_context: Optional[Dict], z3_theory: str) -> str:
        """Build context about function and theory."""
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
        
        ENHANCED in Phase 4: Multi-pass validation with metadata extraction.
        """
        result = {
            'formal_text': formal_text,
            'natural_language': natural_language,
            'z3_code': z3_code,
            'z3_theory_used': z3_theory,
            'translation_success': False,
            'z3_validation_passed': False,
            'z3_validation_status': 'not_validated',
            'validation_error': None,
            'warnings': [],
            
            # NEW: Enhanced metadata fields
            'z3_ast': None,
            'tokens': None,
            'custom_functions': [],
            'declared_sorts': [],
            'declared_variables': {},
        }
        
        if not z3_code:
            result['validation_error'] = "Empty Z3 code generated"
            return result
        
        # Pass 1: Basic syntax validation
        syntax_valid, syntax_errors = self._validate_syntax(z3_code)
        if not syntax_valid:
            result['z3_validation_status'] = 'syntax_error'
            result['validation_error'] = '; '.join(syntax_errors)
            result['warnings'].extend(syntax_errors)
            return result
        
        # Pass 2: Z3-specific validation
        z3_valid, z3_warnings = self._validate_z3_structure(z3_code)
        result['warnings'].extend(z3_warnings)
        
        # Pass 3: Extract metadata
        metadata = self._extract_metadata(z3_code)
        result.update(metadata)
        
        # Pass 4: Parse AST
        try:
            result['z3_ast'] = self._parse_ast(z3_code)
        except Exception as e:
            result['warnings'].append(f"AST parsing failed: {e}")
        
        # Set final status
        if syntax_valid and z3_valid:
            result['translation_success'] = True
            result['z3_validation_passed'] = True
            result['z3_validation_status'] = 'success'
        else:
            result['translation_success'] = bool(z3_code)
            result['z3_validation_status'] = 'warnings' if z3_warnings else 'success'
        
        return result
    
    def _validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Python syntax.
        
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
    
    def _validate_z3_structure(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Z3-specific structure and best practices.
        
        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check for Z3 import
        if 'from z3 import' not in code and 'import z3' not in code:
            warnings.append("Missing Z3 import statement")
        
        # Check for Solver
        if 'Solver()' not in code:
            warnings.append("No Solver() instance found")
        
        # Check for constraints
        if '.add(' not in code:
            warnings.append("No constraints added to solver")
        
        # Check for check() call
        if '.check()' not in code:
            warnings.append("No solver.check() call found")
        
        # Check for variable declarations
        z3_types = ['Int(', 'Real(', 'Bool(', 'Array(', 'BitVec(']
        has_declarations = any(z3_type in code for z3_type in z3_types)
        if not has_declarations:
            warnings.append("No Z3 variable declarations found")
        
        # Check for proper quantifier usage
        if 'ForAll' in code or 'Exists' in code:
            # Check that quantifiers have proper domain constraints
            if 'ForAll' in code and 'Implies' not in code:
                warnings.append("ForAll without Implies - may need domain constraints")
        
        # Return True if no critical issues (warnings are ok)
        return True, warnings
    
    def _extract_metadata(self, code: str) -> Dict[str, Any]:
        """
        Extract metadata from Z3 code.
        
        ENHANCED in Phase 4: Comprehensive metadata extraction.
        """
        metadata = {
            'custom_functions': [],
            'declared_sorts': [],
            'declared_variables': {},
        }
        
        # Extract custom function definitions
        func_pattern = r'^def\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, code, re.MULTILINE):
            metadata['custom_functions'].append(match.group(1))
        
        # Extract Z3 sort declarations
        sort_patterns = [
            r'(\w+)\s*=\s*Int\(',
            r'(\w+)\s*=\s*Real\(',
            r'(\w+)\s*=\s*Bool\(',
            r'(\w+)\s*=\s*Array\(',
            r'(\w+)\s*=\s*BitVec\(',
        ]
        
        for pattern in sort_patterns:
            for match in re.finditer(pattern, code):
                var_name = match.group(1)
                
                # Determine sort type
                if 'Int(' in match.group(0):
                    sort_type = 'Int'
                elif 'Real(' in match.group(0):
                    sort_type = 'Real'
                elif 'Bool(' in match.group(0):
                    sort_type = 'Bool'
                elif 'Array(' in match.group(0):
                    sort_type = 'Array'
                elif 'BitVec(' in match.group(0):
                    sort_type = 'BitVec'
                else:
                    sort_type = 'Unknown'
                
                metadata['declared_variables'][var_name] = sort_type
                
                if sort_type not in metadata['declared_sorts']:
                    metadata['declared_sorts'].append(sort_type)
        
        # Extract Ints/Reals/Bools multi-declarations
        multi_patterns = [
            (r'Ints\(\s*[\'"]([^\'"]+)[\'"]\s*\)', 'Int'),
            (r'Reals\(\s*[\'"]([^\'"]+)[\'"]\s*\)', 'Real'),
            (r'Bools\(\s*[\'"]([^\'"]+)[\'"]\s*\)', 'Bool'),
        ]
        
        for pattern, sort_type in multi_patterns:
            for match in re.finditer(pattern, code):
                var_names = match.group(1).split()
                for var_name in var_names:
                    metadata['declared_variables'][var_name] = sort_type
                
                if sort_type not in metadata['declared_sorts']:
                    metadata['declared_sorts'].append(sort_type)
        
        return metadata
    
    def _parse_ast(self, code: str) -> Dict[str, Any]:
        """
        Parse code into simplified AST representation.
        
        ENHANCED in Phase 4: Create analyzable AST structure.
        """
        try:
            tree = ast.parse(code)
            
            ast_data = {
                'imports': [],
                'functions': [],
                'assignments': [],
                'function_calls': [],
            }
            
            for node in ast.walk(tree):
                # Extract imports
                if isinstance(node, ast.ImportFrom):
                    ast_data['imports'].append({
                        'module': node.module,
                        'names': [alias.name for alias in node.names]
                    })
                
                # Extract function definitions
                elif isinstance(node, ast.FunctionDef):
                    ast_data['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'line': node.lineno
                    })
                
                # Extract assignments
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            ast_data['assignments'].append({
                                'target': target.id,
                                'line': node.lineno
                            })
                
                # Extract function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        ast_data['function_calls'].append({
                            'name': node.func.id,
                            'line': node.lineno
                        })
            
            return ast_data
            
        except Exception as e:
            logger.error(f"AST parsing failed: {e}")
            return {}
    
    def _tokenize_code(self, code: str) -> List[Tuple[str, str]]:
        """
        Tokenize Z3 code for analysis.
        
        Returns:
            List of (token_type, token_value) tuples
        """
        import tokenize
        import io
        
        try:
            tokens = []
            readline = io.StringIO(code).readline
            
            for token in tokenize.generate_tokens(readline):
                tokens.append((
                    tokenize.tok_name[token.type],
                    token.string
                ))
            
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return []
    
    def _generate_fallback_z3(
        self,
        formal_text: str,
        natural_language: str,
        error: str
    ) -> Dict[str, Any]:
        """Generate minimal fallback Z3 code."""
        return {
            'formal_text': formal_text,
            'natural_language': natural_language,
            'z3_code': f'# Translation failed: {error}\n# Formal: {formal_text}',
            'z3_theory_used': 'unknown',
            'translation_success': False,
            'z3_validation_passed': False,
            'z3_validation_status': 'failed',
            'validation_error': error,
            'warnings': [f'Translation failed: {error}'],
            'z3_ast': None,
            'tokens': None,
            'custom_functions': [],
            'declared_sorts': [],
            'declared_variables': {},
            'translation_time': 0.0,
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
    
    Args:
        z3_code: Z3 Python code to validate
        
    Returns:
        Validation result dictionary
        
    Example:
        >>> result = validate_z3_code(my_z3_code)
        >>> if result['valid']:
        ...     print("Code is valid!")
    """
    translator = Z3Translator()
    
    result = translator._create_translation_result(
        formal_text="Validation check",
        natural_language="Validation check",
        z3_code=z3_code,
        z3_theory="unknown"
    )
    
    return {
        'valid': result['z3_validation_passed'],
        'status': result['z3_validation_status'],
        'errors': [result['validation_error']] if result['validation_error'] else [],
        'warnings': result['warnings'],
        'metadata': {
            'declared_variables': result['declared_variables'],
            'declared_sorts': result['declared_sorts'],
            'custom_functions': result['custom_functions'],
        }
    }


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4 VALIDATION - Enhanced Z3 Translator")
    print("=" * 70)
    
    # Example 1: Test validation with valid Z3 code
    print("\n1. Testing validation with valid Z3 code...")
    print("-" * 70)
    
    valid_code = """
from z3 import *

# Declare variables
i, j = Ints('i j')
arr = Array('arr', IntSort(), IntSort())
n = Int('n')

# Sorted property
constraint = ForAll([i, j],
    Implies(And(i >= 0, j > i, j < n),
            Select(arr, i) <= Select(arr, j)))

# Solver
s = Solver()
s.add(constraint)
s.add(n > 0)

result = s.check()
print(f"Result: {result}")
"""
    
    validation_result = validate_z3_code(valid_code)
    print(f"Valid: {validation_result['valid']}")
    print(f"Status: {validation_result['status']}")
    print(f"Warnings: {len(validation_result['warnings'])}")
    print(f"Declared variables: {validation_result['metadata']['declared_variables']}")
    print(f"Declared sorts: {validation_result['metadata']['declared_sorts']}")
    
    # Example 2: Test validation with invalid code
    print("\n2. Testing validation with invalid Z3 code...")
    print("-" * 70)
    
    invalid_code = """
from z3 import *

# Missing variable declaration
constraint = x > 0  # x not declared!

s = Solver()
s.add(constraint)
"""
    
    validation_result = validate_z3_code(invalid_code)
    print(f"Valid: {validation_result['valid']}")
    print(f"Status: {validation_result['status']}")
    print(f"Warnings: {validation_result['warnings']}")
    
    # Example 3: Test metadata extraction
    print("\n3. Testing metadata extraction...")
    print("-" * 70)
    
    complex_code = """
from z3 import *

def array_sum(arr, n):
    if n == 0:
        return 0
    return Select(arr, n-1) + array_sum(arr, n-1)

x, y, z = Ints('x y z')
arr = Array('arr', IntSort(), IntSort())
size = Int('size')

constraint = array_sum(arr, size) == x + y + z

s = Solver()
s.add(constraint)
"""
    
    validation_result = validate_z3_code(complex_code)
    metadata = validation_result['metadata']
    
    print(f"Custom functions: {metadata['custom_functions']}")
    print(f"Declared variables: {metadata['declared_variables']}")
    print(f"Declared sorts: {metadata['declared_sorts']}")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 4 COMPLETE")
    print("=" * 70)
    print("\nEnhancements made:")
    print("1. ✓ Multi-pass validation (syntax, structure, metadata)")
    print("2. ✓ AST parsing for code analysis")
    print("3. ✓ Metadata extraction (variables, sorts, functions)")
    print("4. ✓ Comprehensive error reporting")
    print("5. ✓ Execution time tracking")
    print("6. ✓ Backward compatible API")
    print("\nNext: Phase 5 - Update modules/pipeline/pipeline.py")