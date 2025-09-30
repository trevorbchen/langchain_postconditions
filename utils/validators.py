"""
Validation Utilities

This module provides validation functions for inputs and outputs throughout
the postcondition generation system.

Key features:
- Type validation
- Format validation
- Content validation
- Error reporting with helpful messages
"""

from typing import Optional, List, Tuple
import re
import ast

from core.models import (
    Function,
    EnhancedPostcondition,
    Z3Translation,
    CompleteEnhancedResult
)


# ============================================================================
# FUNCTION VALIDATION
# ============================================================================

def validate_function(function: Function) -> Tuple[bool, List[str]]:
    """
    Validate a Function model.
    
    Args:
        function: Function to validate
        
    Returns:
        (is_valid, list_of_errors)
        
    Example:
        >>> func = Function(name="test", description="Test")
        >>> is_valid, errors = validate_function(func)
        >>> if not is_valid:
        ...     print("Errors:", errors)
    """
    errors = []
    
    # Check name
    if not function.name or not function.name.strip():
        errors.append("Function name cannot be empty")
    elif not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', function.name):
        errors.append(f"Invalid function name: '{function.name}' (must be valid C identifier)")
    
    # Check description
    if not function.description or not function.description.strip():
        errors.append("Function description cannot be empty")
    
    # Check return type
    if not function.return_type:
        errors.append("Function must have a return type")
    
    # Check parameters
    for i, param in enumerate(function.input_parameters):
        param_valid, param_errors = validate_parameter(param)
        if not param_valid:
            errors.extend([f"Parameter {i} ({param.name}): {e}" for e in param_errors])
    
    # Check complexity format
    if function.complexity and not re.match(r'O\([^)]+\)', function.complexity):
        errors.append(f"Invalid complexity format: '{function.complexity}' (should be O(n), O(log n), etc.)")
    
    return len(errors) == 0, errors


def validate_parameter(param) -> Tuple[bool, List[str]]:
    """
    Validate a function parameter.
    
    Args:
        param: FunctionParameter to validate
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if not param.name or not param.name.strip():
        errors.append("Parameter name cannot be empty")
    elif not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', param.name):
        errors.append(f"Invalid parameter name: '{param.name}'")
    
    if not param.data_type or not param.data_type.strip():
        errors.append("Parameter data type cannot be empty")
    
    return len(errors) == 0, errors


# ============================================================================
# POSTCONDITION VALIDATION
# ============================================================================

def validate_postcondition(
    postcondition: EnhancedPostcondition
) -> Tuple[bool, List[str]]:
    """
    Validate a postcondition.
    
    Args:
        postcondition: Postcondition to validate
        
    Returns:
        (is_valid, list_of_errors)
        
    Example:
        >>> pc = EnhancedPostcondition(
        ...     formal_text="x > 0",
        ...     natural_language="x is positive"
        ... )
        >>> is_valid, errors = validate_postcondition(pc)
    """
    errors = []
    
    # Check formal text
    if not postcondition.formal_text or not postcondition.formal_text.strip():
        errors.append("Formal text cannot be empty")
    
    # Check natural language
    if not postcondition.natural_language or not postcondition.natural_language.strip():
        errors.append("Natural language description cannot be empty")
    
    # Check confidence score
    if not 0 <= postcondition.confidence_score <= 1:
        errors.append(f"Confidence score must be between 0 and 1, got {postcondition.confidence_score}")
    
    # Check quality scores if set
    if postcondition.clarity_score != 0.0:
        if not 0 <= postcondition.clarity_score <= 1:
            errors.append(f"Clarity score must be between 0 and 1, got {postcondition.clarity_score}")
    
    if postcondition.completeness_score != 0.0:
        if not 0 <= postcondition.completeness_score <= 1:
            errors.append(f"Completeness score must be between 0 and 1, got {postcondition.completeness_score}")
    
    if postcondition.testability_score != 0.0:
        if not 0 <= postcondition.testability_score <= 1:
            errors.append(f"Testability score must be between 0 and 1, got {postcondition.testability_score}")
    
    return len(errors) == 0, errors


# ============================================================================
# Z3 CODE VALIDATION
# ============================================================================

def validate_z3_code(z3_code: str) -> Tuple[bool, List[str]]:
    """
    Validate Z3 Python code syntax.
    
    Args:
        z3_code: Z3 code string
        
    Returns:
        (is_valid, list_of_errors)
        
    Example:
        >>> code = "from z3 import *\\nx = Int('x')"
        >>> is_valid, errors = validate_z3_code(code)
    """
    errors = []
    warnings = []
    
    if not z3_code or not z3_code.strip():
        errors.append("Z3 code cannot be empty")
        return False, errors
    
    # Check Python syntax
    try:
        ast.parse(z3_code)
    except SyntaxError as e:
        errors.append(f"Python syntax error: {e}")
        return False, errors
    
    # Check for Z3 import
    if 'from z3 import' not in z3_code and 'import z3' not in z3_code:
        warnings.append("Missing Z3 import statement")
    
    # Check for Solver
    if 'Solver()' not in z3_code:
        warnings.append("No Solver() instance found")
    
    # Check for constraints
    if '.add(' not in z3_code:
        warnings.append("No constraints added to solver")
    
    # Check for check() call
    if '.check()' not in z3_code:
        warnings.append("No solver.check() call found")
    
    # Return warnings as part of errors for visibility
    return True, warnings


def validate_z3_translation(
    translation: Z3Translation
) -> Tuple[bool, List[str]]:
    """
    Validate a complete Z3Translation.
    
    Args:
        translation: Z3Translation to validate
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if not translation.formal_text:
        errors.append("Translation must have formal text")
    
    if translation.translation_success and not translation.z3_code:
        errors.append("Translation marked successful but no Z3 code generated")
    
    if translation.z3_code:
        code_valid, code_errors = validate_z3_code(translation.z3_code)
        if not code_valid:
            errors.extend(code_errors)
    
    return len(errors) == 0, errors


# ============================================================================
# SPECIFICATION VALIDATION
# ============================================================================

def validate_specification(specification: str) -> Tuple[bool, List[str]]:
    """
    Validate a natural language specification.
    
    Args:
        specification: Specification text
        
    Returns:
        (is_valid, list_of_errors)
        
    Example:
        >>> is_valid, errors = validate_specification("Sort an array")
        >>> if is_valid:
        ...     print("Valid specification")
    """
    errors = []
    warnings = []
    
    if not specification or not specification.strip():
        errors.append("Specification cannot be empty")
        return False, errors
    
    # Check minimum length
    if len(specification.strip()) < 5:
        warnings.append("Specification is very short, consider adding more detail")
    
    # Check if it's just a function name (likely needs more context)
    if len(specification.split()) <= 2:
        warnings.append("Specification is very brief, consider adding more context")
    
    # Check for common issues
    if specification.isupper():
        warnings.append("Specification is all caps, consider normal case")
    
    if not any(char.isalpha() for char in specification):
        errors.append("Specification must contain some text")
    
    return len(errors) == 0, errors + warnings


# ============================================================================
# RESULT VALIDATION
# ============================================================================

def validate_pipeline_result(
    result: CompleteEnhancedResult
) -> Tuple[bool, List[str]]:
    """
    Validate a complete pipeline result.
    
    Args:
        result: CompleteEnhancedResult to validate
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    warnings = []
    
    # Check session ID
    if not result.session_id:
        errors.append("Result must have a session ID")
    
    # Check specification
    spec_valid, spec_errors = validate_specification(result.specification)
    if not spec_valid:
        errors.extend([f"Specification: {e}" for e in spec_errors])
    
    # Check consistency
    if result.pseudocode_success and not result.functions_created:
        warnings.append("Pseudocode successful but no functions created")
    
    if result.total_postconditions == 0 and len(result.function_results) > 0:
        warnings.append("Functions generated but no postconditions")
    
    # Check statistics consistency
    calculated_total = sum(fr.postcondition_count for fr in result.function_results)
    if calculated_total != result.total_postconditions:
        errors.append(
            f"Inconsistent postcondition count: "
            f"sum of function results ({calculated_total}) != "
            f"total_postconditions ({result.total_postconditions})"
        )
    
    return len(errors) == 0, errors + warnings


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_and_report(obj, validator_func) -> bool:
    """
    Validate and print report.
    
    Args:
        obj: Object to validate
        validator_func: Validation function to use
        
    Returns:
        True if valid
        
    Example:
        >>> func = Function(name="test", description="Test")
        >>> if validate_and_report(func, validate_function):
        ...     print("Valid!")
    """
    is_valid, messages = validator_func(obj)
    
    if is_valid:
        print("✅ Validation passed")
        if messages:
            print("⚠️  Warnings:")
            for msg in messages:
                print(f"   - {msg}")
    else:
        print("❌ Validation failed")
        for msg in messages:
            print(f"   - {msg}")
    
    return is_valid


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from core.models import FunctionParameter, PostconditionCategory
    
    print("=" * 70)
    print("VALIDATORS - EXAMPLE USAGE")
    print("=" * 70)
    
    # Example 1: Validate function
    print("\n✅ Example 1: Valid function")
    print("-" * 70)
    
    valid_func = Function(
        name="bubble_sort",
        description="Sort an array using bubble sort",
        return_type="void",
        input_parameters=[
            FunctionParameter(name="arr", data_type="int[]"),
            FunctionParameter(name="size", data_type="int")
        ],
        complexity="O(n^2)"
    )
    
    validate_and_report(valid_func, validate_function)
    
    # Example 2: Invalid function
    print("\n❌ Example 2: Invalid function")
    print("-" * 70)
    
    invalid_func = Function(
        name="123invalid",  # Invalid name
        description="",  # Empty description
        return_type="void",
        complexity="bad complexity"  # Invalid format
    )
    
    validate_and_report(invalid_func, validate_function)
    
    # Example 3: Validate postcondition
    print("\n✅ Example 3: Valid postcondition")
    print("-" * 70)
    
    valid_pc = EnhancedPostcondition(
        formal_text="∀i: arr[i] ≤ arr[i+1]",
        natural_language="Array is sorted",
        confidence_score=0.95,
        category=PostconditionCategory.CORRECTNESS
    )
    
    validate_and_report(valid_pc, validate_postcondition)
    
    # Example 4: Validate Z3 code
    print("\n✅ Example 4: Valid Z3 code")
    print("-" * 70)
    
    z3_code = """
from z3 import *

x = Int('x')
s = Solver()
s.add(x > 0)
result = s.check()
print(result)
"""
    
    is_valid, messages = validate_z3_code(z3_code)
    if is_valid:
        print("✅ Z3 code is valid")
        if messages:
            print("Notes:")
            for msg in messages:
                print(f"   - {msg}")
    
    # Example 5: Validate specification
    print("\n⚠️  Example 5: Specification with warnings")
    print("-" * 70)
    
    is_valid, messages = validate_specification("sort")
    print(f"Valid: {is_valid}")
    for msg in messages:
        print(f"   - {msg}")
    
    print("\n" + "=" * 70)
    print("✅ EXAMPLES COMPLETED")
    print("=" * 70)