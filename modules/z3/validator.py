"""
Z3 Code Validation Module

Single comprehensive module containing:
- Code validation logic (syntax, imports, structure)
- Isolated execution environment
- Validation result tracking
- Summary reporting

This module validates Z3 code by:
1. Checking Python syntax
2. Verifying Z3 imports
3. Executing code in isolated environment
4. Verifying Solver() creation
5. Tracking execution metrics
"""

import ast
import sys
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter, defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ValidationResult:
    """Result of Z3 code validation."""
    
    passed: bool
    status: str  # "success", "syntax_error", "import_error", "runtime_error", "timeout_error"
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    error_line: Optional[int] = None
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    # Execution details
    solver_created: bool = False
    constraints_added: int = 0
    variables_declared: int = 0
    
    # Output capture
    stdout: str = ""
    stderr: str = ""


@dataclass
class ValidationSummary:
    """Summary statistics for validation results."""
    
    total_validations: int
    successful_validations: int
    failed_validations: int
    success_rate: float
    
    # Error breakdown
    syntax_errors: int = 0
    import_errors: int = 0
    runtime_errors: int = 0
    timeout_errors: int = 0
    
    # Performance metrics
    avg_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = 0.0
    total_execution_time: float = 0.0
    
    # Quality metrics
    avg_constraints_per_code: float = 0.0
    avg_variables_per_code: float = 0.0
    solver_creation_rate: float = 0.0
    
    # Timing
    report_generated_at: str = ""
    
    # Detailed error info
    error_breakdown: Dict[str, int] = field(default_factory=dict)
    common_error_patterns: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Z3 CODE VALIDATOR
# ============================================================================

class Z3CodeValidator:
    """
    Validates Z3 code through multiple passes.
    
    Validation pipeline:
    1. Syntax validation (AST parsing)
    2. Import validation (Z3 imports present)
    3. Execution validation (run code in isolated environment)
    4. Solver validation (verify Solver() created)
    """
    
    def __init__(self, timeout: int = 5, execution_method: str = "subprocess"):
        """
        Initialize validator.
        
        Args:
            timeout: Maximum execution time in seconds
            execution_method: "subprocess" (safer) or "exec" (faster)
        """
        self.timeout = timeout
        self.execution_method = execution_method
        
    def validate(self, z3_code: str) -> ValidationResult:
        """
        Run full validation pipeline.
        
        Args:
            z3_code: Z3 Python code to validate
            
        Returns:
            ValidationResult with detailed status
        """
        start_time = time.time()
        result = ValidationResult(
            passed=False,
            status="not_validated"
        )
        
        # Pass 1: Syntax validation
        syntax_valid, syntax_error = self._validate_syntax(z3_code)
        if not syntax_valid:
            result.status = "syntax_error"
            result.error_message = syntax_error
            result.error_type = "SyntaxError"
            result.execution_time = time.time() - start_time
            return result
        
        # Pass 2: Import validation
        imports_valid, import_error = self._validate_imports(z3_code)
        if not imports_valid:
            result.status = "import_error"
            result.error_message = import_error
            result.error_type = "ImportError"
            result.execution_time = time.time() - start_time
            result.warnings.append(import_error)
            return result
        
        # Pass 3: Execution validation
        exec_result = self._execute_code(z3_code)
        result.stdout = exec_result.get("stdout", "")
        result.stderr = exec_result.get("stderr", "")
        result.execution_time = exec_result.get("execution_time", 0.0)
        
        if not exec_result.get("success", False):
            result.status = exec_result.get("status", "runtime_error")
            result.error_message = exec_result.get("error_message", "Unknown error")
            result.error_type = exec_result.get("error_type", "RuntimeError")
            return result
        
        # Pass 4: Solver validation
        solver_check = self._validate_solver_creation(z3_code, result.stdout)
        result.solver_created = solver_check["solver_created"]
        result.constraints_added = solver_check["constraints_count"]
        result.variables_declared = solver_check["variables_count"]
        
        if solver_check["warnings"]:
            result.warnings.extend(solver_check["warnings"])
        
        # Success!
        result.passed = True
        result.status = "success"
        result.execution_time = time.time() - start_time
        
        return result
    
    def _validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax using AST parsing.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            return False, error_msg
        except Exception as e:
            error_msg = f"Parse error: {str(e)}"
            return False, error_msg
    
    def _validate_imports(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that Z3 imports are present.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for Z3 import statements
        has_z3_import = (
            'from z3 import' in code or
            'import z3' in code
        )
        
        if not has_z3_import:
            return False, "Missing Z3 import statement (need 'from z3 import *' or 'import z3')"
        
        return True, None
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Z3 code in isolated environment.
        
        Returns:
            Dictionary with execution results
        """
        if self.execution_method == "subprocess":
            return self._execute_subprocess(code)
        else:
            return self._execute_inline(code)
    
    def _execute_subprocess(self, code: str) -> Dict[str, Any]:
        """
        Execute code in subprocess (safer, more isolated).
        
        Returns:
            Execution result dictionary
        """
        try:
            start_time = time.time()
            
            # Run code in subprocess with timeout
            result = subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Check for errors
            if result.returncode != 0:
                # Parse error type from stderr
                error_type = "RuntimeError"
                if "NameError" in result.stderr:
                    error_type = "NameError"
                elif "TypeError" in result.stderr:
                    error_type = "TypeError"
                elif "AttributeError" in result.stderr:
                    error_type = "AttributeError"
                elif "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
                    error_type = "ImportError"
                
                return {
                    "success": False,
                    "status": "runtime_error",
                    "error_message": result.stderr.strip(),
                    "error_type": error_type,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": execution_time
                }
            
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "status": "timeout_error",
                "error_message": f"Execution timed out after {self.timeout} seconds (infinite loop or too complex)",
                "error_type": "TimeoutError",
                "stdout": "",
                "stderr": "",
                "execution_time": self.timeout
            }
        except Exception as e:
            return {
                "success": False,
                "status": "runtime_error",
                "error_message": str(e),
                "error_type": type(e).__name__,
                "stdout": "",
                "stderr": str(e),
                "execution_time": 0.0
            }
    
    def _execute_inline(self, code: str) -> Dict[str, Any]:
        """
        Execute code inline with exec() (faster but less isolated).
        
        Returns:
            Execution result dictionary
        """
        try:
            start_time = time.time()
            
            # Create isolated namespace
            namespace = {
                '__builtins__': __builtins__,
            }
            
            # Capture stdout/stderr
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "status": "runtime_error",
                "error_message": str(e),
                "error_type": type(e).__name__,
                "stdout": "",
                "stderr": str(e),
                "execution_time": execution_time
            }
    
    def _validate_solver_creation(self, code: str, stdout: str) -> Dict[str, Any]:
        """
        Validate that Solver() was created and used.
        
        Args:
            code: Z3 code
            stdout: Captured output from execution
            
        Returns:
            Dictionary with solver validation results
        """
        result = {
            "solver_created": False,
            "constraints_count": 0,
            "variables_count": 0,
            "warnings": []
        }
        
        # Check if Solver() appears in code
        if 'Solver()' in code or 's = Solver' in code:
            result["solver_created"] = True
        else:
            result["warnings"].append("No Solver() instance found in code")
        
        # Count constraints (s.add calls)
        result["constraints_count"] = code.count('.add(')
        
        # Count variable declarations (heuristic)
        # Look for Int(), Real(), Bool(), Array(), etc.
        variable_patterns = ['Int(', 'Real(', 'Bool(', 'Array(', 'BitVec(']
        for pattern in variable_patterns:
            result["variables_count"] += code.count(pattern)
        
        # Check if check() was called
        if '.check()' not in code:
            result["warnings"].append("Solver.check() not called")
        
        # Check output for results
        if 'sat' in stdout.lower() or 'unsat' in stdout.lower():
            # Good - solver ran and produced results
            pass
        elif result["solver_created"]:
            result["warnings"].append("Solver created but no results in output")
        
        return result


# ============================================================================
# VALIDATION REPORTER
# ============================================================================

class ValidationReporter:
    """Generates summary reports from validation results."""
    
    @staticmethod
    def generate_summary(results: List[ValidationResult]) -> ValidationSummary:
        """
        Generate comprehensive summary from validation results.
        
        Args:
            results: List of validation results
            
        Returns:
            ValidationSummary with statistics
        """
        if not results:
            return ValidationSummary(
                total_validations=0,
                successful_validations=0,
                failed_validations=0,
                success_rate=0.0,
                report_generated_at=datetime.now().isoformat()
            )
        
        total = len(results)
        successful = sum(1 for r in results if r.passed)
        failed = total - successful
        
        # Count error types
        error_counter = Counter(r.error_type for r in results if r.error_type)
        
        # Execution time statistics
        exec_times = [r.execution_time for r in results]
        
        # Quality metrics
        constraints = [r.constraints_added for r in results]
        variables = [r.variables_declared for r in results]
        solvers_created = sum(1 for r in results if r.solver_created)
        
        # Error analysis
        error_breakdown = dict(error_counter)
        common_patterns = ValidationReporter._find_error_patterns(results)
        
        return ValidationSummary(
            total_validations=total,
            successful_validations=successful,
            failed_validations=failed,
            success_rate=successful / total if total > 0 else 0.0,
            
            # Error counts
            syntax_errors=error_counter.get("SyntaxError", 0),
            import_errors=error_counter.get("ImportError", 0),
            runtime_errors=error_counter.get("RuntimeError", 0) + 
                          error_counter.get("NameError", 0) +
                          error_counter.get("TypeError", 0) +
                          error_counter.get("AttributeError", 0),
            timeout_errors=error_counter.get("TimeoutError", 0),
            
            # Performance
            avg_execution_time=sum(exec_times) / len(exec_times) if exec_times else 0.0,
            max_execution_time=max(exec_times) if exec_times else 0.0,
            min_execution_time=min(exec_times) if exec_times else 0.0,
            total_execution_time=sum(exec_times),
            
            # Quality
            avg_constraints_per_code=sum(constraints) / len(constraints) if constraints else 0.0,
            avg_variables_per_code=sum(variables) / len(variables) if variables else 0.0,
            solver_creation_rate=solvers_created / total if total > 0 else 0.0,
            
            # Metadata
            report_generated_at=datetime.now().isoformat(),
            error_breakdown=error_breakdown,
            common_error_patterns=common_patterns
        )
    
    @staticmethod
    def _find_error_patterns(results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """
        Find common error patterns in failed validations.
        
        Returns:
            List of pattern dictionaries with counts
        """
        failed = [r for r in results if not r.passed and r.error_message]
        if not failed:
            return []
        
        patterns = []
        error_messages = [r.error_message for r in failed]
        
        # Common error patterns to check
        pattern_checks = {
            "undefined variable": "NameError: Variables not properly defined",
            "solver": "Solver object issues",
            "import": "Import statement problems",
            "syntax error": "Python syntax errors",
            "timeout": "Execution timeout (infinite loop or too complex)",
            "type": "Type errors (wrong argument types)",
            "attribute": "AttributeError (invalid Z3 methods)"
        }
        
        for pattern_key, description in pattern_checks.items():
            matching = [msg for msg in error_messages if pattern_key.lower() in msg.lower()]
            if matching:
                patterns.append({
                    "pattern": description,
                    "count": len(matching),
                    "percentage": len(matching) / len(failed) * 100,
                    "example": matching[0][:200]  # First 200 chars
                })
        
        return sorted(patterns, key=lambda x: x["count"], reverse=True)
    
    @staticmethod
    def format_summary(summary: ValidationSummary) -> str:
        """
        Format summary as readable text report.
        
        Args:
            summary: ValidationSummary to format
            
        Returns:
            Formatted string report
        """
        lines = [
            "=" * 80,
            "Z3 CODE VALIDATION SUMMARY",
            "=" * 80,
            "",
            f"📊 Overview:",
            f"  Total Validations:  {summary.total_validations}",
            f"  ✅ Successful:      {summary.successful_validations} ({summary.success_rate:.1%})",
            f"  ❌ Failed:          {summary.failed_validations}",
            "",
            f"⚠️ Error Breakdown:",
            f"  Syntax Errors:      {summary.syntax_errors}",
            f"  Import Errors:      {summary.import_errors}",
            f"  Runtime Errors:     {summary.runtime_errors}",
            f"  Timeout Errors:     {summary.timeout_errors}",
            "",
            f"⚡ Performance:",
            f"  Avg Execution Time: {summary.avg_execution_time:.3f}s",
            f"  Max Execution Time: {summary.max_execution_time:.3f}s",
            f"  Min Execution Time: {summary.min_execution_time:.3f}s",
            f"  Total Time:         {summary.total_execution_time:.3f}s",
            "",
            f"📈 Quality Metrics:",
            f"  Solver Creation Rate:     {summary.solver_creation_rate:.1%}",
            f"  Avg Constraints per Code: {summary.avg_constraints_per_code:.1f}",
            f"  Avg Variables per Code:   {summary.avg_variables_per_code:.1f}",
        ]
        
        if summary.common_error_patterns:
            lines.extend([
                "",
                f"🔍 Common Error Patterns:",
            ])
            for pattern in summary.common_error_patterns[:5]:  # Top 5
                lines.append(
                    f"  • {pattern['pattern']}: {pattern['count']} occurrences ({pattern['percentage']:.1f}%)"
                )
        
        lines.extend([
            "",
            f"Generated: {summary.report_generated_at}",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_recommendations(summary: ValidationSummary) -> List[str]:
        """
        Generate actionable recommendations based on validation results.
        
        Args:
            summary: ValidationSummary to analyze
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if summary.failed_validations == 0:
            recommendations.append("✅ Perfect! All Z3 code validated successfully.")
            return recommendations
        
        failure_rate = summary.failed_validations / summary.total_validations if summary.total_validations > 0 else 0
        
        # High failure rate
        if failure_rate > 0.3:
            recommendations.append(
                f"⚠️ High failure rate ({failure_rate:.1%}). Consider reviewing Z3 code generation prompts and templates."
            )
        
        # Syntax errors
        if summary.syntax_errors > 0:
            recommendations.append(
                "🔧 Syntax errors detected. Review code generation templates for proper Python syntax. "
                "Ensure proper indentation and bracket matching."
            )
        
        # Import errors
        if summary.import_errors > 0:
            recommendations.append(
                "📦 Import errors found. Ensure 'from z3 import *' is included at the top of all generated Z3 code."
            )
        
        # Runtime errors
        if summary.runtime_errors > 0:
            recommendations.append(
                "⚡ Runtime errors detected. Common issues: undefined variables, incorrect Z3 API usage, "
                "wrong function signatures. Review variable declarations and Z3 function calls."
            )
        
        # Timeout errors
        if summary.timeout_errors > 0:
            recommendations.append(
                "⏱️ Timeout errors found. Generated constraints may be too complex, contain infinite loops, "
                "or have performance issues. Simplify constraint logic or increase timeout."
            )
        
        # Low solver creation rate
        if summary.solver_creation_rate < 0.8:
            recommendations.append(
                f"🔍 Low solver creation rate ({summary.solver_creation_rate:.1%}). "
                "Ensure 'Solver()' is properly instantiated in generated code. "
                "Add validation to check solver creation."
            )
        
        # Performance issues
        if summary.avg_execution_time > 2.0:
            recommendations.append(
                f"🐌 Slow average execution time ({summary.avg_execution_time:.2f}s). "
                "Generated Z3 code may be overly complex. Consider simplifying constraints."
            )
        
        return recommendations


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_z3_code(
    z3_code: str,
    timeout: int = 5,
    execution_method: str = "subprocess"
) -> ValidationResult:
    """
    Convenience function to validate a single Z3 code snippet.
    
    Args:
        z3_code: Z3 Python code to validate
        timeout: Maximum execution time in seconds
        execution_method: "subprocess" or "exec"
        
    Returns:
        ValidationResult
    """
    validator = Z3CodeValidator(timeout=timeout, execution_method=execution_method)
    return validator.validate(z3_code)


def validate_z3_codes(
    z3_codes: List[str],
    timeout: int = 5,
    execution_method: str = "subprocess"
) -> List[ValidationResult]:
    """
    Validate multiple Z3 code snippets.
    
    Args:
        z3_codes: List of Z3 Python code strings
        timeout: Maximum execution time in seconds
        execution_method: "subprocess" or "exec"
        
    Returns:
        List of ValidationResult
    """
    validator = Z3CodeValidator(timeout=timeout, execution_method=execution_method)
    return [validator.validate(code) for code in z3_codes]


def generate_validation_report(results: List[ValidationResult]) -> str:
    """
    Generate a formatted validation report.
    
    Args:
        results: List of validation results
        
    Returns:
        Formatted report string
    """
    summary = ValidationReporter.generate_summary(results)
    report = ValidationReporter.format_summary(summary)
    
    recommendations = ValidationReporter.generate_recommendations(summary)
    if recommendations:
        report += "\n\n🎯 RECOMMENDATIONS:\n"
        for rec in recommendations:
            report += f"  {rec}\n"
    
    return report


# ============================================================================
# MAIN - TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Z3 CODE VALIDATOR - TEST")
    print("=" * 80)
    
    # Test cases
    test_codes = [
        # Valid code
        """
from z3 import *
x = Int('x')
s = Solver()
s.add(x > 0)
result = s.check()
print(f"Result: {result}")
""",
        # Syntax error
        """
from z3 import *
x = Int('x'
s = Solver()
""",
        # Missing import
        """
x = Int('x')
s = Solver()
s.add(x > 0)
print(s.check())
""",
        # Runtime error (undefined variable)
        """
from z3 import *
s = Solver()
s.add(undefined_var > 0)
print(s.check())
"""
    ]
    
    print("\nRunning validation tests...\n")
    
    results = validate_z3_codes(test_codes, timeout=3)
    
    for i, result in enumerate(results, 1):
        print(f"Test {i}: {'✅ PASSED' if result.passed else '❌ FAILED'}")
        print(f"  Status: {result.status}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
        print(f"  Execution time: {result.execution_time:.3f}s")
        print(f"  Solver created: {result.solver_created}")
        print()
    
    # Generate report
    print(generate_validation_report(results))