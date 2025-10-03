#!/usr/bin/env python3
"""
Interactive Main Interface for Postcondition Generation Pipeline

FIXED VERSION - Enhanced file saving verification and error handling
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.pipeline.pipeline import PostconditionPipeline
from core.models import ProcessingStatus


# ============================================================================
# COLOR CODES FOR TERMINAL OUTPUT
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_banner():
    """Print application banner."""
    banner = f"""
{Colors.HEADER}{'='*80}
    POSTCONDITION GENERATION PIPELINE
    Enhanced with LangChain - Rich Output System
    🔧 FIXED VERSION - Verified File Saving
{'='*80}{Colors.ENDC}
"""
    print(banner)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{title}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*len(title)}{Colors.ENDC}")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.OKBLUE}ℹ️  {message}{Colors.ENDC}")


# ============================================================================
# INPUT FUNCTIONS
# ============================================================================

def get_specification_interactive() -> str:
    """
    Get specification from user interactively.
    
    Returns:
        User's specification string
    """
    print_section("📝 Step 1: Enter Your Specification")
    
    print(f"\n{Colors.BOLD}Enter a natural language specification of what you want to implement.{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Examples:{Colors.ENDC}")
    print("  • Sort an array in ascending order using bubble sort")
    print("  • Search for an element in a binary search tree")
    print("  • Implement a hash table with collision handling")
    print("  • Reverse a linked list in place")
    print("  • Find the shortest path in a weighted graph")
    
    print(f"\n{Colors.BOLD}Your specification (press Enter when done):{Colors.ENDC}")
    print(f"{Colors.OKCYAN}> {Colors.ENDC}", end="")
    
    specification = input().strip()
    
    # Allow multi-line input if first line is too short
    if len(specification) < 10:
        print_warning("Specification seems short. You can continue on multiple lines (empty line to finish):")
        lines = [specification]
        while True:
            print(f"{Colors.OKCYAN}> {Colors.ENDC}", end="")
            line = input().strip()
            if not line:
                break
            lines.append(line)
        specification = " ".join(lines)
    
    return specification


def confirm_specification(specification: str) -> bool:
    """
    Ask user to confirm specification.
    
    Args:
        specification: The specification to confirm
        
    Returns:
        True if confirmed, False otherwise
    """
    print(f"\n{Colors.BOLD}You entered:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  \"{specification}\"{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Proceed with this specification? (y/n):{Colors.ENDC} ", end="")
    response = input().strip().lower()
    
    return response in ['y', 'yes']


def get_output_preference() -> Path:
    """
    Ask user where to save results.
    
    Returns:
        Absolute path to output directory
    """
    print_section("💾 Step 2: Output Location")
    
    # 🔴 FIX: Use absolute path
    default_path = (Path.cwd() / "output" / "pipeline_results").resolve()
    
    print(f"\n{Colors.BOLD}Default output location:{Colors.ENDC} {default_path}")
    print(f"{Colors.BOLD}Use default? (y/n):{Colors.ENDC} ", end="")
    
    response = input().strip().lower()
    
    if response in ['y', 'yes', '']:
        return default_path
    
    print(f"{Colors.BOLD}Enter custom output path:{Colors.ENDC} ", end="")
    custom_path = input().strip()
    
    if custom_path:
        return Path(custom_path).resolve()
    else:
        return default_path


# ============================================================================
# RESULT DISPLAY FUNCTIONS
# ============================================================================

def display_result_summary(result):
    """
    Display a summary of the pipeline result.
    
    Args:
        result: CompleteEnhancedResult object
    """
    print_section("📊 Results Summary")
    
    # Status
    status_color = Colors.OKGREEN if result.status == ProcessingStatus.SUCCESS else Colors.WARNING
    print(f"\n{Colors.BOLD}Status:{Colors.ENDC} {status_color}{result.status.value.upper()}{Colors.ENDC}")
    print(f"{Colors.BOLD}Session ID:{Colors.ENDC} {result.session_id}")
    print(f"{Colors.BOLD}Processing Time:{Colors.ENDC} {result.total_processing_time:.1f}s")
    
    # Generation statistics
    print(f"\n{Colors.BOLD}Generation Statistics:{Colors.ENDC}")
    print(f"  • Functions Created: {result.total_functions}")
    print(f"  • Total Postconditions: {result.total_postconditions}")
    
    # Calculate Z3 stats from function_results
    successful_z3 = sum(fr.z3_translations_count for fr in result.function_results)
    validated_z3 = sum(fr.z3_validations_passed for fr in result.function_results)
    
    print(f"  • Z3 Translations: {successful_z3}/{result.total_z3_translations}")
    print(f"  • Validated Z3 Code: {validated_z3}/{result.total_z3_translations}")
    
    # Quality metrics (if available)
    if result.average_quality_score > 0:
        print(f"\n{Colors.BOLD}Quality Metrics:{Colors.ENDC}")
        print(f"  • Average Quality: {result.average_quality_score:.2f}")
        print(f"  • Average Robustness: {result.average_robustness_score:.2f}")
        if result.z3_validation_success_rate > 0:
            print(f"  • Z3 Validation Rate: {result.z3_validation_success_rate:.1%}")
    
    # Function details
    if result.function_results:
        print(f"\n{Colors.BOLD}Function Details:{Colors.ENDC}")
        for func_result in result.function_results:
            print(f"\n  {Colors.OKCYAN}Function:{Colors.ENDC} {func_result.function_name}")
            print(f"    Signature: {func_result.function_signature}")
            print(f"    Postconditions: {func_result.postcondition_count}")
            
            if func_result.postcondition_count > 0:
                print(f"    Avg Quality: {func_result.average_quality_score:.2f}")
                print(f"    Avg Robustness: {func_result.average_robustness_score:.2f}")
                edge_cases_per_pc = (
                    func_result.total_edge_cases_covered / func_result.postcondition_count 
                    if func_result.postcondition_count > 0 
                    else 0.0
                )
                print(f"    Edge Cases/PC: {edge_cases_per_pc:.1f}")
                print(f"    Z3 Validated: {func_result.z3_validations_passed}/{func_result.postcondition_count}")


def display_sample_postconditions(result, max_samples: int = 2):
    """
    Display sample postconditions from the result.
    
    Args:
        result: CompleteEnhancedResult object
        max_samples: Maximum number of samples to show
    """
    print_section("📋 Sample Postconditions")
    
    if not result.function_results:
        print(f"{Colors.WARNING}No postconditions generated.{Colors.ENDC}")
        return
    
    shown = 0
    for func_result in result.function_results:
        if shown >= max_samples:
            break
        
        if not func_result.postconditions:
            continue
        
        print(f"\n{Colors.OKCYAN}Function:{Colors.ENDC} {func_result.function_name}")
        
        for i, pc in enumerate(func_result.postconditions[:2]):  # Show first 2 per function
            if shown >= max_samples:
                break
            
            print(f"\n  {Colors.BOLD}Postcondition {i+1}:{Colors.ENDC}")
            print(f"    Formal: {pc.formal_text}")
            print(f"    Natural: {pc.natural_language}")
            
            # Check for optional enhanced fields
            if pc.precise_translation:
                print(f"    Translation: {pc.precise_translation[:100]}...")
            
            if pc.reasoning:
                print(f"    Reasoning: {pc.reasoning[:100]}...")
            
            # Use property for overall quality score
            print(f"    Quality: {pc.overall_quality_score:.2f}")
            
            if pc.z3_theory and pc.z3_theory != "unknown":
                print(f"    Z3 Theory: {pc.z3_theory}")
            
            shown += 1
    
    if result.total_postconditions > max_samples:
        remaining = result.total_postconditions - max_samples
        print(f"\n{Colors.OKBLUE}... and {remaining} more postconditions{Colors.ENDC}")


def display_errors_warnings(result):
    """
    Display any errors or warnings from the result.
    
    Args:
        result: CompleteEnhancedResult object
    """
    if result.errors:
        print_section("⚠️  Errors")
        for error in result.errors:
            print_error(error)
    
    if result.warnings:
        print_section("ℹ️  Warnings")
        for warning in result.warnings:
            print_warning(warning)


def display_saved_location(saved_path: Path):
    """
    Display where results were saved with verification.
    
    Args:
        saved_path: Path to saved results directory
    """
    print(f"\n✅ Results saved to: {Colors.BOLD}{saved_path}{Colors.ENDC}")
    
    # 🔴 FIX: Verify files actually exist
    if saved_path.exists():
        files = list(saved_path.rglob("*"))
        files = [f for f in files if f.is_file()]  # Only show files, not directories
        
        if files:
            print(f"\n📁 Created {len(files)} files:")
            for file in sorted(files):
                relative = file.relative_to(saved_path)
                size = file.stat().st_size
                print(f"   📄 {relative} ({size:,} bytes)")
        else:
            print_error(f"No files found in {saved_path}!")
            print(f"   Directory exists but is empty")
    else:
        print_error(f"Output directory does not exist: {saved_path}")


# ============================================================================
# MAIN PIPELINE EXECUTION (🔴 FIXED VERSION)
# ============================================================================

def run_pipeline(specification: str, output_dir: Path, codebase_path: Optional[Path] = None):
    """
    Run the postcondition generation pipeline.
    
    🔴 FIXED: Enhanced file saving verification
    
    Args:
        specification: Natural language specification
        output_dir: Where to save results (absolute path)
        codebase_path: Optional path to existing codebase
        
    Returns:
        CompleteEnhancedResult or None if failed
    """
    print_section("⚡ Step 3: Generating Postconditions")
    
    # 🔴 FIX 1: Verify output directory is absolute
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
        print_info(f"Using absolute path: {output_dir}")
    
    # 🔴 FIX 2: Create output directory immediately
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Output directory ready: {output_dir}")
    except Exception as e:
        print_error(f"Failed to create output directory: {e}")
        return None
    
    try:
        # Initialize pipeline
        print_info("Initializing pipeline...")
        pipeline = PostconditionPipeline(codebase_path=codebase_path)
        
        # Process specification
        print_info("Processing specification...")
        result = pipeline.process_sync(specification)
        
        # 🔴 FIX 3: Explicit save with verification
        print_info("Saving results...")
        
        try:
            pipeline.save_results(result, output_dir)
            
            # 🔴 FIX 4: Verify files were actually created
            created_files = list(output_dir.rglob("*"))
            created_files = [f for f in created_files if f.is_file()]
            
            if created_files:
                print_success(f"Created {len(created_files)} files")
                for file in created_files:
                    print_info(f"  ✓ {file.name} ({file.stat().st_size:,} bytes)")
            else:
                print_error(f"No files created in {output_dir}!")
                print_warning("This may indicate a file saving issue")
                
                # Try to save manually as fallback
                print_info("Attempting manual save as fallback...")
                manual_save_fallback(result, output_dir)
        
        except Exception as save_error:
            print_error(f"Save failed: {save_error}")
            import traceback
            traceback.print_exc()
            
            # Try manual fallback
            print_info("Attempting manual save as fallback...")
            manual_save_fallback(result, output_dir)
        
        # Display results
        print_success("Pipeline completed!")
        display_result_summary(result)
        display_sample_postconditions(result)
        display_errors_warnings(result)
        display_saved_location(output_dir)
        
        return result
    
    except Exception as e:
        print_error(f"Pipeline failed: {str(e)}")
        import traceback
        print(f"\n{Colors.FAIL}Traceback:{Colors.ENDC}")
        traceback.print_exc()
        return None


def manual_save_fallback(result, output_dir: Path):
    """
    Fallback manual save if pipeline save_results fails.
    
    🔴 NEW: Emergency backup save method
    """
    try:
        # Save JSON result
        json_path = output_dir / "complete_result.json"
        print_info(f"Writing to: {json_path}")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(result.model_dump_json(indent=2))
        
        if json_path.exists():
            print_success(f"Saved: {json_path} ({json_path.stat().st_size:,} bytes)")
        else:
            print_error(f"Failed to create: {json_path}")
        
        # Save simple summary
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("POSTCONDITION GENERATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Session ID: {result.session_id}\n")
            f.write(f"Specification: {result.specification}\n")
            f.write(f"Status: {result.status.value}\n")
            f.write(f"Functions: {result.total_functions}\n")
            f.write(f"Postconditions: {result.total_postconditions}\n")
            f.write(f"Z3 Translations: {result.total_z3_translations}\n")
        
        if summary_path.exists():
            print_success(f"Saved: {summary_path}")
        
    except Exception as e:
        print_error(f"Manual fallback save also failed: {e}")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Run in interactive mode with user prompts."""
    print_banner()
    
    # Get specification
    while True:
        specification = get_specification_interactive()
        
        if not specification:
            print_error("Specification cannot be empty!")
            continue
        
        if confirm_specification(specification):
            break
        
        print_warning("Let's try again...")
    
    # Get output location
    output_dir = get_output_preference()
    
    # 🔴 FIX: Show absolute path
    print_info(f"Saving to absolute path: {output_dir}")
    
    # Run pipeline
    result = run_pipeline(specification, output_dir)
    
    # Final message
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    if result and result.status == ProcessingStatus.SUCCESS:
        print_success("All done! Check the output files for detailed results.")
        print_info(f"Output location: {output_dir}")
    else:
        print_warning("Pipeline completed with issues. Check errors above.")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")


# ============================================================================
# COMMAND-LINE MODE
# ============================================================================

def command_line_mode(args):
    """Run in command-line mode with arguments."""
    print_banner()
    
    specification = args.spec
    output_dir = Path(args.output).resolve() if args.output else (Path.cwd() / "output" / "pipeline_results").resolve()
    codebase_path = Path(args.codebase).resolve() if args.codebase else None
    
    print(f"{Colors.BOLD}Specification:{Colors.ENDC} {specification}")
    print(f"{Colors.BOLD}Output Directory:{Colors.ENDC} {output_dir}")
    if codebase_path:
        print(f"{Colors.BOLD}Codebase Path:{Colors.ENDC} {codebase_path}")
    
    result = run_pipeline(specification, output_dir, codebase_path)
    
    # Exit with appropriate code
    if result and result.status == ProcessingStatus.SUCCESS:
        sys.exit(0)
    else:
        sys.exit(1)


# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Postcondition Generation Pipeline - Interactive Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py
  
  # Direct mode with specification
  python main.py --spec "Sort an array using bubble sort"
  
  # With custom output directory
  python main.py --spec "Binary search" --output results/
  
  # With existing codebase context
  python main.py --spec "Add function" --codebase /path/to/code/
        """
    )
    
    parser.add_argument(
        '--spec',
        type=str,
        help='Natural language specification (if omitted, runs in interactive mode)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results (default: output/pipeline_results)'
    )
    
    parser.add_argument(
        '--codebase', '-c',
        type=str,
        help='Path to existing codebase for context (optional)'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Postcondition Pipeline v2.0.0 (Fixed)'
    )
    
    return parser.parse_args()


# ============================================================================
# STARTUP CHECKS
# ============================================================================

def check_environment():
    """
    Check if the environment is properly configured.
    
    Returns:
        True if ready, False otherwise
    """
    issues = []
    
    # Check for .env file
    env_path = project_root / ".env"
    if not env_path.exists():
        issues.append("Missing .env file with OPENAI_API_KEY")
    
    # Check for OpenAI API key
    try:
        from config.settings import settings
        if not settings.openai_api_key:
            issues.append("OPENAI_API_KEY not set in .env")
        elif not settings.openai_api_key.startswith("sk-"):
            issues.append("OPENAI_API_KEY appears invalid (should start with 'sk-')")
    except Exception as e:
        issues.append(f"Failed to load settings: {e}")
    
    # Check for required modules
    required_modules = [
        'langchain',
        'langchain_openai',
        'openai',
        'pydantic'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            issues.append(f"Missing required module: {module}")
    
    # 🔴 FIX: Check write permissions
    test_dir = Path.cwd() / "output" / "test_write"
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "test.txt"
        test_file.write_text("test")
        test_file.unlink()
        test_dir.rmdir()
    except Exception as e:
        issues.append(f"Cannot write to output directory: {e}")
    
    # Display results
    if issues:
        print_error("Environment check failed!")
        print(f"\n{Colors.FAIL}Issues found:{Colors.ENDC}")
        for issue in issues:
            print(f"  ❌ {issue}")
        
        print(f"\n{Colors.BOLD}To fix:{Colors.ENDC}")
        print("1. Create .env file with: OPENAI_API_KEY=sk-your-key-here")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Ensure write permissions in current directory")
        print()
        return False
    
    return True


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    # First, check environment
    print(f"{Colors.OKBLUE}Checking environment...{Colors.ENDC}")
    if not check_environment():
        sys.exit(1)
    print_success("Environment OK")
    
    try:
        args = parse_arguments()
        
        # Check if running in interactive or command-line mode
        if args.spec:
            # Command-line mode: specification provided
            command_line_mode(args)
        else:
            # Interactive mode: no specification provided
            interactive_mode()
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user.{Colors.ENDC}")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n{Colors.FAIL}Fatal Error: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Ensure errors are visible
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Run with error catching
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)