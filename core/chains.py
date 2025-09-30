"""
LangChain chains for postcondition generation system.

This module replaces scattered OpenAI API calls with reusable, composable chains.
Each chain handles one specific task with proper error handling, retries, and caching.

Replaces:
- Direct openai.ChatCompletion.create() calls throughout codebase
- Manual prompt construction in pseudocode.py, logic_generator.py, etc.
- Redundant LLM initialization and configuration
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, 
    PromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.cache import SQLiteCache
import langchain

from typing import List, Dict, Any, Optional
from pathlib import Path

from config.settings import settings
from core.models import (
    Function,
    PseudocodeResult,
    EnhancedPostcondition,
    PostconditionStrength,
    Z3Translation,
    FunctionParameter
)


# ============================================================================
# CACHING SETUP
# ============================================================================

if settings.enable_cache:
    langchain.llm_cache = SQLiteCache(database_path=str(settings.llm_cache_db))


# ============================================================================
# BASE LLM FACTORY
# ============================================================================

class LLMFactory:
    """
    Factory for creating configured LLM instances.
    Centralizes all LLM configuration in one place.
    """
    
    @staticmethod
    def create_llm(
        temperature: Optional[float] = None,
        streaming: bool = False,
        callbacks: Optional[list] = None
    ) -> ChatOpenAI:
        """
        Create a configured ChatOpenAI instance.
        
        Args:
            temperature: Override default temperature
            streaming: Enable streaming responses
            callbacks: Custom callbacks (auto-adds streaming if enabled)
            
        Returns:
            Configured ChatOpenAI instance
        """
        if streaming and callbacks is None:
            callbacks = [StreamingStdOutCallbackHandler()]
        
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature or settings.temperature,
            max_tokens=settings.max_tokens,
            openai_api_key=settings.openai_api_key,
            streaming=streaming,
            callbacks=callbacks,
            max_retries=settings.max_retries,
            request_timeout=settings.request_timeout
        )
    
    @staticmethod
    def create_embeddings() -> OpenAIEmbeddings:
        """Create OpenAI embeddings instance."""
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )


# ============================================================================
# PSEUDOCODE GENERATION CHAIN
# ============================================================================

class PseudocodeChain:
    """
    Chain for generating C pseudocode from specifications.
    
    Replaces: The 1000+ lines in pseudocode.py that manually call OpenAI
    and parse responses.
    """
    
    def __init__(self, streaming: bool = True):
        self.llm = LLMFactory.create_llm(streaming=streaming)
        self.parser = PydanticOutputParser(pydantic_object=PseudocodeResult)
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create the pseudocode generation chain."""
        
        system_template = """You are an expert C programmer who generates structured pseudocode with ZERO ambiguity.

CRITICAL REQUIREMENTS:

1. **TYPE SYSTEM**: Use complete C types directly - no separate flags
   - "int*" for pointer to int
   - "int**" for pointer to pointer to int
   - "char[]" for character array
   - "struct Node*" for pointer to struct

2. **FUNCTION PARAMETERS**: 
   - Input parameters: function arguments
   - Output parameters: parameters modified by reference (pointers)
   - ALL parameters must have clear, unambiguous types

3. **RETURN VALUES**:
   - List ALL possible return values with conditions
   - Example: {{"condition": "success", "value": "0", "description": "Operation successful"}}

4. **COMPLEXITY**:
   - Time complexity in Big-O notation (e.g., "O(n)", "O(n log n)")
   - Space complexity separately

5. **EDGE CASES**: List ALL edge cases that must be handled
   - Null pointers
   - Empty arrays
   - Boundary conditions
   - Integer overflow
   - Memory allocation failures

6. **DEPENDENCIES**: 
   - List functions called with source ("stdlib", "codebase", "generated")
   - Include header files needed

{format_instructions}

Generate complete, production-ready pseudocode."""

        human_template = """Specification: {specification}

{context}

Generate complete C pseudocode following ALL requirements above."""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.parser,
            verbose=settings.verbose
        )
    
    def generate(
        self, 
        specification: str,
        codebase_context: Optional[Dict[str, Any]] = None
    ) -> PseudocodeResult:
        """
        Generate pseudocode from specification.
        
        Args:
            specification: The function specification
            codebase_context: Optional context from existing codebase
            
        Returns:
            PseudocodeResult with generated functions, structs, etc.
            
        Example:
            >>> chain = PseudocodeChain()
            >>> result = chain.generate("Sort an array using bubble sort")
            >>> print(result.functions[0].name)
            'bubble_sort'
        """
        context = ""
        if codebase_context:
            context = f"""
Available functions from codebase:
{self._format_codebase_context(codebase_context)}
"""
        
        return self.chain.invoke({
            "specification": specification,
            "context": context,
            "format_instructions": self.parser.get_format_instructions()
        })
    
    async def agenerate(
        self,
        specification: str,
        codebase_context: Optional[Dict[str, Any]] = None
    ) -> PseudocodeResult:
        """Async version of generate()."""
        context = ""
        if codebase_context:
            context = f"Available functions: {codebase_context}"
        
        return await self.chain.ainvoke({
            "specification": specification,
            "context": context,
            "format_instructions": self.parser.get_format_instructions()
        })
    
    def _format_codebase_context(self, context: Dict[str, Any]) -> str:
        """Format codebase context for prompt."""
        lines = []
        for func_name, func_info in context.items():
            lines.append(f"- {func_name}: {func_info.get('description', 'No description')}")
        return "\n".join(lines)


# ============================================================================
# POSTCONDITION GENERATION CHAIN
# ============================================================================

class PostconditionChain:
    """
    Chain for generating formal postconditions from functions.
    
    Replaces: The 3000+ lines in logic_generator.py that manually build
    prompts and parse responses.
    """
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming)
        self.parser = JsonOutputParser()
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create the postcondition generation chain."""
        
        system_template = """You are an expert in formal verification and postcondition generation.

Your task is to generate comprehensive, mathematically precise postconditions for C functions.

POSTCONDITION STRUCTURE:
{{
    "formal_text": "Mathematical logic using ∀, ∃, →, ∧, ∨",
    "natural_language": "Clear English explanation",
    "strength": "minimal|standard|comprehensive",
    "category": "return_value|state_change|side_effect|error_condition|memory|correctness",
    "confidence_score": 0.0-1.0,
    "edge_cases": ["List of edge cases this covers"],
    "z3_theory": "arrays|bitvectors|datatypes|arithmetic|etc"
}}

CRITICAL REQUIREMENTS:

1. **FORMAL TEXT**: Use precise mathematical notation
   - ∀ (for all), ∃ (exists)
   - → (implies), ∧ (and), ∨ (or), ¬ (not)
   - Array notation: arr[i], size(arr)
   - Ranges: 0 ≤ i < n

2. **CATEGORIES**:
   - return_value: What the function returns
   - state_change: How parameters/state are modified
   - side_effect: External effects (I/O, global state)
   - error_condition: Error handling guarantees
   - memory: Memory safety (no leaks, valid pointers)
   - correctness: Algorithm correctness properties

3. **STRENGTHS**:
   - minimal: Basic guarantees only
   - standard: Normal comprehensive postconditions
   - comprehensive: Exhaustive coverage including edge cases

4. **Z3 THEORIES**:
   - arrays: For array operations
   - bitvectors: For bit manipulation
   - datatypes: For structs/enums
   - arithmetic: For math operations
   - strings: For string operations

5. **EDGE CASES**: Consider:
   - Null pointers
   - Empty/single-element arrays
   - Boundary conditions (0, max, min values)
   - Integer overflow/underflow
   - Memory allocation failures
   - Concurrent access (if applicable)

Generate MULTIPLE postconditions covering different aspects."""

        human_template = """Function to analyze:

Name: {function_name}
Signature: {function_signature}
Description: {function_description}

Parameters:
{parameters}

Return Type: {return_type}

Specification: {specification}

Known Edge Cases:
{edge_cases}

Generate comprehensive postconditions as a JSON array."""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.parser,
            verbose=settings.verbose
        )
    
    def generate(
        self,
        function: Function,
        specification: str,
        edge_cases: Optional[List[str]] = None
    ) -> List[EnhancedPostcondition]:
        """
        Generate postconditions for a function.
        
        Args:
            function: The Function model
            specification: Original specification
            edge_cases: Known edge cases to consider
            
        Returns:
            List of EnhancedPostcondition objects
            
        Example:
            >>> chain = PostconditionChain()
            >>> postconditions = chain.generate(bubble_sort_func, "Sort array")
            >>> print(postconditions[0].formal_text)
            '∀i,j: 0 ≤ i < j < size → arr[i] ≤ arr[j]'
        """
        parameters_str = "\n".join([
            f"- {p.name}: {p.data_type} - {p.description}"
            for p in function.input_parameters
        ])
        
        edge_cases_str = "\n".join(edge_cases or function.edge_cases or [
            "Empty input",
            "Null pointers",
            "Boundary values"
        ])
        
        result = self.chain.invoke({
            "function_name": function.name,
            "function_signature": function.signature,
            "function_description": function.description,
            "parameters": parameters_str,
            "return_type": function.return_type,
            "specification": specification,
            "edge_cases": edge_cases_str
        })
        
        # Parse JSON response into EnhancedPostcondition objects
        return self._parse_postconditions(result)
    
    async def agenerate(
        self,
        function: Function,
        specification: str,
        edge_cases: Optional[List[str]] = None
    ) -> List[EnhancedPostcondition]:
        """Async version of generate()."""
        parameters_str = "\n".join([
            f"- {p.name}: {p.data_type} - {p.description}"
            for p in function.input_parameters
        ])
        
        edge_cases_str = "\n".join(edge_cases or function.edge_cases or [])
        
        result = await self.chain.ainvoke({
            "function_name": function.name,
            "function_signature": function.signature,
            "function_description": function.description,
            "parameters": parameters_str,
            "return_type": function.return_type,
            "specification": specification,
            "edge_cases": edge_cases_str
        })
        
        return self._parse_postconditions(result)
    
    def _parse_postconditions(self, result: Any) -> List[EnhancedPostcondition]:
        """Parse LLM output into EnhancedPostcondition objects."""
        postconditions = []
        
        # Result is already parsed by JsonOutputParser
        if isinstance(result, dict) and 'text' in result:
            result = result['text']
        
        if isinstance(result, str):
            import json
            result = json.loads(result)
        
        if not isinstance(result, list):
            result = [result]
        
        for pc_data in result:
            try:
                postconditions.append(EnhancedPostcondition(**pc_data))
            except Exception as e:
                print(f"Warning: Failed to parse postcondition: {e}")
                continue
        
        return postconditions


# ============================================================================
# Z3 TRANSLATION CHAIN
# ============================================================================

class Z3TranslationChain:
    """
    Chain for translating formal postconditions to Z3 code.
    
    Replaces: The 2000+ lines in logic2postcondition.py that manually
    construct Z3 code.
    """
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming, temperature=0.1)
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create the Z3 translation chain."""
        
        system_template = """You are an expert in Z3 theorem prover and SMT solving.

Translate formal logic postconditions into executable Z3 Python code.

Z3 CODE REQUIREMENTS:

1. **IMPORTS**: Always start with `from z3 import *`

2. **VARIABLE DECLARATIONS**:
   - Use appropriate Z3 sorts: Int(), Bool(), Array(IntSort(), IntSort())
   - Declare ALL variables used in the postcondition

3. **CONSTRAINTS**:
   - Translate logical formulas to Z3 syntax
   - ∀x: P(x) → ForAll([x], P(x))
   - ∃x: P(x) → Exists([x], P(x))
   - a ∧ b → And(a, b)
   - a ∨ b → Or(a, b)
   - a → b → Implies(a, b)

4. **ARRAYS**:
   - Use Array(IntSort(), IntSort()) for integer arrays
   - Access: Select(arr, i)
   - Update: Store(arr, i, val)

5. **QUANTIFIERS**:
   - Bind variables properly: [x], [x, y]
   - Specify types for quantified variables

6. **SOLVER**:
   - Create solver: s = Solver()
   - Add constraints: s.add(constraint)
   - Check: result = s.check()
   - Assert expected result with explanation

7. **CODE STRUCTURE**:
```python
from z3 import *

# Declare variables
x = Int('x')
arr = Array('arr', IntSort(), IntSort())
size = Int('size')

# Define constraints
constraint = ForAll([x], 
    Implies(And(x >= 0, x < size),
        Select(arr, x) <= Select(arr, x + 1)))

# Create solver and add constraints
s = Solver()
s.add(constraint)
s.add(size > 0)  # Preconditions

# Check satisfiability
result = s.check()
assert result == sat, "Postcondition should be satisfiable"

print(f"Verification result: {{result}}")
if result == sat:
    print("Model:", s.model())
```

Generate ONLY executable Z3 Python code. No explanations outside code comments."""

        human_template = """Formal Postcondition:
{formal_text}

Natural Language:
{natural_language}

Function Context:
{function_context}

Z3 Theory to Use: {z3_theory}

Generate executable Z3 Python code to verify this postcondition."""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=settings.verbose
        )
    
    def translate(
        self,
        postcondition: EnhancedPostcondition,
        function_context: Optional[Dict[str, Any]] = None
    ) -> Z3Translation:
        """
        Translate postcondition to Z3 code.
        
        Args:
            postcondition: The postcondition to translate
            function_context: Context about the function
            
        Returns:
            Z3Translation with generated code and validation status
            
        Example:
            >>> chain = Z3TranslationChain()
            >>> translation = chain.translate(postcondition)
            >>> print(translation.z3_code)
            'from z3 import *\n...'
        """
        result = self.chain.invoke({
            "formal_text": postcondition.formal_text,
            "natural_language": postcondition.natural_language,
            "function_context": self._format_function_context(function_context),
            "z3_theory": postcondition.z3_theory or "arithmetic"
        })
        
        z3_code = self._extract_code(result)
        
        # Create Z3Translation object
        translation = Z3Translation(
            formal_text=postcondition.formal_text,
            natural_language=postcondition.natural_language,
            z3_code=z3_code,
            z3_theory_used=postcondition.z3_theory or "arithmetic",
            translation_success=bool(z3_code)
        )
        
        # Validate the generated code
        self._validate_z3_code(translation)
        
        return translation
    
    async def atranslate(
        self,
        postcondition: EnhancedPostcondition,
        function_context: Optional[Dict[str, Any]] = None
    ) -> Z3Translation:
        """Async version of translate()."""
        result = await self.chain.ainvoke({
            "formal_text": postcondition.formal_text,
            "natural_language": postcondition.natural_language,
            "function_context": self._format_function_context(function_context),
            "z3_theory": postcondition.z3_theory or "arithmetic"
        })
        
        z3_code = self._extract_code(result)
        
        translation = Z3Translation(
            formal_text=postcondition.formal_text,
            natural_language=postcondition.natural_language,
            z3_code=z3_code,
            z3_theory_used=postcondition.z3_theory or "arithmetic",
            translation_success=bool(z3_code)
        )
        
        self._validate_z3_code(translation)
        
        return translation
    
    def _extract_code(self, result: Any) -> str:
        """Extract Z3 code from LLM response."""
        if isinstance(result, dict):
            code = result.get('text', '')
        else:
            code = str(result)
        
        # Extract code from markdown code blocks if present
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            code = code.split('```')[1].split('```')[0]
        
        return code.strip()
    
    def _validate_z3_code(self, translation: Z3Translation) -> None:
        """Validate Z3 code syntax and update translation object."""
        import ast
        
        if not translation.z3_code:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "not_validated"
            return
        
        try:
            # Check syntax
            ast.parse(translation.z3_code)
            
            # Basic checks
            if 'from z3 import' not in translation.z3_code:
                translation.warnings.append("Missing Z3 import statement")
            
            if 'Solver()' not in translation.z3_code:
                translation.warnings.append("No Solver() instance created")
            
            translation.z3_validation_passed = True
            translation.z3_validation_status = "success"
            
        except SyntaxError as e:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "syntax_error"
            translation.validation_error = str(e)
        except Exception as e:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "runtime_error"
            translation.validation_error = str(e)
    
    def _format_function_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format function context for prompt."""
        if not context:
            return "No additional context"
        
        lines = []
        if 'parameters' in context:
            lines.append("Parameters:")
            for param in context['parameters']:
                lines.append(f"  - {param.get('name')}: {param.get('data_type')}")
        
        if 'return_type' in context:
            lines.append(f"Return Type: {context['return_type']}")
        
        return "\n".join(lines) if lines else "No additional context"


# ============================================================================
# EDGE CASE ANALYSIS CHAIN
# ============================================================================

class EdgeCaseChain:
    """
    Chain for analyzing edge cases in specifications.
    
    Replaces: Edge case analysis code scattered throughout logic_generator.py
    """
    
    def __init__(self):
        self.llm = LLMFactory.create_llm()
        self.parser = JsonOutputParser()
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create the edge case analysis chain."""
        
        system_template = """You are an expert at identifying edge cases in software specifications.

Analyze the specification and identify ALL edge cases that must be considered.

EDGE CASE CATEGORIES:

1. **Input Validation**:
   - Null/undefined inputs
   - Empty collections
   - Single element cases
   - Invalid types or formats

2. **Boundary Conditions**:
   - Minimum/maximum values
   - Zero values
   - Negative values
   - Integer overflow/underflow

3. **Resource Constraints**:
   - Memory allocation failures
   - Stack overflow
   - File system errors
   - Network failures

4. **Concurrency**:
   - Race conditions
   - Deadlocks
   - Thread safety

5. **Special Values**:
   - NaN, Infinity (for floats)
   - MIN_INT, MAX_INT
   - Empty strings
   - Special characters

Return JSON array of edge cases:
[
  {{
    "category": "input_validation",
    "description": "Null pointer passed as array",
    "severity": "critical|high|medium|low",
    "test_case": "Example test case"
  }}
]"""

        human_template = """Specification: {specification}

Function: {function_description}

Parameters:
{parameters}

Identify ALL edge cases."""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.parser,
            verbose=settings.verbose
        )
    
    def analyze(
        self,
        specification: str,
        function: Function
    ) -> List[str]:
        """
        Analyze edge cases for a function.
        
        Args:
            specification: Original specification
            function: Function model
            
        Returns:
            List of edge case descriptions
        """
        parameters_str = "\n".join([
            f"- {p.name}: {p.data_type}"
            for p in function.input_parameters
        ])
        
        result = self.chain.invoke({
            "specification": specification,
            "function_description": function.description,
            "parameters": parameters_str
        })
        
        # Parse result
        if isinstance(result, dict) and 'text' in result:
            result = result['text']
        
        if isinstance(result, str):
            import json
            result = json.loads(result)
        
        if not isinstance(result, list):
            result = [result]
        
        return [ec.get('description', str(ec)) for ec in result]


# ============================================================================
# CHAIN FACTORY - Main Entry Point
# ============================================================================

class ChainFactory:
    """
    Factory for creating and managing all chains.
    
    Use this as the main entry point for accessing chains throughout your codebase.
    
    Example:
        >>> factory = ChainFactory()
        >>> pseudocode = factory.pseudocode.generate("sort an array")
        >>> postconditions = factory.postcondition.generate(function, spec)
        >>> z3_code = factory.z3.translate(postcondition)
    """
    
    def __init__(self):
        self._pseudocode_chain = None
        self._postcondition_chain = None
        self._z3_chain = None
        self._edge_case_chain = None
    
    @property
    def pseudocode(self) -> PseudocodeChain:
        """Get or create pseudocode generation chain."""
        if self._pseudocode_chain is None:
            self._pseudocode_chain = PseudocodeChain()
        return self._pseudocode_chain
    
    @property
    def postcondition(self) -> PostconditionChain:
        """Get or create postcondition generation chain."""
        if self._postcondition_chain is None:
            self._postcondition_chain = PostconditionChain()
        return self._postcondition_chain
    
    @property
    def z3(self) -> Z3TranslationChain:
        """Get or create Z3 translation chain."""
        if self._z3_chain is None:
            self._z3_chain = Z3TranslationChain()
        return self._z3_chain
    
    @property
    def edge_case(self) -> EdgeCaseChain:
        """Get or create edge case analysis chain."""
        if self._edge_case_chain is None:
            self._edge_case_chain = EdgeCaseChain()
        return self._edge_case_chain


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Generate pseudocode
    print("=" * 70)
    print("EXAMPLE 1: Pseudocode Generation")
    print("=" * 70)
    
    factory = ChainFactory()
    
    pseudocode_result = factory.pseudocode.generate(
        specification="Sort an array using bubble sort algorithm"
    )
    
    print(f"Generated {len(pseudocode_result.functions)} function(s)")
    if pseudocode_result.functions:
        func = pseudocode_result.functions[0]
        print(f"Function: {func.signature}")
        print(f"Complexity: {func.complexity}")
    
    # Example 2: Generate postconditions
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Postcondition Generation")
    print("=" * 70)
    
    # Create a sample function
    test_function = Function(
        name="bubble_sort",
        description="Sort array using bubble sort",
        return_type="void",
        input_parameters=[
            FunctionParameter(name="arr", data_type="int[]"),
            FunctionParameter(name="size", data_type="int")
        ]
    )
    
    postconditions = factory.postcondition.generate(
        function=test_function,
        specification="Sort array in ascending order"
    )
    
    print(f"Generated {len(postconditions)} postcondition(s)")
    for i, pc in enumerate(postconditions[:2]):  # Show first 2
        print(f"\n{i+1}. {pc.natural_language}")
        print(f"   Formal: {pc.formal_text}")
        print(f"   Confidence: {pc.confidence_score}")
    
    # Example 3: Translate to Z3
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Z3 Translation")
    print("=" * 70)
    
    if postconditions:
        translation = factory.z3.translate(postconditions[0])
        print(f"Translation successful: {translation.translation_success}")
        print(f"Validation passed: {translation.z3_validation_passed}")
        if translation.z3_code:
            print("\nGenerated Z3 code (first 200 chars):")
            print(translation.z3_code[:200] + "...")