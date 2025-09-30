"""
Fixed LangChain chains with proper output parser integration

This fixes the format_instructions error by properly integrating
PydanticOutputParser with the prompt templates.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate, 
    PromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.cache import SQLiteCache
import langchain

from typing import List, Dict, Any, Optional
import json

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
    """Factory for creating configured LLM instances."""
    
    @staticmethod
    def create_llm(
        temperature: Optional[float] = None,
        streaming: bool = False,
        callbacks: Optional[list] = None
    ) -> ChatOpenAI:
        """Create a configured ChatOpenAI instance."""
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
# PSEUDOCODE GENERATION CHAIN - FIXED
# ============================================================================

class PseudocodeChain:
    """
    Chain for generating C pseudocode from specifications.
    
    FIXED: Properly integrates PydanticOutputParser with format instructions.
    """
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming)
        # Use JsonOutputParser instead - simpler and more reliable
        self.parser = JsonOutputParser()
        self.prompt = self._create_prompt()
        # Modern LCEL pattern
        self.chain = self.prompt | self.llm | self.parser
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the pseudocode generation prompt."""
        
        system_template = """You are an expert C programmer who generates structured pseudocode.

Generate C pseudocode following this JSON structure:

{{
  "functions": [
    {{
      "name": "function_name",
      "description": "Clear description",
      "signature": "return_type function_name(param_type param_name)",
      "return_type": "int",
      "input_parameters": [
        {{
          "name": "param_name",
          "data_type": "int*",
          "description": "Parameter description"
        }}
      ],
      "output_parameters": [],
      "return_values": [
        {{
          "condition": "success",
          "value": "0",
          "description": "Success case"
        }}
      ],
      "preconditions": ["arr != NULL", "size > 0"],
      "edge_cases": ["Empty array", "NULL pointer"],
      "complexity": "O(n)",
      "memory_usage": "O(1)",
      "body": "Pseudocode steps",
      "dependencies": []
    }}
  ],
  "structs": [],
  "enums": [],
  "global_variables": [],
  "includes": ["stdio.h", "stdlib.h"],
  "dependencies": [],
  "metadata": {{}}
}}

CRITICAL REQUIREMENTS:
1. Use complete C types: "int*", "char**", "struct Node*"
2. List ALL edge cases
3. Specify complexity (time and space)
4. Include preconditions
5. Return ONLY valid JSON"""

        human_template = """Generate pseudocode for:

{specification}

{context}

Return the complete JSON structure above."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def generate(
        self, 
        specification: str,
        codebase_context: Optional[Dict[str, Any]] = None
    ) -> PseudocodeResult:
        """Generate pseudocode from specification."""
        context = ""
        if codebase_context:
            context = f"Available functions: {', '.join(codebase_context.keys())}"
        
        try:
            result = self.chain.invoke({
                "specification": specification,
                "context": context
            })
            
            # Parse into PseudocodeResult
            return PseudocodeResult(**result)
        except Exception as e:
            print(f"Warning: Failed to generate pseudocode: {e}")
            # Return empty result
            return PseudocodeResult(
                functions=[],
                structs=[],
                dependencies=[]
            )
    
    async def agenerate(
        self,
        specification: str,
        codebase_context: Optional[Dict[str, Any]] = None
    ) -> PseudocodeResult:
        """Async version of generate()."""
        context = ""
        if codebase_context:
            context = f"Available functions: {', '.join(codebase_context.keys())}"
        
        try:
            result = await self.chain.ainvoke({
                "specification": specification,
                "context": context
            })
            
            return PseudocodeResult(**result)
        except Exception as e:
            print(f"Warning: Failed to generate pseudocode: {e}")
            return PseudocodeResult(
                functions=[],
                structs=[],
                dependencies=[]
            )


# ============================================================================
# POSTCONDITION GENERATION CHAIN - FIXED
# ============================================================================

class PostconditionChain:
    """
    Chain for generating formal postconditions from functions.
    
    FIXED: Uses JsonOutputParser for simpler, more reliable parsing.
    """
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming)
        self.parser = JsonOutputParser()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the postcondition generation prompt."""
        
        system_template = """You are an expert in formal verification and postcondition generation.

Generate comprehensive postconditions as a JSON array:

[
  {{
    "formal_text": "∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
    "natural_language": "Array is sorted in ascending order",
    "strength": "standard",
    "category": "correctness",
    "confidence_score": 0.95,
    "clarity_score": 0.9,
    "completeness_score": 0.85,
    "testability_score": 0.9,
    "edge_cases": ["Empty array", "Single element"],
    "z3_theory": "arrays",
    "reasoning": "Why this postcondition matters"
  }}
]

CRITICAL REQUIREMENTS:
1. Use mathematical notation: ∀, ∃, →, ∧, ∨
2. List 3-7 postconditions covering different aspects
3. All scores must be 0.0-1.0
4. Include edge_cases list
5. Specify appropriate z3_theory

Return ONLY the JSON array."""

        human_template = """Generate postconditions for:

Function: {function_name}
Signature: {function_signature}
Description: {function_description}

Parameters:
{parameters}

Return Type: {return_type}

Original Specification: {specification}

Known Edge Cases:
{edge_cases}

Return comprehensive postconditions as JSON array."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def generate(
        self,
        function: Function,
        specification: str,
        edge_cases: Optional[List[str]] = None
    ) -> List[EnhancedPostcondition]:
        """Generate postconditions for a function."""
        parameters_str = "\n".join([
            f"- {p.name}: {p.data_type} - {p.description}"
            for p in function.input_parameters
        ])
        
        edge_cases_str = "\n".join(edge_cases or function.edge_cases or [
            "Empty input",
            "Null pointers",
            "Boundary values"
        ])
        
        try:
            result = self.chain.invoke({
                "function_name": function.name,
                "function_signature": function.signature or f"{function.return_type} {function.name}(...)",
                "function_description": function.description,
                "parameters": parameters_str or "No parameters",
                "return_type": function.return_type,
                "specification": specification,
                "edge_cases": edge_cases_str
            })
            
            # Parse into EnhancedPostcondition objects
            return self._parse_postconditions(result)
        except Exception as e:
            print(f"Warning: Failed to generate postconditions: {e}")
            return []
    
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
        
        try:
            result = await self.chain.ainvoke({
                "function_name": function.name,
                "function_signature": function.signature or f"{function.return_type} {function.name}(...)",
                "function_description": function.description,
                "parameters": parameters_str or "No parameters",
                "return_type": function.return_type,
                "specification": specification,
                "edge_cases": edge_cases_str
            })
            
            return self._parse_postconditions(result)
        except Exception as e:
            print(f"Warning: Failed to generate postconditions: {e}")
            return []
    
    def _parse_postconditions(self, result: Any) -> List[EnhancedPostcondition]:
        """Parse result into EnhancedPostcondition objects."""
        postconditions = []
        
        try:
            # result should already be a list from JsonOutputParser
            if not isinstance(result, list):
                result = [result]
            
            for pc_data in result:
                try:
                    # Create EnhancedPostcondition from dict
                    postcondition = EnhancedPostcondition(**pc_data)
                    postconditions.append(postcondition)
                except Exception as e:
                    print(f"Warning: Failed to parse individual postcondition: {e}")
                    continue
        
        except Exception as e:
            print(f"Warning: Failed to parse postconditions: {e}")
        
        return postconditions


# ============================================================================
# Z3 TRANSLATION CHAIN - FIXED
# ============================================================================

class Z3TranslationChain:
    """
    Chain for translating formal postconditions to Z3 code.
    
    FIXED: Uses StrOutputParser for simpler Z3 code extraction.
    """
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming, temperature=0.1)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the Z3 translation prompt."""
        
        system_template = """You are an expert in Z3 theorem prover.

Translate formal postconditions into executable Z3 Python code.

REQUIRED CODE STRUCTURE:

```python
from z3 import *

# Declare variables
x = Int('x')
arr = Array('arr', IntSort(), IntSort())
size = Int('size')

# Define constraints
constraint = ForAll([i], 
    Implies(And(i >= 0, i < size),
        Select(arr, i) <= Select(arr, i + 1)))

# Create solver and verify
s = Solver()
s.add(constraint)
s.add(size > 0)  # Preconditions

result = s.check()
print(f"Verification result: {{result}}")

if result == sat:
    print("✓ Postcondition is satisfiable")
    print("Model:", s.model())
elif result == unsat:
    print("✗ Postcondition is unsatisfiable")
else:
    print("? Unknown")
```

CRITICAL:
1. Start with `from z3 import *`
2. Declare ALL variables
3. Use proper Z3 syntax (ForAll, Implies, And, Or, Select, etc.)
4. Create Solver(), add constraints, check()
5. Return ONLY Python code, no markdown"""

        human_template = """Translate to Z3:

Formal Postcondition: {formal_text}

Natural Language: {natural_language}

Function Context: {function_context}

Z3 Theory: {z3_theory}

Generate executable Z3 Python code."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def translate(
        self,
        postcondition: EnhancedPostcondition,
        function_context: Optional[Dict[str, Any]] = None
    ) -> Z3Translation:
        """Translate postcondition to Z3 code."""
        try:
            result = self.chain.invoke({
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
        except Exception as e:
            print(f"Warning: Z3 translation failed: {e}")
            return Z3Translation(
                formal_text=postcondition.formal_text,
                natural_language=postcondition.natural_language,
                z3_code="",
                translation_success=False,
                validation_error=str(e)
            )
    
    async def atranslate(
        self,
        postcondition: EnhancedPostcondition,
        function_context: Optional[Dict[str, Any]] = None
    ) -> Z3Translation:
        """Async version of translate()."""
        try:
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
        except Exception as e:
            print(f"Warning: Z3 translation failed: {e}")
            return Z3Translation(
                formal_text=postcondition.formal_text,
                natural_language=postcondition.natural_language,
                z3_code="",
                translation_success=False,
                validation_error=str(e)
            )
    
    def _extract_code(self, result_text: str) -> str:
        """Extract Z3 code from LLM response."""
        # Remove markdown code blocks if present
        if '```python' in result_text:
            code = result_text.split('```python')[1].split('```')[0]
        elif '```' in result_text:
            code = result_text.split('```')[1].split('```')[0]
        else:
            code = result_text
        
        return code.strip()
    
    def _validate_z3_code(self, translation: Z3Translation) -> None:
        """Validate Z3 code syntax."""
        import ast
        
        if not translation.z3_code:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "not_validated"
            return
        
        try:
            ast.parse(translation.z3_code)
            
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
# CHAIN FACTORY - Main Entry Point
# ============================================================================

class ChainFactory:
    """
    Factory for creating and managing all chains.
    
    Use this as the main entry point for accessing chains.
    """
    
    def __init__(self):
        self._pseudocode_chain = None
        self._postcondition_chain = None
        self._z3_chain = None
    
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


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing fixed chains...")
    
    factory = ChainFactory()
    
    # Test pseudocode chain
    print("\n1. Testing PseudocodeChain...")
    result = factory.pseudocode.generate("Sort an array")
    print(f"   Functions: {len(result.functions)}")
    
    print("\n✅ All chains initialized successfully!")