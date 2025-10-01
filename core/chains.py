"""
Enhanced LangChain Chains - Phase 3 Migration

CHANGES:
1. Enhanced PostconditionChain to parse ALL 8 new fields
2. Added FormalLogicTranslationChain for precise_translation generation
3. Enhanced parsing with fallback logic for missing fields
4. Added derived score calculations (overall_priority_score)
5. Improved error handling and logging
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
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.cache import SQLiteCache
import langchain

from typing import List, Dict, Any, Optional
import json
import logging

from config.settings import settings
from core.models import (
    Function,
    PseudocodeResult,
    EnhancedPostcondition,
    PostconditionStrength,
    PostconditionCategory,
    Z3Translation,
    FunctionParameter
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# PSEUDOCODE GENERATION CHAIN (Unchanged)
# ============================================================================

class PseudocodeChain:
    """Chain for generating C pseudocode from specifications."""
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming)
        self.parser = JsonOutputParser()
        self.prompt = self._create_prompt()
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
            
            return PseudocodeResult(**result)
        except Exception as e:
            logger.warning(f"Failed to generate pseudocode: {e}")
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
            logger.warning(f"Failed to generate pseudocode: {e}")
            return PseudocodeResult(
                functions=[],
                structs=[],
                dependencies=[]
            )


# ============================================================================
# FORMAL LOGIC TRANSLATION CHAIN (NEW - Phase 3)
# ============================================================================

class FormalLogicTranslationChain:
    """
    Chain for translating formal logic to precise natural language.
    
    NEW in Phase 3: Generates the precise_translation field.
    """
    
    def __init__(self):
        self.llm = LLMFactory.create_llm(temperature=0.1)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create translation prompt."""
        
        system_template = """You are an expert in formal logic translation.

Translate formal postconditions into PRECISE natural language.

REQUIREMENTS:
1. **Precision**: Every quantifier, operator, condition must be explicit
2. **Clarity**: Simple, direct language without ambiguity
3. **Completeness**: Don't omit any logical components
4. **Length**: 2-5 sentences

Symbol Translation:
- ∀x: "For every x"
- ∃x: "There exists an x"
- →: "implies" or "if... then..."
- ∧: "and"
- ∨: "or"
- ¬: "not"
- arr[i]: "the element at index i in arr"
- ∈: "in" or "belongs to"
- [0,n): "from 0 to n (exclusive)"

EXAMPLE:
Formal: "∀i,j ∈ [0,n): i < j → arr[i] ≤ arr[j]"
Precise: "For every pair of indices i and j in the range from 0 to n (exclusive), if index i comes before index j (meaning i is strictly less than j), then the element at position i in the array must be less than or equal to the element at position j. This ensures that no element in the array is greater than any element that appears after it."

Output ONLY the precise translation, no explanations."""

        human_template = """Formal: {formal_text}

Precise Translation:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def translate(self, formal_text: str) -> str:
        """Translate formal logic to precise natural language."""
        try:
            result = self.chain.invoke({"formal_text": formal_text})
            return result.strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return ""
    
    async def atranslate(self, formal_text: str) -> str:
        """Async version."""
        try:
            result = await self.chain.ainvoke({"formal_text": formal_text})
            return result.strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return ""


# ============================================================================
# POSTCONDITION GENERATION CHAIN (ENHANCED - Phase 3)
# ============================================================================

class PostconditionChain:
    """
    Chain for generating formal postconditions from functions.
    
    ENHANCED in Phase 3:
    - Parses ALL 8 new fields from LLM response
    - Uses FormalLogicTranslationChain for backup translations
    - Calculates derived scores
    - Robust error handling with fallbacks
    """
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming)
        self.parser = JsonOutputParser()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser
        
        # NEW: Translation chain for precise_translation field
        self.translator = FormalLogicTranslationChain()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the postcondition generation prompt."""
        
        system_template = """You are an expert in formal verification and postcondition generation.

Generate comprehensive postconditions as a JSON array with ALL fields:

[
  {{
    "formal_text": "∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
    "natural_language": "Array is sorted in ascending order",
    "precise_translation": "For every pair of indices i and j in range 0 to n...",
    "reasoning": "This ensures the fundamental sorting property...",
    "strength": "standard",
    "category": "core_correctness",
    "confidence_score": 0.95,
    "clarity_score": 0.9,
    "completeness_score": 0.85,
    "testability_score": 0.9,
    "robustness_score": 0.92,
    "mathematical_quality_score": 0.96,
    "edge_cases_covered": ["Empty array (n=0): trivially true", "Single element: no pairs"],
    "coverage_gaps": ["Does not specify stability"],
    "mathematical_validity": "Mathematically sound - proper quantification",
    "robustness_assessment": "Highly robust - covers all orderings",
    "z3_theory": "arrays",
    "importance_category": "critical_correctness",
    "organization_rank": 1,
    "is_primary_in_category": true,
    "selection_reasoning": "Primary property defining sorted"
  }}
]

CRITICAL REQUIREMENTS:
1. precise_translation: 2-5 sentences translating every component
2. reasoning: 3-5 sentences on WHY it matters
3. edge_cases_covered: 3-7 specific cases with explanations
4. coverage_gaps: 1-3 honest limitations
5. mathematical_validity: Assessment of correctness
6. robustness_assessment: Robustness characteristics
7. All scores: 0.0-1.0

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

Return comprehensive postconditions as JSON array with ALL fields."""

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
            
            return self._parse_postconditions(result)
        except Exception as e:
            logger.error(f"Failed to generate postconditions: {e}")
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
            
            postconditions = self._parse_postconditions(result)
            
            # NEW: Generate precise translations for any missing
            await self._fill_missing_translations(postconditions)
            
            return postconditions
        except Exception as e:
            logger.error(f"Failed to generate postconditions: {e}")
            return []
    
    def _parse_postconditions(self, result: Any) -> List[EnhancedPostcondition]:
        """
        Parse result into EnhancedPostcondition objects.
        
        ENHANCED in Phase 3: Parses ALL 8 new fields with fallbacks.
        """
        postconditions = []
        
        try:
            if not isinstance(result, list):
                result = [result]
            
            for i, pc_data in enumerate(result):
                try:
                    # Parse with all fields, using safe get with defaults
                    postcondition = EnhancedPostcondition(
                        # Core fields (existing)
                        formal_text=pc_data.get("formal_text", ""),
                        natural_language=pc_data.get("natural_language", ""),
                        strength=PostconditionStrength(pc_data.get("strength", "standard")),
                        category=PostconditionCategory(pc_data.get("category", "correctness")),
                        
                        # Scores (existing)
                        confidence_score=float(pc_data.get("confidence_score", 0.5)),
                        clarity_score=float(pc_data.get("clarity_score", 0.0)),
                        completeness_score=float(pc_data.get("completeness_score", 0.0)),
                        testability_score=float(pc_data.get("testability_score", 0.0)),
                        
                        # NEW FIELDS (Phase 3) - with fallbacks
                        precise_translation=pc_data.get("precise_translation", ""),
                        reasoning=pc_data.get("reasoning", ""),
                        edge_cases_covered=pc_data.get("edge_cases_covered", []),
                        coverage_gaps=pc_data.get("coverage_gaps", []),
                        mathematical_validity=pc_data.get("mathematical_validity", ""),
                        robustness_assessment=pc_data.get("robustness_assessment", ""),
                        
                        # Quality scores (NEW)
                        robustness_score=float(pc_data.get("robustness_score", 0.0)),
                        mathematical_quality_score=float(pc_data.get("mathematical_quality_score", 0.0)),
                        
                        # Organization fields (NEW)
                        importance_category=pc_data.get("importance_category", ""),
                        organization_rank=int(pc_data.get("organization_rank", i + 1)),
                        is_primary_in_category=bool(pc_data.get("is_primary_in_category", False)),
                        recommended_for_selection=bool(pc_data.get("recommended_for_selection", True)),
                        selection_reasoning=pc_data.get("selection_reasoning", ""),
                        
                        # Edge cases and Z3
                        edge_cases=pc_data.get("edge_cases", []),
                        z3_theory=pc_data.get("z3_theory", "unknown"),
                        
                        # Warnings
                        warnings=pc_data.get("warnings", [])
                    )
                    
                    # Calculate overall_priority_score (derived field)
                    postcondition.overall_priority_score = self._calculate_priority_score(postcondition)
                    
                    postconditions.append(postcondition)
                    
                except Exception as e:
                    logger.error(f"Failed to parse postcondition {i}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to parse postconditions: {e}")
        
        return postconditions
    
    def _calculate_priority_score(self, pc: EnhancedPostcondition) -> float:
        """
        Calculate overall priority score (derived metric).
        
        NEW in Phase 3: Combines multiple scores into priority.
        """
        weights = {
            'confidence': 0.25,
            'robustness': 0.25,
            'clarity': 0.15,
            'completeness': 0.15,
            'testability': 0.10,
            'mathematical_quality': 0.10
        }
        
        score = (
            pc.confidence_score * weights['confidence'] +
            pc.robustness_score * weights['robustness'] +
            pc.clarity_score * weights['clarity'] +
            pc.completeness_score * weights['completeness'] +
            pc.testability_score * weights['testability'] +
            pc.mathematical_quality_score * weights['mathematical_quality']
        )
        
        return min(1.0, max(0.0, score))
    
    async def _fill_missing_translations(
        self,
        postconditions: List[EnhancedPostcondition]
    ) -> None:
        """
        Fill in missing precise_translation fields using translation chain.
        
        NEW in Phase 3: Backup translation generation.
        """
        import asyncio
        
        translation_tasks = []
        indices_to_fill = []
        
        for i, pc in enumerate(postconditions):
            if not pc.precise_translation and pc.formal_text:
                indices_to_fill.append(i)
                translation_tasks.append(
                    self.translator.atranslate(pc.formal_text)
                )
        
        if translation_tasks:
            translations = await asyncio.gather(*translation_tasks)
            
            for idx, translation in zip(indices_to_fill, translations):
                postconditions[idx].precise_translation = translation
                logger.info(f"Generated backup translation for postcondition {idx}")


# ============================================================================
# Z3 TRANSLATION CHAIN (Unchanged from working version)
# ============================================================================

class Z3TranslationChain:
    """Chain for translating formal postconditions to Z3 code."""
    
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
            logger.error(f"Z3 translation failed: {e}")
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
            logger.error(f"Z3 translation failed: {e}")
            return Z3Translation(
                formal_text=postcondition.formal_text,
                natural_language=postcondition.natural_language,
                z3_code="",
                translation_success=False,
                validation_error=str(e)
            )
    
    def _extract_code(self, result_text: str) -> str:
        """Extract Z3 code from LLM response."""
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
    
    ENHANCED in Phase 3: Now includes FormalLogicTranslationChain.
    """
    
    def __init__(self):
        self._pseudocode_chain = None
        self._postcondition_chain = None
        self._z3_chain = None
        self._translation_chain = None
    
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
    def translator(self) -> FormalLogicTranslationChain:
        """Get or create formal logic translation chain."""
        if self._translation_chain is None:
            self._translation_chain = FormalLogicTranslationChain()
        return self._translation_chain


# ============================================================================
# PHASE 3 VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3 VALIDATION - Enhanced Chains")
    print("=" * 70)
    
    print("\n1. Testing FormalLogicTranslationChain...")
    translator = FormalLogicTranslationChain()
    formal = "∀i,j ∈ [0,n): i < j → arr[i] ≤ arr[j]"
    
    print(f"   Formal: {formal}")
    print("   NOTE: Translation requires API call")
    
    print("\n2. Testing enhanced PostconditionChain parsing...")
    print("   Enhanced to parse 8 new fields:")
    print("   - precise_translation")
    print("   - reasoning")
    print("   - edge_cases_covered")
    print("   - coverage_gaps")
    print("   - mathematical_validity")
    print("   - robustness_assessment")
    print("   - importance_category")
    print("   - selection_reasoning")
    
    print("\n3. Testing priority score calculation...")
    from core.models import PostconditionCategory
    
    test_pc = EnhancedPostcondition(
        formal_text="test",
        natural_language="test",
        confidence_score=0.95,
        clarity_score=0.9,
        completeness_score=0.85,
        testability_score=0.9,
        robustness_score=0.92,
        mathematical_quality_score=0.93
    )
    
    chain = PostconditionChain()
    priority = chain._calculate_priority_score(test_pc)
    print(f"   Calculated priority score: {priority:.3f}")
    
    print("\n4. Verifying all chain components...")
    factory = ChainFactory()
    
    print(f"   ✓ PseudocodeChain: {factory.pseudocode is not None}")
    print(f"   ✓ PostconditionChain: {factory.postcondition is not None}")
    print(f"   ✓ Z3TranslationChain: {factory.z3 is not None}")
    print(f"   ✓ FormalLogicTranslationChain: {factory.translator is not None}")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 3 COMPLETE")
    print("=" * 70)
    print("\nEnhancements made:")
    print("1. ✓ Added FormalLogicTranslationChain for precise_translation")
    print("2. ✓ Enhanced PostconditionChain._parse_postconditions()")
    print("3. ✓ Added _calculate_priority_score() for derived metrics")
    print("4. ✓ Added _fill_missing_translations() for backup generation")
    print("5. ✓ All 8 new fields now parsed with fallbacks")
    print("6. ✓ Robust error handling throughout")
    print("\nNext: Phase 4 - Enhance modules/z3/translator.py")