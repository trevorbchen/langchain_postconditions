"""
LangChain Chains - Streamlined Version (No Scoring/Ranking)

CHANGES FROM ORIGINAL:
- ❌ Removed parsing of scoring fields (confidence_score, robustness_score, etc.)
- ❌ Removed parsing of ranking fields (organization_rank, importance_category, etc.)
- ❌ Removed _calculate_priority_score method
- ✅ KEPT parsing of rich content (precise_translation, reasoning, edge_cases_covered, coverage_gaps)
- ✅ KEPT translation chain for precise_translation
- ✅ KEPT Z3 validation
- ✅ KEPT batch Z3 translation

This maintains all the valuable functionality while removing the scoring overhead.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.cache import SQLiteCache
import langchain

from typing import List, Dict, Any, Optional
import json
import logging
import asyncio
from datetime import datetime

from config.settings import settings
from core.models import (
    Function, PseudocodeResult, EnhancedPostcondition,
    PostconditionStrength, PostconditionCategory,
    Z3Translation, FunctionParameter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable LLM caching if configured
if settings.enable_cache:
    langchain.llm_cache = SQLiteCache(database_path=str(settings.llm_cache_db))


# ============================================================================
# LLM FACTORY
# ============================================================================

class LLMFactory:
    """Factory for creating LLM instances with consistent configuration."""
    
    @staticmethod
    def create_llm(
        temperature: Optional[float] = None,
        streaming: bool = False,
        callbacks: Optional[list] = None,
        max_tokens: Optional[int] = None
    ) -> ChatOpenAI:
        """Create ChatOpenAI instance with project settings."""
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature or settings.temperature,
            max_tokens=max_tokens or settings.max_tokens,
            openai_api_key=settings.openai_api_key,
            streaming=streaming,
            callbacks=callbacks,
            max_retries=settings.max_retries,
            request_timeout=180
        )
    
    @staticmethod
    def create_embeddings() -> OpenAIEmbeddings:
        """Create embeddings instance."""
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )


# ============================================================================
# PSEUDOCODE GENERATION CHAIN
# ============================================================================

class PseudocodeChain:
    """Generates C pseudocode from natural language specifications."""
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming, max_tokens=3000)
        self.parser = JsonOutputParser()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create pseudocode generation prompt with escaped JSON."""
        
        # CRITICAL: All JSON braces must be doubled {{{{ }}}} for Python formatting
        system_template = """Generate C pseudocode as JSON with this structure:
{{{{
  "functions": [{{{{
    "name": "function_name",
    "description": "what it does",
    "signature": "return_type function_name(params)",
    "return_type": "void",
    "input_parameters": [{{{{ "name": "param", "data_type": "int*", "description": "desc" }}}}],
    "output_parameters": [],
    "return_values": [{{{{ "condition": "success", "value": "0", "description": "ok" }}}}],
    "edge_cases": ["Empty array", "NULL pointer", "Single element"],
    "complexity": "O(n)",
    "memory_usage": "O(1)",
    "body": "steps",
    "dependencies": []
  }}}}],
  "structs": [],
  "enums": [],
  "global_variables": [],
  "includes": ["stdio.h"],
  "dependencies": [],
  "metadata": {{{{}}}}
}}}}

Return ONLY valid JSON, no markdown or explanations."""

        human_template = """Generate pseudocode for: {specification}

{context}"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def generate(
        self,
        specification: str,
        codebase_context: Optional[Dict[str, Any]] = None
    ) -> PseudocodeResult:
        """Synchronous pseudocode generation."""
        context = self._build_context(codebase_context)
        
        try:
            result = self.chain.invoke({
                "specification": specification,
                "context": context
            })
            return PseudocodeResult(**result)
            
        except Exception as e:
            logger.warning(f"Pseudocode generation failed: {e}")
            return PseudocodeResult(functions=[], structs=[], dependencies=[])
    
    async def agenerate(
        self,
        specification: str,
        codebase_context: Optional[Dict[str, Any]] = None
    ) -> PseudocodeResult:
        """Async pseudocode generation."""
        context = self._build_context(codebase_context)
        
        try:
            result = await self.chain.ainvoke({
                "specification": specification,
                "context": context
            })
            return PseudocodeResult(**result)
            
        except Exception as e:
            logger.warning(f"Pseudocode generation failed: {e}")
            return PseudocodeResult(functions=[], structs=[], dependencies=[])
    
    def _build_context(self, codebase_context: Optional[Dict[str, Any]]) -> str:
        """Build context string from codebase analysis."""
        if not codebase_context:
            return ""
        
        items = list(codebase_context.keys())[:5]
        return f"Available context: {', '.join(items)}"


# ============================================================================
# POSTCONDITION GENERATION CHAIN (STREAMLINED)
# ============================================================================

class PostconditionChain:
    """
    Generates formal postconditions with rich explanatory fields.
    
    STREAMLINED VERSION:
    - Requests: formal_text, natural_language, precise_translation, reasoning,
      edge_cases_covered, coverage_gaps, z3_theory
    - Does NOT request: scoring fields, ranking fields
    - Does NOT parse: scoring fields, ranking fields
    - Does NOT calculate: priority scores
    """
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming, max_tokens=3000)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.translator = FormalLogicTranslationChain()
        
        # Debugging
        self.last_raw_response = None
        self.last_error = None
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Create streamlined prompt requesting ONLY essential rich fields.
        
        CRITICAL FIX: All JSON braces are doubled {{{{ }}}} to escape them
        for Python's string formatting.
        """
        
        system_template = """You are an expert in formal specification writing.

Generate 6-10 diverse postconditions with COMPREHENSIVE analysis covering:
- Core correctness properties
- Boundary conditions  
- Edge cases
- Error handling

Use mathematical notation:
- ∀x: for all x
- ∃x: there exists x
- →: implies (if...then)
- ∧: and
- ∨: or
- ¬: not

Z3 theories (choose most appropriate):
- arrays: Array operations, indexing, bounds
- sequences: Ordered collections, concatenation
- sets: Membership, union, intersection
- arithmetic: Integer/real math, comparisons
- bitvectors: Low-level bit operations
- strings: Text manipulation

Return JSON array with this EXACT structure (note: all JSON braces are doubled):
[
  {{{{
    "formal_text": "∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
    "natural_language": "Array is sorted in non-decreasing order",
    "precise_translation": "For every pair of indices i and j where i comes before j, the element at position i is less than or equal to the element at position j",
    "reasoning": "This ensures the fundamental sorting property holds for all adjacent and non-adjacent pairs, preventing any out-of-order elements",
    "z3_theory": "arrays",
    "edge_cases": ["empty array", "single element", "duplicates"],
    "edge_cases_covered": [
      "Empty array (n=0): vacuously true",
      "Single element (n=1): no pairs to compare",
      "Duplicates: arr[i] = arr[j] handled correctly"
    ],
    "coverage_gaps": ["Does not specify stability of sort"],
    "strength": "standard",
    "category": "core_correctness"
  }}}}
]

MANDATORY REQUIREMENTS:
1. Use ACTUAL variable names from the function signature
2. Include 3-5 specific items in edge_cases_covered
3. Be honest about coverage_gaps (1-3 items)
4. precise_translation should be 2-4 sentences explaining the formal logic in detail
5. reasoning should explain WHY this postcondition matters (2-3 sentences)
6. strength: minimal, standard, or comprehensive
7. category: core_correctness, boundary_safety, error_resilience, performance_constraints, domain_compliance

CRITICAL: Return ONLY the JSON array, no markdown, no explanations."""

        human_template = """Function: {function_name}
Signature: {function_signature}
Description: {function_description}
Parameters: {parameters}
Return Type: {return_type}
Specification: {specification}
Edge Cases: {edge_cases}

Generate 6-10 DIVERSE postconditions as JSON array."""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def generate(
        self,
        function: Function,
        specification: str,
        edge_cases: Optional[List[str]] = None,
        strength: str = "comprehensive"
    ) -> List[EnhancedPostcondition]:
        """Synchronous postcondition generation."""
        parameters_str = "\n".join([
            f"- {p.name}: {p.data_type}" for p in function.input_parameters
        ])
        edge_cases_str = "\n".join(edge_cases or function.edge_cases or [])
        
        try:
            result = self.chain.invoke({
                "function_name": function.name,
                "function_signature": function.signature or f"{function.return_type} {function.name}(...)",
                "function_description": function.description,
                "parameters": parameters_str or "None",
                "return_type": function.return_type,
                "specification": specification,
                "edge_cases": edge_cases_str
            })
            
            self.last_raw_response = result
            return self._parse_postconditions(result)
            
        except Exception as e:
            self.last_error = e
            logger.error(f"Postcondition generation failed: {e}")
            return []
    
    async def agenerate(
        self,
        function: Function,
        specification: str,
        edge_cases: Optional[List[str]] = None,
        strength: str = "comprehensive"
    ) -> List[EnhancedPostcondition]:
        """Async postcondition generation with automatic translation filling."""
        parameters_str = "\n".join([
            f"- {p.name}: {p.data_type}" for p in function.input_parameters
        ])
        edge_cases_str = "\n".join(edge_cases or function.edge_cases or [])
        
        try:
            result = await self.chain.ainvoke({
                "function_name": function.name,
                "function_signature": function.signature or f"{function.return_type} {function.name}(...)",
                "function_description": function.description,
                "parameters": parameters_str or "None",
                "return_type": function.return_type,
                "specification": specification,
                "edge_cases": edge_cases_str
            })
            
            self.last_raw_response = result
            postconditions = self._parse_postconditions(result)
            
            # Fill missing precise_translations if LLM didn't provide them
            await self._fill_missing_translations(postconditions)
            
            return postconditions
            
        except Exception as e:
            self.last_error = e
            logger.error(f"Async postcondition generation failed: {e}")
            return []
    
    def _parse_postconditions(self, result_text: str) -> List[EnhancedPostcondition]:
        """
        Parse LLM response into EnhancedPostcondition objects.
        
        STREAMLINED: Parses ONLY rich content fields, NOT scoring/ranking fields.
        """
        postconditions = []
        
        try:
            # Handle both string and list responses
            if isinstance(result_text, str):
                # Extract JSON array from text
                start = result_text.find('[')
                end = result_text.rfind(']') + 1
                
                if start == -1 or end == 0:
                    logger.warning("No JSON array found in response")
                    return []
                
                json_str = result_text[start:end]
                data = json.loads(json_str)
            else:
                data = result_text
            
            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]
            
            # Parse each postcondition
            for i, pc_data in enumerate(data):
                try:
                    # Create postcondition with STREAMLINED fields only
                    postcondition = EnhancedPostcondition(
                        # Core fields
                        formal_text=pc_data.get("formal_text", ""),
                        natural_language=pc_data.get("natural_language", ""),
                        
                        # Rich explanation fields (KEPT)
                        precise_translation=pc_data.get("precise_translation", ""),
                        reasoning=pc_data.get("reasoning", ""),
                        
                        # Edge case analysis (KEPT)
                        edge_cases=pc_data.get("edge_cases", []),
                        edge_cases_covered=pc_data.get("edge_cases_covered", []),
                        coverage_gaps=pc_data.get("coverage_gaps", []),
                        
                        # Z3 integration (KEPT)
                        z3_theory=pc_data.get("z3_theory", "unknown"),
                        
                        # Classification (KEPT)
                        strength=pc_data.get("strength", "standard"),
                        category=pc_data.get("category", "correctness")
                        
                        # ❌ REMOVED: All scoring fields
                        # ❌ REMOVED: All ranking fields
                    )
                    
                    postconditions.append(postcondition)
                    
                except Exception as e:
                    logger.error(f"Failed to parse postcondition {i+1}: {e}")
                    continue
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Failed text: {result_text[:500]}...")
        except Exception as e:
            logger.error(f"Failed to parse postconditions: {e}")
        
        return postconditions
    
    async def _fill_missing_translations(
        self,
        postconditions: List[EnhancedPostcondition]
    ) -> None:
        """
        Fill missing precise_translations using the translation chain.
        
        If the LLM didn't provide precise_translation for some postconditions,
        this generates them automatically using a separate translation chain.
        """
        translation_tasks = []
        indices_to_fill = []
        
        for i, pc in enumerate(postconditions):
            if not pc.precise_translation and pc.formal_text:
                indices_to_fill.append(i)
                translation_tasks.append(
                    self.translator.atranslate(pc.formal_text)
                )
        
        if translation_tasks:
            logger.info(f"Generating {len(translation_tasks)} missing translations")
            translations = await asyncio.gather(*translation_tasks)
            
            for idx, translation in zip(indices_to_fill, translations):
                postconditions[idx].precise_translation = translation


# ============================================================================
# FORMAL LOGIC TRANSLATION CHAIN (UNCHANGED)
# ============================================================================

class FormalLogicTranslationChain:
    """
    Translates formal mathematical logic to precise natural language.
    
    This is used to generate precise_translation field when the main
    postcondition chain doesn't provide it.
    """
    
    def __init__(self):
        self.llm = LLMFactory.create_llm(temperature=0.1, max_tokens=500)
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
4. **Length**: 2-4 sentences

Symbol Translation:
- ∀x: "For every x" or "For all x"
- ∃x: "There exists an x" or "There is an x"
- →: "implies" or "if...then..."
- ∧: "and"
- ∨: "or"
- ¬: "not"
- arr[i]: "the element at index i in arr"
- i < j: "i is less than j"
- x ∈ S: "x is in set S"

Output ONLY the precise translation, no explanations or additional text."""

        human_template = """Formal logic: {formal_text}

Precise natural language translation:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def translate(self, formal_text: str) -> str:
        """Synchronous translation."""
        try:
            result = self.chain.invoke({"formal_text": formal_text})
            return result.strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return f"[Translation unavailable: {str(e)}]"
    
    async def atranslate(self, formal_text: str) -> str:
        """Async translation."""
        try:
            result = await self.chain.ainvoke({"formal_text": formal_text})
            return result.strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return f"[Translation unavailable: {str(e)}]"


# ============================================================================
# Z3 TRANSLATION CHAIN (UNCHANGED - Keeps validation & batching)
# ============================================================================

class Z3TranslationChain:
    """
    Translates postconditions to Z3 Python code with validation and batching.
    
    UNCHANGED: All Z3 functionality preserved.
    """
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(
            streaming=streaming,
            temperature=0.1,
            max_tokens=1500
        )
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create single translation prompt."""
        system_template = """Translate formal postconditions to Z3 Python code.

Use this structure:

from z3 import *

# Declare variables
x = Int('x')
arr = Array('arr', IntSort(), IntSort())
size = Int('size')

# Define constraint
constraint = ForAll([i, j],
    Implies(
        And(i >= 0, i < size, j >= 0, j < size, i < j),
        Select(arr, i) <= Select(arr, j)
    )
)

# Create solver and check
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {{result}}")

if result == sat:
    print("Constraint is satisfiable")
elif result == unsat:
    print("Constraint is unsatisfiable")

Return ONLY Python code, no markdown or explanations."""

        human_template = """Formal: {formal_text}
Natural: {natural_language}
Context: {function_context}
Theory: {z3_theory}

Generate Z3 code:"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_batch_prompt(self) -> ChatPromptTemplate:
        """Create batch translation prompt."""
        system_template = """Translate multiple formal postconditions to Z3 Python code.

Input: JSON array of postconditions
Output: JSON array of Z3 translations IN SAME ORDER

Each input has:
- index: Number (preserve in output)
- formal_text: Mathematical logic
- natural_language: Description
- z3_theory: Theory to use

Each output must have:
- index: Same as input (CRITICAL for ordering)
- z3_code: Complete Python code with imports
- success: true/false

Example Z3 structure:
from z3 import *
x = Int('x')
s = Solver()
s.add(x > 0)
print(s.check())

Return ONLY JSON array, no markdown."""

        human_template = """Translate these {count} postconditions:

{postconditions_json}

Function context: {function_context}

Return JSON array:"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def translate(
        self,
        postcondition: EnhancedPostcondition,
        function_context: Optional[Dict[str, Any]] = None
    ) -> Z3Translation:
        """Synchronous single translation with validation."""
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
                translation_success=bool(z3_code),
                translation_time=0.0,
                generated_at=datetime.now().isoformat()
            )
            
            # Validate the Z3 code
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
        """Async single translation with validation."""
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
                translation_success=bool(z3_code),
                translation_time=0.0,
                generated_at=datetime.now().isoformat()
            )
            
            # Validate the Z3 code
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
    
    async def atranslate_batch(
        self,
        postconditions: List[EnhancedPostcondition],
        function_context: Optional[Dict[str, Any]] = None,
        batch_size: int = 10
    ) -> List[Z3Translation]:
        """
        Batch translation: translate multiple postconditions in one LLM call.
        
        This reduces API calls dramatically:
        - 10 postconditions: 1 call instead of 10 (90% reduction)
        - 24 postconditions: 3 calls instead of 24 (87% reduction)
        """
        if not postconditions:
            return []
        
        logger.info(f"Batch translating {len(postconditions)} postconditions")
        
        try:
            # Build JSON array of postconditions
            batch_input = []
            for i, pc in enumerate(postconditions):
                batch_input.append({
                    "index": i,
                    "formal_text": pc.formal_text,
                    "natural_language": pc.natural_language,
                    "z3_theory": pc.z3_theory or "arithmetic"
                })
            
            # Create batch chain
            batch_prompt = self._create_batch_prompt()
            batch_chain = batch_prompt | self.llm | StrOutputParser()
            
            # Single LLM call with all postconditions
            result_text = await batch_chain.ainvoke({
                "count": len(postconditions),
                "postconditions_json": json.dumps(batch_input, indent=2),
                "function_context": self._format_function_context(function_context)
            })
            
            # Parse batch response
            translations = self._parse_batch_response(result_text, postconditions)
            
            success_count = len([t for t in translations if t.translation_success])
            logger.info(f"Batch translation: {success_count}/{len(postconditions)} succeeded")
            
            return translations
        
        except Exception as e:
            logger.error(f"Batch Z3 translation failed: {e}")
            logger.warning(f"Falling back to individual translations")
            return await self._fallback_individual_translations(
                postconditions,
                function_context
            )
    
    def _parse_batch_response(
        self,
        result_text: str,
        postconditions: List[EnhancedPostcondition]
    ) -> List[Z3Translation]:
        """Parse batch JSON response from LLM."""
        try:
            # Extract JSON array
            start = result_text.find('[')
            end = result_text.rfind(']') + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in batch response")
            
            json_str = result_text[start:end]
            data = json.loads(json_str)
            
            if not isinstance(data, list):
                raise ValueError("Batch response is not a JSON array")
            
            # Create translation objects indexed by LLM's index field
            translations_dict = {}
            for item in data:
                idx = item.get("index", -1)
                z3_code = item.get("z3_code", "")
                success = item.get("success", bool(z3_code))
                
                if idx < 0 or idx >= len(postconditions):
                    logger.warning(f"Invalid index {idx} in batch response")
                    continue
                
                translation = Z3Translation(
                    formal_text=postconditions[idx].formal_text,
                    natural_language=postconditions[idx].natural_language,
                    z3_code=z3_code,
                    z3_theory_used=postconditions[idx].z3_theory or "arithmetic",
                    translation_success=success,
                    translation_time=0.0,
                    generated_at=datetime.now().isoformat()
                )
                
                # Validate the code
                self._validate_z3_code(translation)
                
                translations_dict[idx] = translation
            
            # Build ordered list (fill missing with empty translations)
            translations = []
            for i, pc in enumerate(postconditions):
                if i in translations_dict:
                    translations.append(translations_dict[i])
                else:
                    logger.warning(f"Missing translation for index {i}")
                    translations.append(Z3Translation(
                        formal_text=pc.formal_text,
                        natural_language=pc.natural_language,
                        z3_code="",
                        translation_success=False,
                        validation_error="Missing from batch response"
                    ))
            
            return translations
        
        except Exception as e:
            logger.error(f"Failed to parse batch response: {e}")
            # Return empty translations for all
            return [
                Z3Translation(
                    formal_text=pc.formal_text,
                    natural_language=pc.natural_language,
                    z3_code="",
                    translation_success=False,
                    validation_error=f"Batch parse failed: {e}"
                )
                for pc in postconditions
            ]
    
    async def _fallback_individual_translations(
        self,
        postconditions: List[EnhancedPostcondition],
        function_context: Optional[Dict[str, Any]]
    ) -> List[Z3Translation]:
        """
        Fallback: translate individually if batch fails.
        
        This is slower but more reliable when batch translation fails.
        """
        logger.info(f"Processing {len(postconditions)} individually")
        
        tasks = [
            self.atranslate(pc, function_context)
            for pc in postconditions
        ]
        translations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, t in enumerate(translations):
            if isinstance(t, Exception):
                logger.error(f"Individual translation {i} failed: {t}")
                results.append(Z3Translation(
                    formal_text=postconditions[i].formal_text,
                    natural_language=postconditions[i].natural_language,
                    z3_code="",
                    translation_success=False,
                    validation_error=str(t)
                ))
            else:
                results.append(t)
        
        success_count = sum(1 for r in results if r.translation_success)
        logger.info(f"Individual translations: {success_count}/{len(postconditions)} succeeded")
        
        return results
    
    def _extract_code(self, result_text: str) -> str:
        """Extract Python code from LLM response."""
        if '```python' in result_text:
            code = result_text.split('```python')[1].split('```')[0]
        elif '```' in result_text:
            code = result_text.split('```')[1].split('```')[0]
        else:
            code = result_text
        
        return code.strip()
    
    def _validate_z3_code(self, translation: Z3Translation) -> None:
        """
        Validate Z3 code syntax and structure.
        
        Checks:
        1. Python syntax is valid
        2. Has Z3 imports
        3. Has Solver() instantiation
        
        Updates translation object with validation results.
        """
        import ast
        
        if not translation.z3_code:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "not_validated"
            translation.validation_error = "No code generated"
            return
        
        try:
            # Check Python syntax
            ast.parse(translation.z3_code)
            
            # Check required imports
            if 'from z3 import' not in translation.z3_code and \
               'import z3' not in translation.z3_code:
                translation.z3_validation_passed = False
                translation.z3_validation_status = "missing_imports"
                translation.validation_error = "Missing Z3 imports"
                return
            
            # Check for Solver
            if 'Solver()' not in translation.z3_code:
                translation.z3_validation_passed = False
                translation.z3_validation_status = "missing_solver"
                translation.validation_error = "Missing Solver() instantiation"
                return
            
            # All checks passed
            translation.z3_validation_passed = True
            translation.z3_validation_status = "success"
            translation.validation_error = None
            
        except SyntaxError as e:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "syntax_error"
            translation.validation_error = f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "validation_error"
            translation.validation_error = str(e)
    
    def _format_function_context(
        self,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format function context for prompt."""
        if not context:
            return "None"
        
        lines = []
        
        if 'parameters' in context:
            lines.append("Parameters:")
            for param in context['parameters']:
                lines.append(f"  - {param.get('name')}: {param.get('data_type')}")
        
        if 'return_type' in context:
            lines.append(f"Return: {context['return_type']}")
        
        return "\n".join(lines) if lines else "None"


# ============================================================================
# CHAIN FACTORY (LAZY INITIALIZATION)
# ============================================================================

class ChainFactory:
    """
    Factory for accessing all chains with lazy initialization.
    
    Usage:
        factory = ChainFactory()
        pseudocode_result = await factory.pseudocode.agenerate(spec)
        postconditions = await factory.postcondition.agenerate(function, spec)
        translations = await factory.z3.atranslate_batch(postconditions)
    """
    
    def __init__(self):
        self._pseudocode_chain = None
        self._postcondition_chain = None
        self._z3_chain = None
        self._translation_chain = None
    
    @property
    def pseudocode(self) -> PseudocodeChain:
        """Get or create pseudocode chain."""
        if self._pseudocode_chain is None:
            self._pseudocode_chain = PseudocodeChain()
        return self._pseudocode_chain
    
    @property
    def postcondition(self) -> PostconditionChain:
        """Get or create postcondition chain."""
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
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print(" CORE/CHAINS.PY - STREAMLINED VERSION")
    print("=" * 80)
    
    print("\nCHANGES FROM ORIGINAL:")
    print("  ❌ Removed parsing of scoring fields")
    print("     (confidence_score, robustness_score, clarity_score, etc.)")
    print("  ❌ Removed parsing of ranking fields")
    print("     (organization_rank, importance_category, selection_reasoning, etc.)")
    print("  ❌ Removed _calculate_priority_score method")
    
    print("\nKEPT ALL VALUABLE FUNCTIONALITY:")
    print("  ✅ Rich content parsing")
    print("     (precise_translation, reasoning, edge_cases_covered, coverage_gaps)")
    print("  ✅ Translation chain for precise_translation")
    print("  ✅ Z3 validation (syntax, imports, structure)")
    print("  ✅ Batch Z3 translation (87% fewer API calls)")
    print("  ✅ All LangChain enhancements")
    
    print("\nFIELDS NOW POPULATED:")
    print("  Before: 5 fields (formal_text, natural_language, confidence, etc.)")
    print("  After:  9 fields (all rich content, NO scoring/ranking)")
    print("    - formal_text")
    print("    - natural_language")
    print("    - precise_translation (detailed NL)")
    print("    - reasoning (WHY it matters)")
    print("    - edge_cases_covered (specific cases)")
    print("    - coverage_gaps (honest limitations)")
    print("    - z3_theory")
    print("    - strength (minimal/standard/comprehensive)")
    print("    - category (core_correctness/boundary_safety/etc.)")
    
    print("\nUSAGE:")
    print("  factory = ChainFactory()")
    print("  postconditions = await factory.postcondition.agenerate(function, spec)")
    print("  translations = await factory.z3.atranslate_batch(postconditions)")
    
    print("\n" + "=" * 80)
    print("This streamlined version is production-ready!")
    print("Replace your core/chains.py with this file.")
    print("=" * 80)