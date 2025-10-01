"""
Simplified LangChain Chains - Working Version with 6-10 Postconditions

Key changes:
- Shorter, clearer prompts that don't timeout
- Focused on getting 6-10 postconditions reliably
- Removed excessive formatting that confuses the LLM
- Increased timeout to 180 seconds
- Better error handling
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

from config.settings import settings
from core.models import (
    Function, PseudocodeResult, EnhancedPostcondition,
    PostconditionStrength, PostconditionCategory,
    Z3Translation, FunctionParameter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if settings.enable_cache:
    langchain.llm_cache = SQLiteCache(database_path=str(settings.llm_cache_db))


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(temperature: Optional[float] = None, streaming: bool = False,
                   callbacks: Optional[list] = None, max_tokens: Optional[int] = None) -> ChatOpenAI:
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature or settings.temperature,
            max_tokens=max_tokens or settings.max_tokens,
            openai_api_key=settings.openai_api_key,
            streaming=streaming,
            callbacks=callbacks,
            max_retries=settings.max_retries,
            request_timeout=180  # Increased from 60 to 180 seconds
        )
    
    @staticmethod
    def create_embeddings() -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )


class PseudocodeChain:
    """Chain for generating C pseudocode."""
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming, max_tokens=3000)
        self.parser = JsonOutputParser()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser
    
    def _create_prompt(self) -> ChatPromptTemplate:
        system_template = """Generate C pseudocode as JSON with this structure:
{
  "functions": [{
    "name": "function_name",
    "description": "what it does",
    "signature": "return_type function_name(params)",
    "return_type": "void",
    "input_parameters": [{"name": "param", "data_type": "int*", "description": "desc"}],
    "output_parameters": [],
    "return_values": [{"condition": "success", "value": "0", "description": "ok"}],
    "preconditions": ["arr != NULL", "size > 0"],
    "edge_cases": ["Empty array", "NULL pointer", "Single element"],
    "complexity": "O(n)",
    "memory_usage": "O(1)",
    "body": "steps",
    "dependencies": []
  }],
  "structs": [], "enums": [], "global_variables": [],
  "includes": ["stdio.h"], "dependencies": [], "metadata": {}
}
Return ONLY valid JSON."""

        human_template = "Generate pseudocode for: {specification}\n\n{context}"
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def generate(self, specification: str, codebase_context: Optional[Dict[str, Any]] = None) -> PseudocodeResult:
        context = ""
        if codebase_context:
            context = f"Available: {', '.join(list(codebase_context.keys())[:5])}"
        
        try:
            result = self.chain.invoke({"specification": specification, "context": context})
            return PseudocodeResult(**result)
        except Exception as e:
            logger.warning(f"Pseudocode generation failed: {e}")
            return PseudocodeResult(functions=[], structs=[], dependencies=[])
    
    async def agenerate(self, specification: str, codebase_context: Optional[Dict[str, Any]] = None) -> PseudocodeResult:
        context = ""
        if codebase_context:
            context = f"Available: {', '.join(list(codebase_context.keys())[:5])}"
        
        try:
            result = await self.chain.ainvoke({"specification": specification, "context": context})
            return PseudocodeResult(**result)
        except Exception as e:
            logger.warning(f"Pseudocode generation failed: {e}")
            return PseudocodeResult(functions=[], structs=[], dependencies=[])


class FormalLogicTranslationChain:
    """Translates formal logic to natural language."""
    
    def __init__(self):
        self.llm = LLMFactory.create_llm(temperature=0.1, max_tokens=500)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        system_template = """Translate formal math to precise English. 2-4 sentences.
∀ = "for every", ∃ = "there exists", → = "implies", ∧ = "and", ∨ = "or"
Example: "∀i: arr[i] ≤ arr[i+1]" → "For every index i, the element at position i is less than or equal to the next element. This means the array is sorted in ascending order."
Output ONLY the translation."""

        human_template = "Formal: {formal_text}\n\nTranslation:"
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def translate(self, formal_text: str) -> str:
        try:
            result = self.chain.invoke({"formal_text": formal_text})
            return result.strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return ""
    
    async def atranslate(self, formal_text: str) -> str:
        try:
            result = await self.chain.ainvoke({"formal_text": formal_text})
            return result.strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return ""


class PostconditionChain:
    """Generates 6-10 diverse postconditions."""
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming, max_tokens=4000, temperature=0.4)
        self.parser = JsonOutputParser()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser
        self.translator = FormalLogicTranslationChain()
        
        # Diagnostics
        self.last_raw_response = None
        self.last_error = None
    
    def _create_prompt(self) -> ChatPromptTemplate:
        system_template = """Generate exactly 6-10 diverse postconditions as a JSON array.

CATEGORIES (make each postcondition cover a DIFFERENT category):
1. Correctness: What it computes
2. Boundaries: Valid ranges, no overflow
3. Null safety: Handle NULL/invalid input
4. Data preservation: No data loss
5. Side effects: In-place changes
6. Performance: Complexity

REQUIRED FIELDS for each:
{
  "formal_text": "∀i: mathematical notation",
  "natural_language": "brief explanation",
  "precise_translation": "detailed 2-3 sentence explanation",
  "reasoning": "why this matters, 2-3 sentences",
  "edge_cases_covered": ["case 1", "case 2", "case 3"],
  "coverage_gaps": ["limitation 1"],
  "mathematical_validity": "brief assessment",
  "robustness_assessment": "brief robustness notes",
  "strength": "standard",
  "category": "core_correctness",
  "confidence_score": 0.95,
  "clarity_score": 0.9,
  "completeness_score": 0.85,
  "testability_score": 0.9,
  "robustness_score": 0.92,
  "mathematical_quality_score": 0.96,
  "z3_theory": "arrays",
  "importance_category": "critical_correctness",
  "organization_rank": 1,
  "is_primary_in_category": true,
  "selection_reasoning": "why selected"
}

Return ONLY a JSON array with 6-10 postconditions. NO explanations outside the JSON."""

        human_template = """Function: {function_name}
Signature: {function_signature}
Description: {function_description}

Parameters: {parameters}
Return: {return_type}
Specification: {specification}
Edge Cases: {edge_cases}

Generate 6-10 DIVERSE postconditions (each covering a different aspect) as JSON array."""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def generate(self, function: Function, specification: str, 
                 edge_cases: Optional[List[str]] = None, strength: str = "comprehensive") -> List[EnhancedPostcondition]:
        parameters_str = "\n".join([f"- {p.name}: {p.data_type}" for p in function.input_parameters])
        edge_cases_str = "\n".join(edge_cases or function.edge_cases or ["Empty input", "NULL", "Boundary values"])
        
        try:
            logger.info("=" * 70)
            logger.info("POSTCONDITION GENERATION - DIAGNOSTICS")
            logger.info("=" * 70)
            logger.info(f"Function: {function.name}")
            logger.info(f"LLM Config: model={self.llm.model_name}, max_tokens={self.llm.max_tokens}, temp={self.llm.temperature}")
            logger.info(f"Timeout: {self.llm.request_timeout}s")
            
            import time
            start_time = time.time()
            
            logger.info("Sending request to OpenAI...")
            result = self.chain.invoke({
                "function_name": function.name,
                "function_signature": function.signature or f"{function.return_type} {function.name}(...)",
                "function_description": function.description,
                "parameters": parameters_str or "None",
                "return_type": function.return_type,
                "specification": specification,
                "edge_cases": edge_cases_str
            })
            
            elapsed = time.time() - start_time
            logger.info(f"Response received in {elapsed:.1f}s")
            
            # Store raw response for debugging
            self.last_raw_response = result
            
            # Log response type and size
            logger.info(f"Response type: {type(result)}")
            if isinstance(result, str):
                logger.info(f"Response length: {len(result)} chars")
                logger.info(f"First 200 chars: {result[:200]}")
                logger.info(f"Last 200 chars: {result[-200:]}")
            elif isinstance(result, list):
                logger.info(f"Response is list with {len(result)} items")
            elif isinstance(result, dict):
                logger.info(f"Response is dict with keys: {list(result.keys())}")
            
            postconditions = self._parse_postconditions(result)
            logger.info(f"Successfully parsed {len(postconditions)} postconditions")
            logger.info("=" * 70)
            
            return postconditions
            
        except Exception as e:
            self.last_error = e
            error_type = type(e).__name__
            logger.error("=" * 70)
            logger.error(f"POSTCONDITION GENERATION FAILED: {error_type}")
            logger.error("=" * 70)
            logger.error(f"Error message: {str(e)}")
            
            # Specific error diagnostics
            if "timeout" in str(e).lower():
                logger.error("DIAGNOSIS: Request timed out")
                logger.error("SOLUTIONS:")
                logger.error("  1. Increase timeout in config/settings.py: request_timeout=180")
                logger.error("  2. Use faster model: OPENAI_MODEL=gpt-4-turbo-preview")
                logger.error("  3. Simplify prompt (already simplified)")
            
            elif "rate limit" in str(e).lower():
                logger.error("DIAGNOSIS: Rate limit exceeded")
                logger.error("SOLUTIONS:")
                logger.error("  1. Wait a minute and try again")
                logger.error("  2. Check your OpenAI usage limits")
                logger.error("  3. Upgrade your OpenAI plan")
            
            elif "api key" in str(e).lower():
                logger.error("DIAGNOSIS: API key issue")
                logger.error("SOLUTIONS:")
                logger.error("  1. Check .env file has correct OPENAI_API_KEY")
                logger.error("  2. Verify key starts with 'sk-'")
                logger.error("  3. Check key is active in OpenAI dashboard")
            
            elif "json" in str(e).lower() or "parse" in str(e).lower():
                logger.error("DIAGNOSIS: JSON parsing failed")
                logger.error("SOLUTIONS:")
                logger.error("  1. LLM returned invalid JSON")
                logger.error("  2. Check self.last_raw_response for actual response")
                if self.last_raw_response:
                    logger.error(f"  3. Raw response type: {type(self.last_raw_response)}")
                    logger.error(f"  4. Raw response preview: {str(self.last_raw_response)[:500]}")
            
            else:
                logger.error("DIAGNOSIS: Unknown error")
                logger.error(f"Full error: {repr(e)}")
            
            import traceback
            logger.error("\nFull traceback:")
            logger.error(traceback.format_exc())
            logger.error("=" * 70)
            
            return []
    
    async def agenerate(self, function: Function, specification: str,
                       edge_cases: Optional[List[str]] = None, strength: str = "comprehensive") -> List[EnhancedPostcondition]:
        parameters_str = "\n".join([f"- {p.name}: {p.data_type}" for p in function.input_parameters])
        edge_cases_str = "\n".join(edge_cases or function.edge_cases or [])
        
        try:
            logger.info("=" * 70)
            logger.info("ASYNC POSTCONDITION GENERATION - DIAGNOSTICS")
            logger.info("=" * 70)
            logger.info(f"Function: {function.name}")
            logger.info(f"LLM Config: model={self.llm.model_name}, max_tokens={self.llm.max_tokens}, temp={self.llm.temperature}")
            logger.info(f"Timeout: {self.llm.request_timeout}s")
            
            import time
            start_time = time.time()
            
            logger.info("Sending async request to OpenAI...")
            result = await self.chain.ainvoke({
                "function_name": function.name,
                "function_signature": function.signature or f"{function.return_type} {function.name}(...)",
                "function_description": function.description,
                "parameters": parameters_str or "None",
                "return_type": function.return_type,
                "specification": specification,
                "edge_cases": edge_cases_str
            })
            
            elapsed = time.time() - start_time
            logger.info(f"Response received in {elapsed:.1f}s")
            
            self.last_raw_response = result
            
            logger.info(f"Response type: {type(result)}")
            if isinstance(result, list):
                logger.info(f"Response is list with {len(result)} items")
            
            postconditions = self._parse_postconditions(result)
            logger.info(f"Successfully parsed {len(postconditions)} postconditions")
            
            await self._fill_missing_translations(postconditions)
            logger.info("Filled missing translations")
            logger.info("=" * 70)
            
            return postconditions
            
        except Exception as e:
            self.last_error = e
            error_type = type(e).__name__
            logger.error("=" * 70)
            logger.error(f"ASYNC POSTCONDITION GENERATION FAILED: {error_type}")
            logger.error("=" * 70)
            logger.error(f"Error: {str(e)}")
            
            # Specific diagnostics
            if "timeout" in str(e).lower():
                logger.error("DIAGNOSIS: Timeout")
                logger.error("  - Current timeout: 180s")
                logger.error("  - Try: OPENAI_MODEL=gpt-4-turbo-preview")
            elif "rate" in str(e).lower():
                logger.error("DIAGNOSIS: Rate limit")
                logger.error("  - Wait 60 seconds and retry")
            elif "json" in str(e).lower():
                logger.error("DIAGNOSIS: JSON parsing failed")
                if self.last_raw_response:
                    logger.error(f"  - Response: {str(self.last_raw_response)[:300]}")
            
            import traceback
            logger.error("\nTraceback:")
            logger.error(traceback.format_exc())
            logger.error("=" * 70)
            
            return []
    
    def _parse_postconditions(self, result: Any) -> List[EnhancedPostcondition]:
        postconditions = []
        
        try:
            logger.info(f"Parsing postconditions from result type: {type(result)}")
            
            if not isinstance(result, list):
                logger.warning(f"Result is not a list, wrapping it: {type(result)}")
                result = [result]
            
            logger.info(f"Processing {len(result)} items")
            
            for i, pc_data in enumerate(result):
                try:
                    logger.info(f"Parsing postcondition {i+1}/{len(result)}")
                    
                    # Check for required fields
                    required = ['formal_text', 'natural_language']
                    missing = [f for f in required if f not in pc_data]
                    if missing:
                        logger.warning(f"  Missing required fields: {missing}")
                        logger.warning(f"  Available fields: {list(pc_data.keys())}")
                    
                    postcondition = EnhancedPostcondition(
                        formal_text=pc_data.get("formal_text", ""),
                        natural_language=pc_data.get("natural_language", ""),
                        strength=PostconditionStrength(pc_data.get("strength", "standard")),
                        category=PostconditionCategory(pc_data.get("category", "correctness")),
                        confidence_score=float(pc_data.get("confidence_score", 0.5)),
                        clarity_score=float(pc_data.get("clarity_score", 0.0)),
                        completeness_score=float(pc_data.get("completeness_score", 0.0)),
                        testability_score=float(pc_data.get("testability_score", 0.0)),
                        precise_translation=pc_data.get("precise_translation", ""),
                        reasoning=pc_data.get("reasoning", ""),
                        edge_cases_covered=pc_data.get("edge_cases_covered", []),
                        coverage_gaps=pc_data.get("coverage_gaps", []),
                        mathematical_validity=pc_data.get("mathematical_validity", ""),
                        robustness_assessment=pc_data.get("robustness_assessment", ""),
                        robustness_score=float(pc_data.get("robustness_score", 0.0)),
                        mathematical_quality_score=float(pc_data.get("mathematical_quality_score", 0.0)),
                        importance_category=pc_data.get("importance_category", ""),
                        organization_rank=int(pc_data.get("organization_rank", i + 1)),
                        is_primary_in_category=bool(pc_data.get("is_primary_in_category", False)),
                        recommended_for_selection=bool(pc_data.get("recommended_for_selection", True)),
                        selection_reasoning=pc_data.get("selection_reasoning", ""),
                        edge_cases=pc_data.get("edge_cases", []),
                        z3_theory=pc_data.get("z3_theory", "unknown"),
                        warnings=pc_data.get("warnings", [])
                    )
                    
                    postcondition.overall_priority_score = self._calculate_priority_score(postcondition)
                    postconditions.append(postcondition)
                    logger.info(f"  Successfully parsed postcondition {i+1}")
                    
                except Exception as e:
                    logger.error(f"  Failed to parse postcondition {i+1}: {e}")
                    logger.error(f"  Data keys: {list(pc_data.keys()) if isinstance(pc_data, dict) else 'not a dict'}")
                    continue
            
            logger.info(f"Parsing complete: {len(postconditions)}/{len(result)} successful")
        
        except Exception as e:
            logger.error(f"Failed to parse postconditions: {e}")
            logger.error(f"Result type: {type(result)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return postconditions
    
    def _calculate_priority_score(self, pc: EnhancedPostcondition) -> float:
        score = (pc.confidence_score * 0.25 + pc.robustness_score * 0.25 +
                 pc.clarity_score * 0.15 + pc.completeness_score * 0.15 +
                 pc.testability_score * 0.10 + pc.mathematical_quality_score * 0.10)
        return min(1.0, max(0.0, score))
    
    async def _fill_missing_translations(self, postconditions: List[EnhancedPostcondition]) -> None:
        translation_tasks = []
        indices_to_fill = []
        
        for i, pc in enumerate(postconditions):
            if not pc.precise_translation and pc.formal_text:
                indices_to_fill.append(i)
                translation_tasks.append(self.translator.atranslate(pc.formal_text))
        
        if translation_tasks:
            translations = await asyncio.gather(*translation_tasks)
            for idx, translation in zip(indices_to_fill, translations):
                postconditions[idx].precise_translation = translation


class Z3TranslationChain:
    """Translates postconditions to Z3 code."""
    
    def __init__(self, streaming: bool = False):
        self.llm = LLMFactory.create_llm(streaming=streaming, temperature=0.1, max_tokens=1500)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        system_template = """Translate to Z3 Python code. Use this structure:

from z3 import *
# Declare variables
x = Int('x')
arr = Array('arr', IntSort(), IntSort())
# Define constraint
constraint = ForAll([i], Implies(And(i >= 0, i < size), Select(arr, i) <= Select(arr, i+1)))
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")

Return ONLY Python code, no markdown."""

        human_template = """Formal: {formal_text}
Natural: {natural_language}
Context: {function_context}
Theory: {z3_theory}

Generate Z3 code:"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def translate(self, postcondition: EnhancedPostcondition, function_context: Optional[Dict[str, Any]] = None) -> Z3Translation:
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
    
    async def atranslate(self, postcondition: EnhancedPostcondition, function_context: Optional[Dict[str, Any]] = None) -> Z3Translation:
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
        if '```python' in result_text:
            code = result_text.split('```python')[1].split('```')[0]
        elif '```' in result_text:
            code = result_text.split('```')[1].split('```')[0]
        else:
            code = result_text
        return code.strip()
    
    def _validate_z3_code(self, translation: Z3Translation) -> None:
        import ast
        
        if not translation.z3_code:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "not_validated"
            return
        
        try:
            ast.parse(translation.z3_code)
            if 'from z3 import' not in translation.z3_code:
                translation.warnings.append("Missing Z3 import")
            if 'Solver()' not in translation.z3_code:
                translation.warnings.append("No Solver() instance")
            
            translation.z3_validation_passed = True
            translation.z3_validation_status = "success"
        except SyntaxError as e:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "syntax_error"
            translation.validation_error = str(e)
    
    def _format_function_context(self, context: Optional[Dict[str, Any]]) -> str:
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


class ChainFactory:
    """Factory for accessing all chains."""
    
    def __init__(self):
        self._pseudocode_chain = None
        self._postcondition_chain = None
        self._z3_chain = None
        self._translation_chain = None
    
    @property
    def pseudocode(self) -> PseudocodeChain:
        if self._pseudocode_chain is None:
            self._pseudocode_chain = PseudocodeChain()
        return self._pseudocode_chain
    
    @property
    def postcondition(self) -> PostconditionChain:
        if self._postcondition_chain is None:
            self._postcondition_chain = PostconditionChain()
        return self._postcondition_chain
    
    @property
    def z3(self) -> Z3TranslationChain:
        if self._z3_chain is None:
            self._z3_chain = Z3TranslationChain()
        return self._z3_chain
    
    @property
    def translator(self) -> FormalLogicTranslationChain:
        if self._translation_chain is None:
            self._translation_chain = FormalLogicTranslationChain()
        return self._translation_chain


if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLIFIED CHAINS - TEST")
    print("=" * 70)
    print("\nChanges:")
    print("  - Shorter prompts (no timeout)")
    print("  - Timeout increased to 180s")
    print("  - Temperature 0.4 for better variety")
    print("  - Still generates 6-10 postconditions")
    
    print("\nRun: python main.py --spec 'Reverse an array'")