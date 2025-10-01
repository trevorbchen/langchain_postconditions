"""
Simplified LangChain Chains - Enhanced with Rich Field Requests + Batch Z3 Translation

PHASE 1 CHANGES:
- Updated PostconditionChain prompt to request ALL rich fields
- Maintains timeout fix (prompt still concise)
- Keeps 6-10 postcondition requirement
- Adds precise_translation, reasoning, edge_cases_covered, etc.

PHASE 7 CHANGES (BATCHING):
- Added atranslate_batch() to Z3TranslationChain
- Added _create_batch_prompt() for batch translation
- Added _parse_batch_response() for parsing batch results
- Added _fallback_individual_translations() for error recovery
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
            request_timeout=180
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
{{
  "functions": [{{
    "name": "function_name",
    "description": "what it does",
    "signature": "return_type function_name(params)",
    "return_type": "void",
    "input_parameters": [{{"name": "param", "data_type": "int*", "description": "desc"}}],
    "output_parameters": [],
    "return_values": [{{"condition": "success", "value": "0", "description": "ok"}}],
    "preconditions": ["arr != NULL", "size > 0"],
    "edge_cases": ["Empty array", "NULL pointer", "Single element"],
    "complexity": "O(n)",
    "memory_usage": "O(1)",
    "body": "steps",
    "dependencies": []
  }}],
  "structs": [], "enums": [], "global_variables": [],
  "includes": ["stdio.h"], "dependencies": [], "metadata": {{}}
}}
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
        self.last_raw_response = None
        self.last_error = None
    
    def _create_prompt(self) -> ChatPromptTemplate:
        system_template = """Generate exactly 6-10 diverse postconditions as a JSON array.

REQUIRED FIELDS:
{{
  "formal_text": "∀i: mathematical notation",
  "natural_language": "brief explanation",
  "precise_translation": "detailed 2-3 sentences",
  "reasoning": "WHY this matters",
  "edge_cases_covered": ["edge1", "edge2"],
  "coverage_gaps": ["what NOT guaranteed"],
  "mathematical_validity": "assessment",
  "robustness_assessment": "evaluation",
  "strength": "standard",
  "category": "correctness",
  "confidence_score": 0.95,
  "clarity_score": 0.9,
  "completeness_score": 0.85,
  "testability_score": 0.9,
  "robustness_score": 0.92,
  "mathematical_quality_score": 0.96,
  "z3_theory": "arrays",
  "importance_category": "critical",
  "organization_rank": 1,
  "is_primary_in_category": true,
  "selection_reasoning": "why"
}}

Return ONLY JSON array."""

        human_template = """Function: {function_name}
Signature: {function_signature}
Description: {function_description}
Parameters: {parameters}
Return: {return_type}
Specification: {specification}
Edge Cases: {edge_cases}

Generate 6-10 DIVERSE postconditions as JSON array."""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def generate(self, function: Function, specification: str, 
                 edge_cases: Optional[List[str]] = None, strength: str = "comprehensive") -> List[EnhancedPostcondition]:
        parameters_str = "\n".join([f"- {p.name}: {p.data_type}" for p in function.input_parameters])
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
    
    async def agenerate(self, function: Function, specification: str,
                       edge_cases: Optional[List[str]] = None, strength: str = "comprehensive") -> List[EnhancedPostcondition]:
        parameters_str = "\n".join([f"- {p.name}: {p.data_type}" for p in function.input_parameters])
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
            await self._fill_missing_translations(postconditions)
            return postconditions
            
        except Exception as e:
            self.last_error = e
            logger.error(f"Async postcondition generation failed: {e}")
            return []
    
    def _parse_postconditions(self, result: Any) -> List[EnhancedPostcondition]:
        postconditions = []
        
        try:
            if not isinstance(result, list):
                result = [result]
            
            for i, pc_data in enumerate(result):
                try:
                    postcondition = EnhancedPostcondition(
                        formal_text=pc_data.get("formal_text", ""),
                        natural_language=pc_data.get("natural_language", ""),
                        precise_translation=pc_data.get("precise_translation", ""),
                        reasoning=pc_data.get("reasoning", ""),
                        edge_cases_covered=pc_data.get("edge_cases_covered", []),
                        coverage_gaps=pc_data.get("coverage_gaps", []),
                        edge_cases=pc_data.get("edge_cases", []),
                        mathematical_validity=pc_data.get("mathematical_validity", ""),
                        robustness_assessment=pc_data.get("robustness_assessment", ""),
                        confidence_score=float(pc_data.get("confidence_score", 0.5)),
                        clarity_score=float(pc_data.get("clarity_score", 0.0)),
                        completeness_score=float(pc_data.get("completeness_score", 0.0)),
                        testability_score=float(pc_data.get("testability_score", 0.0)),
                        robustness_score=float(pc_data.get("robustness_score", 0.0)),
                        mathematical_quality_score=float(pc_data.get("mathematical_quality_score", 0.0)),
                        strength=PostconditionStrength(pc_data.get("strength", "standard")),
                        category=PostconditionCategory(pc_data.get("category", "correctness")),
                        importance_category=pc_data.get("importance_category", ""),
                        organization_rank=int(pc_data.get("organization_rank", i + 1)),
                        is_primary_in_category=bool(pc_data.get("is_primary_in_category", False)),
                        recommended_for_selection=bool(pc_data.get("recommended_for_selection", True)),
                        selection_reasoning=pc_data.get("selection_reasoning", ""),
                        z3_theory=pc_data.get("z3_theory", "unknown"),
                        warnings=pc_data.get("warnings", [])
                    )
                    
                    postcondition.overall_priority_score = self._calculate_priority_score(postcondition)
                    postconditions.append(postcondition)
                    
                except Exception as e:
                    logger.error(f"Failed to parse postcondition {i+1}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to parse postconditions: {e}")
        
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
    """Translates postconditions to Z3 code with batching support."""
    
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
    
    def _create_batch_prompt(self) -> ChatPromptTemplate:
        """Prompt for batch Z3 translation (NEW IN PHASE 7)."""
        system_template = """Translate multiple formal postconditions to Z3 Python code.

Input: JSON array of postconditions
Output: JSON array of Z3 translations IN SAME ORDER

Each input:
- index: Number (preserve in output)
- formal_text: Mathematical logic
- natural_language: Description
- z3_theory: Theory to use

Each output:
- index: Same as input (CRITICAL)
- z3_code: Complete Python with imports
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

Context: {function_context}

Return JSON array:"""
        
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
            
            if translation.z3_code:
                translation.z3_code = self._add_validation_header(translation)
            
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
        function_context: Optional[Dict[str, Any]] = None
    ) -> List[Z3Translation]:
        """
        NEW IN PHASE 7: Batch translate multiple postconditions in ONE LLM call.
        
        Example: 10 postconditions = 1 call instead of 10 calls = 90% savings
        """
        if not postconditions:
            return []
        
        logger.info(f"🔄 Batch translating {len(postconditions)} postconditions")
        
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
            
            logger.info(f"✅ Batch translation complete: {len([t for t in translations if t.translation_success])}/{len(postconditions)} succeeded")
            
            return translations
        
        except Exception as e:
            logger.error(f"❌ Batch Z3 translation failed: {e}")
            logger.warning(f"⚠️ Falling back to individual translations")
            return await self._fallback_individual_translations(postconditions, function_context)
    
    def _parse_batch_response(self, result_text: str, postconditions: List[EnhancedPostcondition]) -> List[Z3Translation]:
        """Parse batch JSON response from LLM."""
        translations = []
        
        try:
            # Extract JSON array
            start = result_text.find('[')
            end = result_text.rfind(']') + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON array in response")
            
            json_str = result_text[start:end]
            batch_results = json.loads(json_str)
            
            if not isinstance(batch_results, list):
                raise ValueError("Response is not an array")
            
            # Map by index
            result_map = {item.get('index', i): item for i, item in enumerate(batch_results)}
            
            for i, pc in enumerate(postconditions):
                if i in result_map:
                    item = result_map[i]
                    z3_code = self._extract_code(item.get('z3_code', ''))
                    
                    translation = Z3Translation(
                        formal_text=pc.formal_text,
                        natural_language=pc.natural_language,
                        z3_code=z3_code,
                        z3_theory_used=pc.z3_theory or "arithmetic",
                        translation_success=bool(z3_code) and item.get('success', False)
                    )
                    
                    self._validate_z3_code(translation)
                    
                    if translation.z3_code:
                        translation.z3_code = self._add_validation_header(translation)
                    
                    translations.append(translation)
                else:
                    logger.warning(f"Missing result for index {i}")
                    translations.append(Z3Translation(
                        formal_text=pc.formal_text,
                        natural_language=pc.natural_language,
                        z3_code="",
                        translation_success=False,
                        validation_error="Missing from batch"
                    ))
            
            return translations
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Batch parse error: {e}")
            raise
    
    async def _fallback_individual_translations(
        self,
        postconditions: List[EnhancedPostcondition],
        function_context: Optional[Dict[str, Any]] = None
    ) -> List[Z3Translation]:
        """Fallback to individual translations if batch fails."""
        logger.info(f"🔄 Processing {len(postconditions)} individually")
        
        tasks = [self.atranslate(pc, function_context) for pc in postconditions]
        translations = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, t in enumerate(translations):
            if isinstance(t, Exception):
                logger.error(f"Individual {i} failed: {t}")
                results.append(Z3Translation(
                    formal_text=postconditions[i].formal_text,
                    natural_language=postconditions[i].natural_language,
                    z3_code="",
                    translation_success=False,
                    validation_error=str(t)
                ))
            else:
                results.append(t)
        
        success = sum(1 for r in results if r.translation_success)
        logger.info(f"✅ Individual translations: {success}/{len(postconditions)} succeeded")
        
        return results
    
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
            translation.validation_error = "No code generated"
            return
        
        try:
            ast.parse(translation.z3_code)
            translation.z3_validation_passed = True
            translation.z3_validation_status = "success"
        except SyntaxError as e:
            translation.z3_validation_passed = False
            translation.z3_validation_status = "syntax_error"
            translation.validation_error = f"Syntax error: {e.msg}"
    
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
    
    def _add_validation_header(self, translation: Z3Translation) -> str:
        status = "✅ VALIDATED" if translation.z3_validation_passed else "❌ FAILED"
        header = f"# {status}\n# Status: {translation.z3_validation_status}\n\n"
        return header + translation.z3_code


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
    print("PHASE 7 COMPLETE - Batch Z3 Translation Added")
    print("=" * 70)
    print("\n✅ Changes:")
    print("  - Added atranslate_batch() to Z3TranslationChain")
    print("  - Added _create_batch_prompt() for batch prompts")
    print("  - Added _parse_batch_response() for parsing")
    print("  - Added _fallback_individual_translations() for errors")
    print("\n📊 Expected Savings:")
    print("  - 10 postconditions: 10 calls → 1 call (90% reduction)")
    print("  - 24 postconditions: 24 calls → 3 calls (87% reduction)")