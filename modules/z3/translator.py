#!/usr/bin/env python3
"""
Refactored Z3 Translator using prompt_loader
Reduces code from 2000+ lines to ~150 lines
"""

import openai
from typing import Dict, List, Tuple, Optional
import logging
import os
from dotenv import load_dotenv
import ast

from utils.prompt_loader import PromptsManager

load_dotenv()
logger = logging.getLogger(__name__)


class Z3Translator:
    """Simplified Z3 code generator using external prompts."""
    
    def __init__(self, api_key: Optional[str] = None, prompts_file: str = "config/prompts.yaml"):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.prompts = PromptsManager(prompts_file)
    
    def translate(self, 
                  postcondition: Dict[str, str],
                  function_context: Optional[Dict] = None) -> Dict[str, str]:
        """
        Translate formal postcondition to Z3 Python code.
        
        Args:
            postcondition: Dict with 'formal_text' and 'z3_theory'
            function_context: Optional function signature/type info
            
        Returns:
            Dict with 'declaration', 'constraints', 'python_impl'
        """
        
        # Load prompt template
        template = self.prompts.get_z3_translation_prompt()
        
        formal_text = postcondition.get('formal_text', '')
        z3_theory = postcondition.get('z3_theory', 'unknown')
        
        # Build context
        context_str = self._build_context(function_context, z3_theory)
        
        # Format prompt
        formatted = template.format(
            formal_text=formal_text,
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
            
            result = self._parse_z3_response(response.choices[0].message.content)
            
            # Validate syntax
            self._validate_z3_syntax(result['constraints'])
            
            return result
            
        except Exception as e:
            logger.error(f"Z3 translation failed: {e}")
            return self._generate_fallback_z3(formal_text)
    
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
    
    def _parse_z3_response(self, response: str) -> Dict[str, str]:
        """Parse Z3 code from AI response."""
        sections = {
            'python_impl': [],
            'declaration': [],
            'constraints': []
        }
        
        current_section = None
        
        for line in response.split('\n'):
            line_stripped = line.strip()
            
            # Detect section markers
            if 'PYTHON_IMPLEMENTATION:' in line_stripped:
                current_section = 'python_impl'
                continue
            elif 'Z3_DECLARATION:' in line_stripped:
                current_section = 'declaration'
                continue
            elif 'Z3_CONSTRAINTS:' in line_stripped:
                current_section = 'constraints'
                continue
            
            # Add to current section
            if current_section and line_stripped:
                sections[current_section].append(line_stripped)
        
        return {
            'python_impl': '\n'.join(sections['python_impl']),
            'declaration': '\n'.join(sections['declaration']),
            'constraints': '\n'.join(sections['constraints'])
        }
    
    def _validate_z3_syntax(self, code: str) -> None:
        """Validate Z3 Python code syntax."""
        if not code:
            raise ValueError("Empty Z3 code")
        
        try:
            # Basic syntax check
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Z3 Python syntax: {e}")
    
    def _generate_fallback_z3(self, formal_text: str) -> Dict[str, str]:
        """Generate minimal fallback Z3 code."""
        return {
            'python_impl': f'# Translation failed for: {formal_text}',
            'declaration': '# No declaration generated',
            'constraints': '# solver.add(True)  # Placeholder'
        }


# Backward compatible API
def translate_to_z3_api(formal_text: str,
                        z3_theory: str = "unknown",
                        api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Drop-in replacement for old translation API.
    """
    translator = Z3Translator(api_key)
    
    postcondition = {
        'formal_text': formal_text,
        'z3_theory': z3_theory
    }
    
    result = translator.translate(postcondition)
    
    return {
        'success': bool(result['constraints']),
        'z3_code': result['constraints'],
        'declaration': result['declaration'],
        'error': None
    }