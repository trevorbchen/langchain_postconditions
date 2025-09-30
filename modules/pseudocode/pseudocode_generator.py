#!/usr/bin/env python3
"""
Refactored Pseudocode Generator using PromptsManager
Reduces code from 800+ lines to ~200 lines
"""

import openai
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

from utils.prompt_loader import PromptsManager

load_dotenv()
logger = logging.getLogger(__name__)


class PseudocodeGenerator:
    """Simplified pseudocode generator using external prompts."""
    
    def __init__(self, api_key: Optional[str] = None, prompts_file: str = "config/prompts.yaml"):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.prompts = PromptsManager(prompts_file)
        
    def generate(self, 
                 specification: str, 
                 context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate pseudocode from specification.
        
        Args:
            specification: Natural language description
            context: Optional context (codebase analysis, functions, structs)
            
        Returns:
            Structured pseudocode result
        """
        
        # Get prompt template from PromptsManager
        template = self.prompts.get_pseudocode_prompt()
        
        # Build context string
        context_str = self._build_context_string(context) if context else ""
        
        # Format prompt with variables
        formatted = template.format(
            specification=specification,
            context=context_str
        )
        
        system_prompt = formatted["system"]
        user_prompt = formatted["human"]
        
        # Call OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            result = response.choices[0].message.content
            return self._parse_result(result, specification)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._generate_fallback(specification)
    
    def _build_context_string(self, context: Dict) -> str:
        """Build context string from codebase analysis or direct specification."""
        parts = []
        
        if functions := context.get('function_names'):
            parts.append(f"Available functions: {', '.join(functions[:15])}")
            
        if structs := context.get('structs'):
            struct_names = [s.get('name', 'unknown') for s in structs[:5]]
            parts.append(f"Available structs: {', '.join(struct_names)}")
            
        if includes := context.get('includes'):
            parts.append(f"Common includes: {', '.join(includes[:8])}")
        
        if patterns := context.get('patterns'):
            parts.append(f"Common patterns: {', '.join(patterns[:5])}")
            
        return "\n".join(parts)
    
    def _parse_result(self, result: str, original_prompt: str) -> Dict[str, Any]:
        """Parse AI response into structured format."""
        try:
            # Extract JSON from response
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
                
            json_str = result[json_start:json_end]
            data = json.loads(json_str)
            
            # Add metadata
            if 'metadata' not in data:
                data['metadata'] = {}
                
            data['metadata'].update({
                'original_prompt': original_prompt,
                'generation_method': 'ai',
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'pseudocode': data,
                'functions': [f['name'] for f in data.get('functions', [])],
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return self._generate_fallback(original_prompt)
    
    def _generate_fallback(self, specification: str) -> Dict[str, Any]:
        """Generate minimal fallback pseudocode."""
        func_name = self._extract_function_name(specification)
        
        return {
            'success': False,
            'pseudocode': {
                'functions': [{
                    'name': func_name,
                    'description': f'Process: {specification}',
                    'input_parameters': [
                        {
                            'name': 'data',
                            'data_type': 'void*',
                            'description': 'from caller providing generic input data'
                        }
                    ],
                    'output_parameters': [],
                    'return_values': [
                        {
                            'name': 'result',
                            'data_type': 'int',
                            'description': 'returns 0 on success, -1 on error'
                        }
                    ],
                    'body_blocks': [],
                    'complexity': 'O(n)',
                    'memory_usage': 'O(1)'
                }],
                'structs': [],
                'enums': [],
                'global_variables': [],
                'includes': ['stdio.h', 'stdlib.h'],
                'metadata': {
                    'generation_method': 'fallback',
                    'original_prompt': specification
                }
            },
            'functions': [func_name],
            'error': 'AI generation failed, using fallback'
        }
    
    def _extract_function_name(self, specification: str) -> str:
        """Extract likely function name from specification."""
        spec_lower = specification.lower()
        
        if "sort" in spec_lower:
            return "sort_array"
        elif "search" in spec_lower or "find" in spec_lower:
            return "search_element"
        elif "reverse" in spec_lower:
            return "reverse_data"
        elif "calculate" in spec_lower or "compute" in spec_lower:
            return "calculate_result"
        elif "parse" in spec_lower:
            return "parse_input"
        elif "convert" in spec_lower:
            return "convert_data"
        else:
            return "process_data"


# Backward compatible API
def generate_pseudocode_api(prompt: str, 
                            codebase_path: Optional[str] = None,
                            api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Drop-in replacement for old generate_pseudocode_api.
    Note: codebase_path is accepted for backward compatibility but not used in refactored version.
    """
    generator = PseudocodeGenerator(api_key)
    
    # Simple context from just the codebase path
    context = None
    if codebase_path:
        logger.info(f"Codebase path provided: {codebase_path} (note: analysis requires original CodebaseAnalyzer)")
    
    return generator.generate(prompt, context)


def generate_with_context_api(prompt: str, 
                              codebase_path: Optional[str] = None,
                              available_functions: Optional[List[str]] = None,
                              available_structs: Optional[List[str]] = None,
                              includes: Optional[List[str]] = None,
                              api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced API with direct context specification.
    """
    generator = PseudocodeGenerator(api_key)
    
    # Build context from provided parameters
    context = None
    if available_functions or available_structs or includes:
        context = {
            'function_names': available_functions or [],
            'structs': [{'name': s} for s in (available_structs or [])],
            'includes': includes or [],
            'patterns': []
        }
    
    result = generator.generate(prompt, context)
    
    # Add additional metadata
    result['context_info'] = {
        'type': 'direct_specification',
        'has_codebase': codebase_path is not None,
        'has_functions': bool(available_functions),
        'has_structs': bool(available_structs)
    }
    
    return result