#!/usr/bin/env python3
"""
Refactored Postcondition Generator using PromptsManager
Reduces code from 3000+ lines to ~300 lines
"""

import openai
from typing import Dict, List, Optional, Any
import logging
import os
from dotenv import load_dotenv
import json
import sqlite3
import sys
from pathlib import Path

# Get the project root directory (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.prompt_loader import PromptsManager

load_dotenv()
logger = logging.getLogger(__name__)


class PostconditionGenerator:
    """Simplified postcondition generator using external prompts."""
    
    def __init__(self, 
                 context_db_path: str = "context.db",
                 api_key: Optional[str] = None,
                 prompts_file: str = "config/prompts.yaml"):
        
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.prompts = PromptsManager(prompts_file)
        self.context_db_path = context_db_path
        
        # Load domain knowledge once
        self.domain_knowledge = self._load_domain_knowledge()
    
    def generate(self,
                 specification: str,
                 function: Dict[str, Any],
                 edge_cases: Optional[List[str]] = None,
                 strength: str = "standard") -> List[Dict[str, Any]]:
        """
        Generate postconditions for a function.
        
        Args:
            specification: Original natural language specification
            function: Pseudocode function dictionary
            edge_cases: Optional list of edge cases to address
            strength: "minimal", "standard", or "comprehensive"
            
        Returns:
            List of postcondition dictionaries
        """
        
        # Get prompt template using PromptsManager
        template = self.prompts.get_postcondition_prompt()
        
        # Build comprehensive context
        func_context = self._build_function_context(function)
        edge_context = self._build_edge_case_context(edge_cases or [])
        domain_context = self._build_domain_context(specification)
        
        # Format the prompt using template.format()
        formatted = template.format(
            specification=specification,
            function_context=func_context,
            variable_context=self._format_variables(function),
            domain_knowledge=domain_context,
            edge_case_analysis=edge_context
        )
        
        system_prompt = formatted["system"]
        user_prompt = formatted["human"]
        
        # Generate postconditions
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            return self._parse_postconditions(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Postcondition generation failed: {e}")
            return self._generate_fallback_postconditions(function)
    
    def _build_function_context(self, function: Dict) -> str:
        """Build function context using PromptsManager helper."""
        return self.prompts.build_function_context(
            name=function.get('name', 'unknown'),
            description=function.get('description', ''),
            input_params=function.get('input_parameters', []),
            output_params=function.get('output_parameters', []),
            return_values=function.get('return_values', []),
            preconditions=function.get('preconditions', [])
        )
    
    def _build_edge_case_context(self, edge_cases: List[str]) -> str:
        """Build edge case context."""
        if not edge_cases:
            edge_case_dict = {
                'input_edge_cases': ['Standard input assumed'],
                'output_edge_cases': [],
                'algorithmic_edge_cases': [],
                'mathematical_edge_cases': [],
                'boundary_conditions': [],
                'error_conditions': [],
                'performance_edge_cases': [],
                'domain_specific_cases': [],
                'coverage_score': 0.5,
                'completeness_assessment': 'Minimal edge case analysis'
            }
        else:
            # Organize edge cases into categories
            edge_case_dict = {
                'input_edge_cases': edge_cases[:3],
                'output_edge_cases': edge_cases[3:5] if len(edge_cases) > 3 else [],
                'algorithmic_edge_cases': edge_cases[5:7] if len(edge_cases) > 5 else [],
                'mathematical_edge_cases': [],
                'boundary_conditions': [],
                'error_conditions': [],
                'performance_edge_cases': [],
                'domain_specific_cases': [],
                'coverage_score': min(len(edge_cases) / 10.0, 1.0),
                'completeness_assessment': f'{len(edge_cases)} edge cases identified'
            }
        
        return self.prompts.build_edge_case_context(edge_case_dict)
    
    def _build_domain_context(self, specification: str) -> str:
        """Build domain context using PromptsManager."""
        domain = self._infer_domain(specification)
        return self.prompts.build_domain_context(domain)
    
    def _format_variables(self, function: Dict) -> str:
        """Format variable context for prompt."""
        inputs = function.get('input_parameters', [])
        outputs = function.get('output_parameters', [])
        returns = function.get('return_values', [])
        
        parts = []
        if inputs:
            parts.append(f"Inputs: {', '.join([p['name'] for p in inputs])}")
        if outputs:
            parts.append(f"Outputs: {', '.join([p['name'] for p in outputs])}")
        if returns:
            parts.append(f"Returns: {', '.join([r.get('name', 'result') for r in returns])}")
        
        return "; ".join(parts)
    
    def _parse_postconditions(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI response into postcondition list."""
        try:
            # Extract JSON array
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1:
                raise ValueError("No JSON array found")
            
            postconditions = json.loads(response[json_start:json_end])
            
            # Validate structure
            for pc in postconditions:
                if 'formal_text' not in pc or 'natural_language_explanation' not in pc:
                    raise ValueError("Missing required fields in postcondition")
            
            return postconditions
            
        except Exception as e:
            logger.error(f"Failed to parse postconditions: {e}")
            return []
    
    def _load_domain_knowledge(self) -> Dict[str, Dict]:
        """Load domain knowledge from database."""
        knowledge = {}
        try:
            conn = sqlite3.connect(self.context_db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT domain, patterns, examples FROM domain_contexts")
            for row in cursor.fetchall():
                domain, patterns, examples = row
                knowledge[domain] = {
                    'patterns': json.loads(patterns) if patterns else [],
                    'examples': json.loads(examples) if examples else []
                }
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not load domain knowledge: {e}")
        
        return knowledge
    
    def _infer_domain(self, specification: str) -> str:
        """Infer domain from specification text."""
        spec_lower = specification.lower()
        
        # Get all available domains from PromptsManager
        available_domains = self.prompts.get_all_domains()
        
        # Map keywords to domains
        domain_keywords = {
            'collections': ['sort', 'array', 'list', 'queue', 'stack', 'tree'],
            'algorithms': ['search', 'find', 'binary', 'traverse', 'dfs', 'bfs'],
            'strings': ['parse', 'string', 'text', 'concat', 'substring'],
            'numerical': ['calculate', 'compute', 'sum', 'average', 'min', 'max'],
            'graphs': ['graph', 'node', 'edge', 'path', 'cycle']
        }
        
        # Check which domain matches best
        for domain, keywords in domain_keywords.items():
            if domain in available_domains:
                if any(word in spec_lower for word in keywords):
                    return domain
        
        # Default to first available domain or 'collections'
        return available_domains[0] if available_domains else 'collections'
    
    def _generate_fallback_postconditions(self, function: Dict) -> List[Dict]:
        """Generate minimal fallback postconditions."""
        return [{
            'formal_text': 'result != NULL',
            'natural_language_explanation': 'The function returns a valid result',
            'precise_translation': 'The result is not null',
            'strength': 'minimal',
            'confidence_score': 0.5,
            'z3_theory': 'unknown'
        }]


# ============================================================================
# CONVENIENCE FUNCTIONS (for backward compatibility with __init__.py)
# ============================================================================

def generate_postconditions(
    specification: str,
    function: Dict[str, Any],
    edge_cases: Optional[List[str]] = None,
    strength: str = "standard"
) -> List[Dict[str, Any]]:
    """
    Convenience function for generating postconditions.
    
    Args:
        specification: Natural language specification
        function: Pseudocode function dictionary
        edge_cases: Optional list of edge cases
        strength: "minimal", "standard", or "comprehensive"
        
    Returns:
        List of postcondition dictionaries
        
    Example:
        >>> postconditions = generate_postconditions(
        ...     "Sort an array",
        ...     {"name": "bubble_sort", "input_parameters": [...]}
        ... )
    """
    generator = PostconditionGenerator()
    return generator.generate(specification, function, edge_cases, strength)


def generate_postconditions_batch(
    specifications: List[str],
    functions: List[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    """
    Batch generate postconditions for multiple functions.
    
    Args:
        specifications: List of specifications
        functions: List of function dictionaries
        
    Returns:
        List of postcondition lists
        
    Example:
        >>> results = generate_postconditions_batch(
        ...     ["Sort array", "Search array"],
        ...     [func1, func2]
        ... )
    """
    generator = PostconditionGenerator()
    results = []
    
    for spec, func in zip(specifications, functions):
        postconditions = generator.generate(spec, func)
        results.append(postconditions)
    
    return results


# Backward compatible API
def generate_postconditions_api(specification: str,
                                function_dict: Dict,
                                context_db: str = "context.db",
                                api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Drop-in replacement for old API.
    """
    generator = PostconditionGenerator(context_db, api_key)
    postconditions = generator.generate(specification, function_dict)
    
    return {
        'success': len(postconditions) > 0,
        'postconditions': postconditions,
        'count': len(postconditions),
        'error': None if postconditions else 'Generation failed'
    }