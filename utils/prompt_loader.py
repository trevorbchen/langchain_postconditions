"""
Prompt loader and template manager for the postcondition generation system.

This module loads the comprehensive prompts.yaml file and provides easy access
to all templates with variable interpolation.
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
load_dotenv(project_root / ".env")

@dataclass
class PromptTemplate:
    """Represents a loaded prompt template."""
    system: str
    human: str
    
    def format(self, **kwargs) -> Dict[str, str]:
        """Format the template with provided variables."""
        return {
            "system": self.system.format(**kwargs),
            "human": self.human.format(**kwargs)
        }


class PromptsManager:
    """
    Manages all prompt templates from prompts.yaml.
    
    This replaces the scattered prompt strings in your original code with
    a centralized, maintainable prompt management system.
    
    Example:
        >>> prompts = PromptsManager()
        >>> template = prompts.get_pseudocode_prompt()
        >>> formatted = template.format(specification="sort array", context="")
        >>> print(formatted['system'])
    """
    
    def __init__(self, prompts_file: str = "config/prompts.yaml"):
        # Handle both absolute and relative paths
        self.prompts_file = Path(prompts_file)
        if not self.prompts_file.is_absolute():
            self.prompts_file = project_root / self.prompts_file
        
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        if not self.prompts_file.exists():
            raise FileNotFoundError(
                f"Prompts file not found: {self.prompts_file}\n"
                f"Please ensure prompts.yaml exists in the config directory."
            )
        
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    # ========================================================================
    # CORE PROMPT TEMPLATES
    # ========================================================================
    
    def get_pseudocode_prompt(self) -> PromptTemplate:
        """Get the pseudocode generation prompt template."""
        data = self.prompts['pseudocode_generation']
        return PromptTemplate(
            system=data['system'],
            human=data['human']
        )
    
    def get_postcondition_prompt(self) -> PromptTemplate:
        """Get the postcondition generation prompt template."""
        data = self.prompts['postcondition_generation']
        return PromptTemplate(
            system=data['system'],
            human=data['human']
        )
    
    def get_edge_case_prompt(self) -> PromptTemplate:
        """Get the edge case analysis prompt template."""
        data = self.prompts['edge_case_analysis']
        return PromptTemplate(
            system=data['system'],
            human=data['human']
        )
    
    def get_z3_translation_prompt(self) -> PromptTemplate:
        """Get the Z3 translation prompt template."""
        data = self.prompts['z3_translation']
        return PromptTemplate(
            system=data['system'],
            human=data['human']
        )
    
    # ========================================================================
    # DOMAIN KNOWLEDGE ACCESS
    # ========================================================================
    
    def get_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """
        Get domain-specific knowledge.
        
        Args:
            domain: Domain name (collections, numerical, strings, graphs, algorithms)
        
        Returns:
            Dictionary with domain patterns, edge cases, and verification approaches
        """
        domain_data = self.prompts['domain_knowledge'].get(domain, {})
        if not domain_data:
            raise ValueError(f"Unknown domain: {domain}")
        return domain_data
    
    def get_all_domains(self) -> List[str]:
        """Get list of all available domains."""
        return list(self.prompts['domain_knowledge'].keys())
    
    def get_domain_patterns(self, domain: str) -> List[str]:
        """Get common patterns for a domain."""
        return self.get_domain_knowledge(domain).get('common_patterns', [])
    
    def get_domain_edge_cases(self, domain: str) -> List[str]:
        """Get edge cases for a domain."""
        return self.get_domain_knowledge(domain).get('edge_cases', [])
    
    def get_domain_verification_approaches(self, domain: str) -> List[str]:
        """Get verification approaches for a domain."""
        return self.get_domain_knowledge(domain).get('verification_approaches', [])
    
    # ========================================================================
    # CONTEXT BUILDING
    # ========================================================================
    
    def build_function_context(
        self,
        name: str,
        description: str,
        input_params: List[Dict[str, Any]],
        output_params: List[Dict[str, Any]],
        return_values: List[Dict[str, Any]],
        preconditions: List[str]
    ) -> str:
        """Build function signature context for prompt."""
        template = self.prompts['context_building']['function_signature_template']
        
        # Format parameters
        input_str = "\n".join([
            f"  - {p['name']}: {p['data_type']} - {p['description']}"
            for p in input_params
        ])
        
        output_str = "\n".join([
            f"  - {p['name']}: {p['data_type']} - {p['description']}"
            for p in output_params
        ])
        
        return_str = "\n".join([
            f"  - Condition: {r['condition']}, Value: {r['value']} - {r['description']}"
            for r in return_values
        ])
        
        precond_str = "\n".join([f"  - {p}" for p in preconditions])
        
        input_names = ", ".join([p['name'] for p in input_params])
        output_names = ", ".join([p['name'] for p in output_params])
        return_names = ", ".join([r.get('name', 'result') for r in return_values])
        
        return template.format(
            name=name,
            description=description,
            input_params=input_str,
            output_params=output_str,
            return_values=return_str,
            preconditions=precond_str,
            input_names=input_names,
            output_names=output_names,
            return_names=return_names
        )
    
    def build_edge_case_context(self, edge_case_analysis: Dict[str, Any]) -> str:
        """Build edge case analysis context for prompt."""
        template = self.prompts['context_building']['edge_case_template']
        
        def format_cases(cases: List[str]) -> str:
            return "\n".join([f"  • {case}" for case in cases])
        
        return template.format(
            count=len(edge_case_analysis.get('input_edge_cases', [])),
            input_cases=format_cases(edge_case_analysis.get('input_edge_cases', [])),
            output_cases=format_cases(edge_case_analysis.get('output_edge_cases', [])),
            algorithmic_cases=format_cases(edge_case_analysis.get('algorithmic_edge_cases', [])),
            mathematical_cases=format_cases(edge_case_analysis.get('mathematical_edge_cases', [])),
            boundary_cases=format_cases(edge_case_analysis.get('boundary_conditions', [])),
            error_cases=format_cases(edge_case_analysis.get('error_conditions', [])),
            performance_cases=format_cases(edge_case_analysis.get('performance_edge_cases', [])),
            domain_cases=format_cases(edge_case_analysis.get('domain_specific_cases', [])),
            coverage_score=edge_case_analysis.get('coverage_score', 0.0),
            completeness=edge_case_analysis.get('completeness_assessment', 'Not assessed')
        )
    
    def build_domain_context(self, domain: str) -> str:
        """Build domain knowledge context for prompt."""
        template = self.prompts['context_building']['domain_knowledge_template']
        knowledge = self.get_domain_knowledge(domain)
        
        def format_list(items: List[str]) -> str:
            return "\n".join([f"  • {item}" for item in items])
        
        return template.format(
            domain=domain,
            patterns=format_list(knowledge.get('common_patterns', [])),
            examples=format_list(knowledge.get('examples', [])),
            ambiguities=format_list(knowledge.get('ambiguities', [])),
            edge_cases=format_list(knowledge.get('edge_cases', [])),
            verification_approaches=format_list(knowledge.get('verification_approaches', []))
        )
    
    def build_z3_theory_guidance(
        self,
        recommended_theory: str,
        reason: str
    ) -> str:
        """Build Z3 theory guidance context."""
        template = self.prompts['context_building']['z3_theory_template']
        return template.format(
            recommended_theory=recommended_theory,
            reason=reason
        )
    
    def build_full_context(
        self,
        function_context: str,
        edge_case_context: str,
        domain_context: str,
        z3_guidance: str,
        examples: str = ""
    ) -> str:
        """Build complete context for postcondition generation."""
        template = self.prompts['context_building']['full_context_template']
        return template.format(
            function_signature=function_context,
            edge_case_analysis=edge_case_context,
            domain_knowledge=domain_context,
            z3_theory_guidance=z3_guidance,
            examples=examples
        )
    
    # ========================================================================
    # VALIDATION PROMPTS
    # ========================================================================
    
    def get_postcondition_quality_prompt(self) -> str:
        """Get prompt for validating postcondition quality."""
        return self.prompts['validation']['postcondition_quality_check']
    
    def get_z3_code_quality_prompt(self) -> str:
        """Get prompt for validating Z3 code quality."""
        return self.prompts['validation']['z3_code_quality_check']
    
    # ========================================================================
    # EXAMPLES ACCESS
    # ========================================================================
    
    def get_example(self, example_name: str) -> Dict[str, Any]:
        """Get a complete example."""
        return self.prompts['examples'].get(example_name, {})
    
    def get_all_examples(self) -> List[str]:
        """Get names of all available examples."""
        return list(self.prompts['examples'].keys())


# ============================================================================
# SIMPLE TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing PromptsManager...")
    
    try:
        prompts = PromptsManager()
        print(f"✅ Loaded prompts from: {prompts.prompts_file}")
        
        # Test getting a prompt
        template = prompts.get_pseudocode_prompt()
        print(f"✅ Got pseudocode prompt: {len(template.system)} chars")
        
        # Test domains
        domains = prompts.get_all_domains()
        print(f"✅ Found {len(domains)} domains: {domains}")
        
        # Test formatting
        formatted = template.format(specification="test", context="")
        print(f"✅ Formatted prompt: {len(formatted['system'])} chars")
        
        print("\n✅ All tests passed!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure prompts.yaml exists in config/")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()