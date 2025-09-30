"""
LangChain Agents for Postcondition Generation

This module provides intelligent agents that can reason about which tools
to use and orchestrate complex workflows autonomously.

Key features:
- Agents that decide which chains to call
- Tool integration for all major operations
- Autonomous workflow orchestration
- Memory for context retention
"""

from typing import List, Optional, Dict, Any
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage

from core.chains import ChainFactory
from core.models import (
    Function,
    EnhancedPostcondition,
    Z3Translation,
    PseudocodeResult
)
from config.settings import settings


# ============================================================================
# TOOLS DEFINITION
# ============================================================================

class PostconditionTools:
    """
    Tool definitions for the postcondition agent.
    
    These tools wrap our chains and make them available to the agent.
    """
    
    def __init__(self):
        self.factory = ChainFactory()
    
    def create_tools(self) -> List[Tool]:
        """
        Create all tools for the agent.
        
        Returns:
            List of Tool objects
        """
        return [
            Tool(
                name="generate_pseudocode",
                description="""
                Use this tool to generate C pseudocode from a natural language specification.
                Input should be a clear description of what to implement.
                Returns structured pseudocode with functions, parameters, and complexity.
                
                Example input: "Sort an array using bubble sort"
                """,
                func=self._generate_pseudocode_sync
            ),
            
            Tool(
                name="generate_postconditions",
                description="""
                Use this tool to generate formal postconditions for a function.
                Input should be a JSON string with 'function_name' and 'specification'.
                Returns a list of postconditions with formal logic and natural language.
                
                Example input: {"function_name": "bubble_sort", "specification": "sort array"}
                """,
                func=self._generate_postconditions_sync
            ),
            
            Tool(
                name="translate_to_z3",
                description="""
                Use this tool to translate a formal postcondition to Z3 verification code.
                Input should be the formal postcondition text.
                Returns executable Z3 Python code.
                
                Example input: "∀i: arr[i] ≤ arr[i+1]"
                """,
                func=self._translate_to_z3_sync
            ),
            
            Tool(
                name="analyze_edge_cases",
                description="""
                Use this tool to identify edge cases for a specification.
                Input should be the specification or function description.
                Returns a list of edge cases to consider.
                
                Example input: "array sorting function"
                """,
                func=self._analyze_edge_cases_sync
            ),
        ]
    
    def _generate_pseudocode_sync(self, specification: str) -> str:
        """Generate pseudocode synchronously."""
        import asyncio
        result = asyncio.run(self.factory.pseudocode.agenerate(specification))
        return f"Generated {len(result.functions)} functions: {', '.join(result.function_names)}"
    
    def _generate_postconditions_sync(self, input_json: str) -> str:
        """Generate postconditions synchronously."""
        import json
        import asyncio
        
        try:
            data = json.loads(input_json)
            # Create a basic function for demo
            from core.models import Function, FunctionParameter
            func = Function(
                name=data.get('function_name', 'unknown'),
                description=data.get('specification', ''),
                return_type="void"
            )
            
            result = asyncio.run(
                self.factory.postcondition.agenerate(func, data.get('specification', ''))
            )
            return f"Generated {len(result)} postconditions"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _translate_to_z3_sync(self, formal_text: str) -> str:
        """Translate to Z3 synchronously."""
        import asyncio
        from core.models import EnhancedPostcondition, PostconditionCategory
        
        pc = EnhancedPostcondition(
            formal_text=formal_text,
            natural_language="Generated postcondition",
            category=PostconditionCategory.CORRECTNESS
        )
        
        result = asyncio.run(self.factory.z3.atranslate(pc))
        if result.translation_success:
            return f"Z3 translation successful. Code length: {len(result.z3_code)} chars"
        else:
            return f"Translation failed: {result.validation_error}"
    
    def _analyze_edge_cases_sync(self, specification: str) -> str:
        """Analyze edge cases synchronously."""
        # For demo - return common edge cases
        edge_cases = [
            "Empty input",
            "Null pointers",
            "Boundary values (0, max, min)",
            "Single element",
            "Duplicate values"
        ]
        return f"Identified edge cases: {', '.join(edge_cases)}"


# ============================================================================
# POSTCONDITION AGENT
# ============================================================================

class PostconditionAgent:
    """
    Intelligent agent for postcondition generation.
    
    The agent can:
    - Reason about which tools to use
    - Break down complex tasks
    - Remember context across interactions
    - Orchestrate multi-step workflows
    
    Example:
        >>> agent = PostconditionAgent()
        >>> result = agent.run("Generate postconditions for a sorting function")
        >>> print(result)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the postcondition agent.
        
        Args:
            verbose: Whether to print agent reasoning
        """
        self.verbose = verbose
        self.factory = ChainFactory()
        self.tools_manager = PostconditionTools()
        
        # Create tools
        self.tools = self.tools_manager.create_tools()
        
        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent."""
        
        # System prompt
        system_message = """You are an expert assistant for formal verification and postcondition generation.

Your role is to help users generate postconditions for C functions by:
1. Understanding their specifications
2. Generating pseudocode if needed
3. Creating formal postconditions
4. Translating to Z3 verification code
5. Identifying edge cases

You have access to several tools. Use them intelligently to accomplish tasks.

When generating postconditions:
- Be precise and mathematical
- Consider all edge cases
- Ensure completeness
- Validate your work

Always explain your reasoning and what you're doing."""

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_functions_agent(
            llm=self.factory.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    def run(self, task: str) -> str:
        """
        Run the agent on a task.
        
        Args:
            task: Task description for the agent
            
        Returns:
            Agent's response
            
        Example:
            >>> agent = PostconditionAgent()
            >>> result = agent.run("Create postconditions for bubble sort")
            >>> print(result)
        """
        try:
            result = self.agent_executor.invoke({"input": task})
            return result.get("output", "No output generated")
        except Exception as e:
            return f"Error executing task: {str(e)}"
    
    def chat(self, message: str) -> str:
        """
        Chat with the agent (maintains conversation history).
        
        Args:
            message: User message
            
        Returns:
            Agent's response
            
        Example:
            >>> agent = PostconditionAgent()
            >>> agent.chat("I need help with a sorting function")
            >>> agent.chat("Can you generate postconditions for it?")
        """
        return self.run(message)
    
    def reset(self) -> None:
        """
        Reset the agent's memory.
        
        Example:
            >>> agent = PostconditionAgent()
            >>> agent.chat("Hello")
            >>> agent.reset()  # Forgets previous conversation
        """
        self.memory.clear()


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class EdgeCaseAgent:
    """
    Specialized agent for edge case analysis.
    
    Focuses specifically on identifying and analyzing edge cases.
    """
    
    def __init__(self):
        self.factory = ChainFactory()
    
    def analyze(self, specification: str) -> List[str]:
        """
        Analyze edge cases for a specification.
        
        Args:
            specification: What to analyze
            
        Returns:
            List of edge cases
        """
        # Use the edge case chain
        from core.models import Function
        
        func = Function(
            name="unknown",
            description=specification,
            return_type="void"
        )
        
        return self.factory.edge_case.analyze(specification, func)


class OptimizationAgent:
    """
    Specialized agent for optimizing postconditions.
    
    Reviews and improves generated postconditions.
    """
    
    def __init__(self):
        self.factory = ChainFactory()
    
    def optimize(
        self,
        postconditions: List[EnhancedPostcondition]
    ) -> List[EnhancedPostcondition]:
        """
        Optimize a list of postconditions.
        
        Args:
            postconditions: Postconditions to optimize
            
        Returns:
            Optimized postconditions
        """
        # Filter low quality postconditions
        optimized = [
            pc for pc in postconditions
            if pc.overall_quality_score >= 0.7
        ]
        
        # Sort by quality
        optimized.sort(key=lambda x: x.overall_quality_score, reverse=True)
        
        return optimized


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_agent(verbose: bool = True) -> PostconditionAgent:
    """
    Convenience function to create an agent.
    
    Args:
        verbose: Whether to show agent reasoning
        
    Returns:
        PostconditionAgent instance
        
    Example:
        >>> agent = create_agent()
        >>> result = agent.run("Generate postconditions for merge sort")
    """
    return PostconditionAgent(verbose=verbose)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AGENTS - EXAMPLE USAGE")
    print("=" * 70)
    
    # Example 1: Basic agent usage
    print("\n🤖 Example 1: Create and use agent")
    print("-" * 70)
    
    agent = PostconditionAgent(verbose=False)
    
    task = "I need to generate postconditions for a function that sorts an array"
    print(f"Task: {task}")
    print("\nAgent response:")
    result = agent.run(task)
    print(result)
    
    # Example 2: Multi-turn conversation
    print("\n💬 Example 2: Multi-turn conversation")
    print("-" * 70)
    
    agent2 = PostconditionAgent(verbose=False)
    
    print("User: Tell me about edge cases for array operations")
    response1 = agent2.chat("Tell me about edge cases for array operations")
    print(f"Agent: {response1[:200]}...\n")
    
    print("User: What about sorting specifically?")
    response2 = agent2.chat("What about sorting specifically?")
    print(f"Agent: {response2[:200]}...")
    
    # Example 3: Edge case agent
    print("\n🔍 Example 3: Specialized edge case agent")
    print("-" * 70)
    
    edge_agent = EdgeCaseAgent()
    edge_cases = edge_agent.analyze("binary search algorithm")
    
    print("Edge cases for binary search:")
    for i, case in enumerate(edge_cases[:5], 1):
        print(f"  {i}. {case}")
    
    # Example 4: Optimization agent
    print("\n⚡ Example 4: Optimization agent")
    print("-" * 70)
    
    from core.models import PostconditionCategory
    
    # Create some test postconditions
    test_pcs = [
        EnhancedPostcondition(
            formal_text="x > 0",
            natural_language="x is positive",
            category=PostconditionCategory.CORRECTNESS,
            overall_quality_score=0.9
        ),
        EnhancedPostcondition(
            formal_text="y = z",
            natural_language="y equals z",
            category=PostconditionCategory.CORRECTNESS,
            overall_quality_score=0.5
        ),
        EnhancedPostcondition(
            formal_text="arr[i] < arr[j]",
            natural_language="array element i less than j",
            category=PostconditionCategory.CORRECTNESS,
            overall_quality_score=0.8
        ),
    ]
    
    opt_agent = OptimizationAgent()
    optimized = opt_agent.optimize(test_pcs)
    
    print(f"Original: {len(test_pcs)} postconditions")
    print(f"Optimized: {len(optimized)} high-quality postconditions")
    for pc in optimized:
        print(f"  - {pc.natural_language} (quality: {pc.overall_quality_score})")
    
    print("\n" + "=" * 70)
    print("✅ EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nNote: Agents require API calls. Set OPENAI_API_KEY to test fully.")