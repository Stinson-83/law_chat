"""
Query Router - Classifies queries as Simple or Complex

Simple queries: Direct fact lookups, single statute/section, definitions
Complex queries: Multi-faceted legal questions, strategy, comparative analysis
"""

from typing import Literal, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from lex_bot.core.llm_factory import get_llm


ROUTER_PROMPT = """You are a legal query classifier for an Indian law research system.

Analyze the user's query and classify it as either SIMPLE or COMPLEX.

**SIMPLE queries** are:
- Direct fact lookups (e.g., "What is Section 302 IPC?")
- Single statute or section queries
- Definition requests (e.g., "Define cognizable offense")
- Basic legal term explanations
- Latin phrase meanings

**COMPLEX queries** are:
- Multi-faceted legal questions requiring analysis
- Strategy or argument requests (e.g., "What are defense options for...")
- Comparative analysis (e.g., "Compare IPC 304A vs 304B")
- Case law research with multiple aspects
- Questions requiring synthesis from multiple sources
- Questions about legal procedures with multiple steps

User Query: {query}

Respond in JSON format:
{{
    "complexity": "simple" or "complex",
    "reasoning": "Brief explanation of why this classification",
    "suggested_agents": ["list of agent types that should handle this"]
}}

Agent types available: ["research", "law", "case", "strategy", "citation", "explainer"]
- research: For simple RAG + web search
- law: For statutes, acts, sections
- case: For case law and precedents
- strategy: For legal strategy and arguments
- citation: For citation network analysis
- explainer: For simplifying concepts for students
"""


class QueryRouter:
    """
    Routes queries to appropriate agents based on complexity.
    
    Usage:
        router = QueryRouter()
        result = router.classify("What is Section 302 IPC?")
        # Returns: {"complexity": "simple", "reasoning": "...", "suggested_agents": ["research"]}
    """
    
    def __init__(self, mode: Literal["fast", "reasoning"] = "fast"):
        """
        Initialize router with specified LLM mode.
        
        Args:
            mode: LLM mode to use. "fast" recommended for routing decisions.
        """
        self.llm = get_llm(mode=mode)
        self.prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
        self.chain = self.prompt | self.llm | JsonOutputParser()
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify a query as simple or complex.
        
        Args:
            query: User's legal query
            
        Returns:
            Dict with keys: complexity, reasoning, suggested_agents
        """
        try:
            result = self.chain.invoke({"query": query})
            
            # Validate and normalize
            if result.get("complexity") not in ["simple", "complex"]:
                result["complexity"] = "simple"  # Default to simple
            
            if not result.get("suggested_agents"):
                result["suggested_agents"] = ["research"]
            
            return result
            
        except Exception as e:
            print(f"⚠️ Router classification failed: {e}")
            # Fallback: treat as simple
            return {
                "complexity": "simple",
                "reasoning": f"Fallback due to error: {e}",
                "suggested_agents": ["research"]
            }
    
    def is_complex(self, query: str) -> bool:
        """Quick check if query is complex."""
        return self.classify(query).get("complexity") == "complex"


# Singleton for convenience
_router_instance = None

def get_router(mode: Literal["fast", "reasoning"] = "fast") -> QueryRouter:
    """Get or create a QueryRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = QueryRouter(mode=mode)
    return _router_instance
