"""
Citation Agent - Traces case citation networks

Finds how cases have been cited, affirmed, distinguished, or overruled.
"""

from typing import Dict, Any, List
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from lex_bot.agents.base_agent import BaseAgent
from lex_bot.tools.indian_kanoon import IndianKanoonScraper

logger = logging.getLogger(__name__)


CITATION_PROMPT = """You are a legal research specialist analyzing citation networks in Indian case law.

**Case Being Analyzed:**
{case_name}

**Cases That Cite This Case:**
{citing_cases}

**Your Analysis:**

## Citation Summary
Provide an overview of how this case has been received by subsequent courts.

## Treatment Analysis
Categorize the citations:
- **Followed/Approved**: Cases that followed this precedent
- **Distinguished**: Cases that acknowledged but distinguished this case
- **Questioned/Doubted**: Cases that questioned the reasoning
- **Overruled**: If the case has been overruled (IMPORTANT - note by which case)

## Key Observations
- Is this case still good law?
- Which courts have cited it most?
- Any recent citations that affect its authority?

## Practitioner Note
Brief advice for lawyers relying on this precedent."""


class CitationAgent(BaseAgent):
    """
    Citation Network Agent for tracing case precedents.
    
    Features:
    - Finds cases that cite a given case
    - Analyzes how the case has been treated
    - Identifies if case is still good law
    """
    
    def __init__(self, mode: str = "reasoning"):
        """Initialize with reasoning mode for analysis."""
        super().__init__(mode=mode)
        self.kanoon = IndianKanoonScraper()
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute citation analysis workflow.
        
        Args:
            state: AgentState dictionary
            
        Returns:
            Updated state with citation analysis
        """
        query = state.get("original_query", "")
        
        logger.info(f"🔗 CitationAgent analyzing: {query[:50]}...")
        
        # Extract case name from query
        case_name = self._extract_case_name(query)
        
        # Search for the case
        case_results = self.kanoon.search(case_name, max_results=5)
        
        if not case_results:
            return {
                "final_answer": f"Could not find case: {case_name}. Please provide the exact case name or citation.",
                "errors": [f"Case not found: {case_name}"]
            }
        
        # Use first result as the main case
        main_case = case_results[0]
        
        # Search for citing cases
        citing_query = f'"{case_name}" cited'
        citing_results = self.kanoon.search(citing_query, max_results=10)
        
        # Format citing cases
        citing_context = self._format_citing_cases(citing_results)
        
        # Generate analysis
        prompt = ChatPromptTemplate.from_template(CITATION_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            analysis = chain.invoke({
                "case_name": main_case.get('title', case_name),
                "citing_cases": citing_context
            })
        except Exception as e:
            logger.error(f"Citation analysis failed: {e}")
            analysis = f"Citation analysis failed: {e}"
        
        return {
            "final_answer": analysis,
            "case_context": citing_results,
            "tool_results": [{
                "agent": "citation",
                "type": "citation_network",
                "main_case": main_case,
                "citing_count": len(citing_results)
            }]
        }
    
    def _extract_case_name(self, query: str) -> str:
        """Extract case name from query."""
        # Remove common phrases
        import re
        query = re.sub(r'(citation|citations|cited|cite|how|has|been|of|the|case)\s*', '', query, flags=re.IGNORECASE)
        return query.strip()
    
    def _format_citing_cases(self, cases: List[Dict]) -> str:
        """Format citing cases for analysis."""
        if not cases:
            return "No citing cases found in the search results."
        
        parts = []
        for i, case in enumerate(cases, 1):
            title = case.get('title', 'Unknown')
            court = case.get('court', '')
            date = case.get('date', '')
            snippet = case.get('snippet', '')[:300]
            
            parts.append(
                f"{i}. **{title}**\n"
                f"   Court: {court or 'Not specified'} | Date: {date or 'Unknown'}\n"
                f"   Context: {snippet}"
            )
        
        return "\n\n".join(parts)


# Singleton
citation_agent = CitationAgent()
