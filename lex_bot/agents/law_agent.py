from typing import Dict, Any
from .base_agent import BaseAgent
from ..tools.db_search import search_tool
from ..tools.reranker import rerank_documents
from ..config import PREFERRED_DOMAINS

class LawAgent(BaseAgent):
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the Law Agent workflow.
        """
        query = state.get("law_query")
        if not query:
            return {"law_context": []}
            
        print(f"⚖️ Law Agent Processing: {query}")
        
        # 1. Enhance Query
        enhanced_query = self.enhance_query(query, "law")
        print(f"   Enhanced: {enhanced_query}")
        
        # 2. Define Domains (Defaulting to config's preferred, or custom logic if needed)
        domains = PREFERRED_DOMAINS 
        
        # 3. Dynamic Tool Execution via Registry
        from lex_bot.core.tool_registry import tool_registry
        
        # Get all tools capable of "statute_lookup" or "law_search"
        # For now, we prioritize "statute_lookup"
        tools = tool_registry.get_by_capability("statute_lookup")
        if not tools:
            print("⚠️ No law search tools found in registry!")
            tools = [search_tool]
            
        all_results = []
        combined_context = ""
        
        for tool in tools:
            try:
                print(f"   🔧 Running {tool.__class__.__name__}...")
                
                # Handle different tool signatures
                if tool.__class__.__name__ in ["SearchTool", "WebSearchTool"]:
                    res = tool.run(enhanced_query, domains)
                else:
                    # Tools like PenalCodeTool, LatinPhrasesTool only take query
                    res = tool.run(enhanced_query)
                    
                if isinstance(res, tuple):
                    combined_context += res[0] + "\n\n"
                    all_results.extend(res[1])
                elif isinstance(res, list):
                    all_results.extend(res)
                elif isinstance(res, str):
                    combined_context += res + "\n\n"
            except Exception as e:
                print(f"   ❌ Tool {tool.__class__.__name__} failed: {e}")
        
        # 4. Rerank
        reranked = rerank_documents(query, all_results, top_n=10)
        
        # 5. Return update
        return {"law_context": reranked} 

law_agent = LawAgent()
