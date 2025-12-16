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
        
        # 3. Search (DB -> Web Fallback handled in tool)
        context_str, results = search_tool.run(enhanced_query, domains)
        
        # 4. Rerank
        reranked = rerank_documents(query, results, top_n=10)
        
        # 5. Return update
        return {"law_context": reranked} 

law_agent = LawAgent()
