from typing import Dict, Any
from .base_agent import BaseAgent
from ..tools.web_search import web_search_tool
from ..tools.reranker import rerank_documents
from ..config import TARGET_CASE_SITE, PREFERRED_DOMAINS

class CaseAgent(BaseAgent):
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the Case Agent workflow.
        """
        query = state.get("case_query")
        if not query:
            return {"case_context": []}
            
        print(f"🏛️ Case Agent Processing: {query}")
        
        # 1. Enhance Query
        enhanced_query = self.enhance_query(query, "case")
        print(f"   Enhanced: {enhanced_query}")
        
        # 2. Define Domains
        # Prioritize TARGET_CASE_SITE
        domains = [TARGET_CASE_SITE] + [d for d in PREFERRED_DOMAINS if d != TARGET_CASE_SITE]
        
        # 3. Web Search
        context_str, results = web_search_tool.run(enhanced_query, domains)
        
        # 4. Rerank
        reranked = rerank_documents(query, results, top_n=10)
        
        # 5. Return update
        return {"case_context": reranked}

case_agent = CaseAgent()
