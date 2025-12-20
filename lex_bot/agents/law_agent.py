from typing import Dict, Any
from .base_agent import BaseAgent
from ..tools.db_search import search_tool
from ..tools.reranker import rerank_documents
from ..config import PREFERRED_DOMAINS

class LawAgent(BaseAgent):
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the Law Agent workflow.
        Retrieves statutory context based on assigned task.
        """
        # Get specific instruction from router's agent_tasks
        task = state.get("agent_tasks", {}).get("law_agent", {})
        instruction = task.get("instruction", "")
        
        if not instruction:
            # Fallback to original query if no specific task
            instruction = state.get("original_query", "")
        
        if not instruction:
            return {"law_context": []}
            
        print(f"⚖️ Law Agent Task: {instruction[:80]}...")
        
        # 1. Enhance Query
        enhanced_query = self.enhance_query(instruction, "law")
        print(f"   Enhanced: {enhanced_query[:80]}...")
        
        # 2. Define Domains
        domains = PREFERRED_DOMAINS 
        
        # 3. Search (DB -> Web Fallback handled in tool)
        context_str, results = search_tool.run(enhanced_query, domains)
        
        # 4. Rerank against original instruction
        reranked = rerank_documents(instruction, results, top_n=10)
        
        # 5. Return update
        return {"law_context": reranked} 

law_agent = LawAgent()

