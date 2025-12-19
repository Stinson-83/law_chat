from typing import Dict, Any
from .base_agent import BaseAgent
from ..tools.web_search import web_search_tool
from ..tools.reranker import rerank_documents
from ..config import TARGET_CASE_SITE#, PREFERRED_DOMAINS

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
        domains = [TARGET_CASE_SITE] #+ [d for d in PREFERRED_DOMAINS if d != TARGET_CASE_SITE]
        
        # 3. Dynamic Tool Execution via Registry
        from lex_bot.core.tool_registry import tool_registry
        
        # Get all tools capable of "case_search"
        tools = tool_registry.get_by_capability("case_search")
        if not tools:
            print("⚠️ No case search tools found in registry!")
            # Fallback to direct import if registry fails (safety net)
            tools = [web_search_tool]
            
        all_results = []
        combined_context = ""
        
        for tool in tools:
            try:
                tool_name = tool.__class__.__name__
                print(f"   🔧 Running {tool_name}...")
                
                # Handle different tool signatures/returns
                if hasattr(tool, "run"):
                    # Check signature or try/except
                    # WebSearchTool returns (context, results)
                    # ECourtsTool returns results (List[Dict])
                    
                    if tool_name == "WebSearchTool":
                        ctx, res = tool.run(enhanced_query, domains)
                        combined_context += ctx + "\n\n"
                        all_results.extend(res)
                    elif tool_name == "ECourtsTool":
                        res = tool.run(enhanced_query, max_results=5)
                        all_results.extend(res)
                        # Generate context from structured results
                        for r in res:
                            combined_context += f"CASE: {r.get('title')} ({r.get('court')}, {r.get('date')})\n{r.get('url')}\n\n"
                    else:
                        # Generic fallback
                        res = tool.run(enhanced_query)
                        if isinstance(res, tuple):
                            combined_context += res[0] + "\n\n"
                            all_results.extend(res[1])
                        elif isinstance(res, list):
                            all_results.extend(res)
                            
            except Exception as e:
                print(f"   ❌ Tool {tool.__class__.__name__} failed: {e}")
        
        # 4. Rerank
        reranked = rerank_documents(query, all_results, top_n=10)
        
        # 5. Return update
        return {"case_context": reranked}

case_agent = CaseAgent()
