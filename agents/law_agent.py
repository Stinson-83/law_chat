import os
import logging
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from search import hybrid_search
from rerank import rerank
from web_search import web_searcher

logger = logging.getLogger(__name__)

class LawAgent(BaseAgent):
    """
    Handles general Indian Law queries.
    Uses Database (Hybrid Search + Rerank) -> Web Search Fallback.
    """
    
    SEARCH_THRESHOLD = 0.45
    LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-2.5-Flash")

    def __init__(self):
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.LLM_MODEL_NAME)

    def _generate_answer(self, query: str, context: str, source_type: str) -> str:
        """
        Generates the answer using Gemini.
        """
        system_prompt = f"""
        You are an expert Legal AI Assistant for Indian Constitutional Law.
        
        CONTEXT SOURCE: {source_type.upper()}
        
        INSTRUCTIONS:
        1. Answer the USER QUERY specifically using the provided CONTEXT.
        2. CITATIONS:
           - If context is from 'DATABASE': Cite specific Sections, Articles, or Case Names.
           - If context is from 'WEB': Cite the website/domain provided.
        3. TONE: Professional, objective, and legally precise.
        4. FALLBACK: If the answer is not in the context, clearly state: "I cannot find specific legal provisions for this in my current knowledge base."
        
        CONTEXT:
        {context}
        
        USER QUERY: {query}
        """
        try:
            response = self.model.generate_content(system_prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return "I encountered an error while generating the answer from the legal context."

    def process(self, query: str, filters: Optional[Dict] = None, top_n: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execution flow:
        1. Search DB (Hybrid)
        2. Rerank
        3. Check Confidence
        4. (Optional) Fallback to Web
        5. Generate Answer
        """
        logger.info(f"⚖️ Law Agent Processing: {query}")
        
        # 1. Hybrid Search
        candidates = hybrid_search(
            query, 
            filters=filters,
            pre_k=200 
        )
        
        # 2. Rerank
        ranked_results = rerank(query, candidates, top_n=top_n)
        
        # 3. Check Confidence
        top_score = ranked_results[0]['rerank'] if ranked_results else 0.0
        logger.info(f"📊 Top DB Score: {top_score:.4f} (Threshold: {self.SEARCH_THRESHOLD})")
        
        final_context = ""
        final_sources = []
        source_origin = "database"
        
        # ROUTING LOGIC
        if top_score >= self.SEARCH_THRESHOLD:
            # CASE A: High Confidence DB
            source_origin = "database"
            context_blocks = []
            for r in ranked_results:
                context_text = r.get('text', '') 
                meta = f"{r['title']} - {r['heading']}"
                context_blocks.append(f"DOCUMENT: {meta}\nTEXT: {context_text}")
                
                final_sources.append({
                    "id": r['id'],
                    "title": r['title'],
                    "heading": r['heading'],
                    "text": r.get('search_hit', context_text)[:300] + "...",
                    "score": r['rerank'],
                    "type": "db"
                })
            final_context = "\n\n".join(context_blocks)
            
        else:
            # CASE B: Low Confidence -> Web Fallback
            logger.info("⚠️ Low confidence. Triggering Web Search.")
            source_origin = "web"
            
            web_context, web_hits = web_searcher.search(query)
            
            if not web_context:
                # Fallback to weak DB
                if ranked_results:
                    logger.warning("Web search failed. Using weak DB results.")
                    source_origin = "database (low confidence)"
                    final_context = "\n".join([r['text'] for r in ranked_results])
                    final_sources = [{
                        "id": r['id'], 
                        "title": r['title'], 
                        "text": r.get('search_hit', '')[:200], 
                        "type": "db", 
                        "score": r['rerank']
                    } for r in ranked_results]
                else:
                    return {
                        "answer": "I could not find relevant legal information in the database or reliable web sources.",
                        "sources": [],
                        "source_type": "none",
                        "confidence": 0.0
                    }
            else:
                final_context = web_context
                for hit in web_hits:
                    final_sources.append({
                        "id": "web",
                        "title": hit.get('title', 'Unknown'),
                        "url": hit.get('url', '#'),
                        "text": hit.get('text', '')[:200] + "...",
                        "type": "web"
                    })
        
        # 4. Generate Answer
        answer = self._generate_answer(query, final_context, source_origin)
        
        return {
            "answer": answer,
            "sources": final_sources,
            "source_type": source_origin,
            "confidence": top_score
        }
