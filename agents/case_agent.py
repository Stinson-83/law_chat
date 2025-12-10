import os
import logging
import re
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from web_search import web_searcher
from rerank import rerank

logger = logging.getLogger(__name__)

class CaseAgent(BaseAgent):
    """
    Handles specific Case Law research.
    Flow:
    1. Targeted Search (e.g., indiankanoon.org)
    2. Fallback Web Search
    3. Scrape URLs
    4. Chunk Scraped Text
    5. Rerank Chunks (RAG)
    6. Generate Answer
    """
    
    # Configuration Placeholders
    TARGET_SITE = "indiankanoon.org" 
    TARGET_TAGS = ["info_indian_kanoon", "akoma-ntoso"] # Placeholder for future scraping logic

    LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    def __init__(self):
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.LLM_MODEL_NAME)

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Simple text chunker.
        """
        if not text:
            return []
        
        # Split by double newlines to respect paragraphs roughly
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _generate_answer(self, query: str, context: str) -> str:
        prompt = f"""
        Role: Legal Case Researcher
        
        Task: Analyze the provided case documents and answer the user query.
        
        INSTRUCTIONS:
        1. Focus on FACTS, JUDGMENT, and LEGAL PRINCIPLES.
        2. If the user asks for a summary, provide: Facts, Issues, Decision.
        3. Cite the source URL if available in the context.
        
        CONTEXT:
        {context} 
        
        USER QUERY: {query}
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Case Agent LLM Error: {e}")
            return "Error generating case analysis."

    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        logger.info(f"👨‍⚖️ Case Agent Processing: {query}")
        
        # 1. Targeted Search
        # We try to find the specific case on the target site
        context_text, hits = web_searcher.search(query, max_results=5, domains=[self.TARGET_SITE])
        
        # 2. Fallback if empty
        if not hits:
            logger.warning(f"Target site {self.TARGET_SITE} yielded no results. Expanding search.")
            context_text, hits = web_searcher.search(query, max_results=5) # Broad search
            
        if not hits:
            return {
                "answer": f"I could not find details for the case '{query}' online.",
                "sources": [],
                "confidence": 0.0
            }

        # 3. Scrape and Chunk (RAG pipeline on fresh data)
        # Verify context_text availability. web_searcher.search returns scraped text.
        # But we want to rerank specific chunks for better precision.
        
        # Recalculate chunks from the scraped text
        # Note: 'context_text' from web_searcher is big blob. 
        # We might want to perform RAG on it.
        
        chunks = self._chunk_text(context_text)
        candidates = []
        for i, chunk in enumerate(chunks):
            candidates.append({
                "id": f"chunk_{i}",
                "text": chunk, # Parent
                "search_hit": chunk, # Child for reranking
                "title": "Web Search Result",
                "heading": "Snippet"
            })
            
        # 4. Rerank Chunks
        # This helps pick the most relevant parts of the long web page
        ranked_chunks = rerank(query, candidates, top_n=5)
        
        # 5. Build Final Context
        final_context_blocks = []
        final_sources = []
        
        for rc in ranked_chunks:
            final_context_blocks.append(rc['text'])
            final_sources.append({
                "id": rc['id'],
                "title": "Search Result Chunk",
                "text": rc['text'][:200] + "...",
                "score": rc['rerank'],
                "type": "web_rag"
            })
            
        final_context_str = "\n---\n".join(final_context_blocks)
        
        # 6. Generate Answer
        answer = self._generate_answer(query, final_context_str)
        
        return {
            "answer": answer,
            "sources": final_sources,
            "confidence": ranked_chunks[0]['rerank'] if ranked_chunks else 0.0,
            "source_type": "web_rag"
        }
