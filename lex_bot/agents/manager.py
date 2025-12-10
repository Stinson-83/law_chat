import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from .base_agent import BaseAgent
from ..tools.reranker import rerank_documents

class ManagerAgent(BaseAgent):
    def decompose_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the user query and decomposes it.
        """
        original_query = state.get("original_query")
        print(f"🤖 Manager Analyzing: {original_query}")
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following legal query and decompose it into sub-queries for two specific agents:
        1. Law Agent: Searches for Statutes, Acts, Sections, and general legal principles.
        2. Case Agent: Searches for specific Precedents, Case Laws, and Judgments.
        
        If the query is purely about one, leave the other empty.
        If it's mixed, provide efficient queries for both.
        
        Query: {query}
        
        Output JSON format:
        {{
            "law_query": "query for law agent or null",
            "case_query": "query for case law or null"
        }}
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({"query": original_query})
            return {
                "law_query": result.get("law_query"),
                "case_query": result.get("case_query")
            }
        except Exception as e:
            print(f"❌ Decomposition Failed: {e}")
            # Fallback: Send to both
            return {"law_query": original_query, "case_query": original_query}

    def generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates context and generates final answer.
        """
        print("📝 Generating Final Response...")
        law_ctx = state.get("law_context", [])
        case_ctx = state.get("case_context", [])
        
        # Combine all candidates
        all_docs = law_ctx + case_ctx
        
        # Context Management: Rerank everything against original query to find the absolute best chunks
        # Limit to fit context window (e.g. Top 15)
        top_docs = rerank_documents(state["original_query"], all_docs, top_n=15)
        
        # Format context
        context_str = ""
        for i, doc in enumerate(top_docs, 1):
            source_type = doc.get('source', 'Web')
            title = doc.get('title', 'Untitled')
            snippet = doc.get('search_hit') or doc.get('snippet') or doc.get('text', '')
            context_str += f"[{i}] {title} ({doc.get('url')}) [{source_type}]:\n{snippet}\n\n"
            
        prompt = ChatPromptTemplate.from_template("""
        You are an advanced Indian Legal Assistant. 
        Answer the user's query based on the provided context.
        
        Context:
        {context}
        
        Query: {query}
        
        Instructions:
        - Cite your sources using the [Number] format.
        - Differentiate between Statutes (Law) and Precedents (Cases).
        - If the context doesn't contain the answer, say so, but attempt to infer from general legal knowledge if safe.
        - Be professional, precise, and structured.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "query": state["original_query"]})
        
        return {"final_answer": answer}

manager_agent = ManagerAgent()
