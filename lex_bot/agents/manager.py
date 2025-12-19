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

    # def generate_outline(self, state: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Aggregates context and generates outline.
    #     """
    #     print("📝 Generating Outline...")
    #     law_ctx = state.get("law_context", [])
    #     case_ctx = state.get("case_context", [])
        
    #     # Combine all candidates
    #     all_docs = law_ctx + case_ctx
        
    #     # Context Management: Rerank everything against original query to find the absolute best chunks
    #     # Limit to fit context window (e.g. Top 15)
    #     top_docs = rerank_documents(state["original_query"], all_docs, top_n=15)
        
    #     # Format context
    #     context_str = ""
    #     for i, doc in enumerate(top_docs, 1):
    #         source_type = doc.get('source', 'Web')
    #         title = doc.get('title', 'Untitled')
    #         snippet = doc.get('search_hit') or doc.get('snippet') or doc.get('text', '')
    #         context_str += f"[{i}] {title} ({doc.get('url')}) [{source_type}]:\n{snippet}\n\n"
            
    #     prompt = ChatPromptTemplate.from_template("""
        # You are an Assistant to a Legal Advocate specializing in Indian Law.

        # Your task is to produce a structured OUTLINE of how you will research and answer the user's legal query.

        # DO NOT provide the final answer yet.

        # Using ONLY the provided context, create an outline that includes:

        # 1. Key legal issues raised by the query.
        # 2. Relevant statutes (sections of acts) found in context.
        # 3. Relevant case law (precedents) found in context.
        # 4. Sub-questions that must be answered.
        # 5. Which parts of the context apply to each sub-question.
        # 6. What additional general legal principles (Indian law only) may assist, if safe.

        # Rules:
        # - Do NOT generate any legal conclusions or final answers here.
        # - Do NOT hallucinate cases or statutes not present in the context (general principles allowed, but flagged as such).
        # - Focus only on structuring the reasoning.

        # Context:
        # {context}

        # Query:
        # {query}

        # Now produce ONLY the outline.

    #     """)
        
    #     chain = prompt | self.llm | StrOutputParser()
    #     answer = chain.invoke({"context": context_str, "query": state["original_query"]})
        
    #     return {"outline": answer}

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
            
        # Dynamic Mode Switching
        llm_mode = state.get("llm_mode", "fast")
        if self.mode != llm_mode:
            print(f"🔄 Switching Manager Agent to {llm_mode} mode...")
            self.switch_mode(llm_mode)
            
        # Select Prompt based on Mode
        if llm_mode == "reasoning":
            # Chain of Thought Prompt
            prompt_template = """
            You are an expert Legal Research Assistant specializing in Indian Law.
            The user has asked a complex legal query: {query}
            
            You have been provided with the following research context:
            {context}
            
            Your task is to provide a comprehensive, legally sound answer using Chain of Thought reasoning.
            
            Step-by-Step Reasoning Instructions:
            1. **Analyze the Query**: Break down the legal issues involved.
            2. **Evaluate Statutes**: Identify relevant Acts and Sections from the context. Explain how they apply.
            3. **Analyze Case Law**: Discuss relevant precedents. How do they interpret the law? Are they binding?
            4. **Synthesize**: Combine statutory and case law analysis to form a conclusion.
            5. **Final Answer**: Present the final answer clearly, citing sources using [Number] format.
            
            Format your response as follows:
            ### Legal Analysis
            [Your step-by-step reasoning here]
            
            ### Conclusion
            [Your final direct answer here]
            """
        else:
            # Standard Fast Prompt
            prompt_template = """
            You are an Assistant of a Legal Advocate, you expertizes in Indian Laws and Case related to it. 
            The user wants to conduct a legal research this is his query {query}. 
            You have to help the user in his research, so answer the user's query based on the provided context.
            
            Context: {context}
            
            Query: {query}
            
            Instructions for answering the query:
            - To answer, breakdown the query into different aspects and derive the answer for each aspect from the give context.
            - Cite your sources using the [Number] format.
            - Differentiate between Statutes (Law) and Precedents (Cases).
            - If the context doesn't contain the answer, say so, but attempt to infer from general legal knowledge if safe.
            - Be professional, precise, and legally sound.
            """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "query": state["original_query"]})
        
        return {"final_answer": answer}

manager_agent = ManagerAgent()
