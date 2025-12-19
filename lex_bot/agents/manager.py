import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from .base_agent import BaseAgent
from ..tools.reranker import rerank_documents


# Prompt for complexity classification and agent selection
ROUTER_PROMPT = """You are a legal query router. Analyze the user's query and determine:
1. **Complexity**: Is this a SIMPLE or COMPLEX query?
   - SIMPLE: Direct questions about a specific law, section, definition, or basic legal concept.
     Examples: "What is Section 302 IPC?", "Define bail", "What are the grounds for divorce?"
   - COMPLEX: Queries requiring multi-step research, legal strategy, citation analysis, or synthesis.
     Examples: "Analyze the legal strategy for defending a murder case", "How has Kesavananda Bharati been cited?", "Compare IPC and BNS for theft"

2. **Agents** (only if COMPLEX): Which agents are needed? Choose ONLY the ones truly required.
   - research_agent: For general legal research with RAG + web search
   - explainer_agent: For educational explanations, simplifying concepts for students
   - law_agent: For statutes, acts, sections, legal provisions
   - case_agent: For case law, precedents, judgments
   - citation_agent: For citation network analysis (how a case has been cited, affirmed, overruled)
   - strategy_agent: For legal strategy, arguments for/against, risk assessment

Query: {query}

Respond in JSON:
{{
    "complexity": "simple" or "complex",
    "agents": ["agent1", "agent2"] // Only if complex, else empty list
}}
"""


class ManagerAgent(BaseAgent):
    def classify_and_route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        First-stage router: Classifies query complexity and selects agents.
        
        Returns:
            State update with 'complexity' and 'selected_agents'
        """
        original_query = state.get("original_query")
        print(f"🧭 Router Analyzing: {original_query}")
        
        prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({"query": original_query})
            complexity = result.get("complexity", "simple")
            agents = result.get("agents", [])
            
            # Validate agents
            valid_agents = {"research_agent", "explainer_agent", "law_agent", "case_agent", "citation_agent", "strategy_agent"}
            agents = [a for a in agents if a in valid_agents]
            
            # Fallback for complex with no agents
            if complexity == "complex" and not agents:
                agents = ["law_agent", "case_agent"]
            
            print(f"   Complexity: {complexity.upper()}")
            if agents:
                print(f"   Selected Agents: {agents}")
            
            return {
                "complexity": complexity,
                "selected_agents": agents
            }
        except Exception as e:
            print(f"❌ Router Failed: {e}")
            # Fallback: Treat as simple
            return {"complexity": "simple", "selected_agents": []}

    def decompose_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the user query and decomposes it for parallel agent execution.
        Only called for COMPLEX queries.
        """
        original_query = state.get("original_query")
        selected_agents = state.get("selected_agents", [])
        print(f"🤖 Manager Decomposing for agents: {selected_agents}")
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following legal query and create optimized sub-queries for the selected agents.
        
        Selected Agents: {agents}
        - law_agent: Searches for Statutes, Acts, Sections, and general legal principles.
        - case_agent: Searches for specific Precedents, Case Laws, and Judgments.
        - citation_agent: Analyzes citation networks for a specific case.
        - strategy_agent: Develops legal strategy and arguments.
        
        Create efficient, targeted queries for ONLY the selected agents.
        
        Query: {query}
        
        Output JSON format:
        {{
            "law_query": "query for law agent or null",
            "case_query": "query for case law or null"
        }}
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({"query": original_query, "agents": selected_agents})
            return {
                "law_query": result.get("law_query") if "law_agent" in selected_agents else None,
                "case_query": result.get("case_query") if "case_agent" in selected_agents else None
            }
        except Exception as e:
            print(f"❌ Decomposition Failed: {e}")
            # Fallback: Send original to all selected
            return {
                "law_query": original_query if "law_agent" in selected_agents else None,
                "case_query": original_query if "case_agent" in selected_agents else None
            }


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
        
        # Context Management: Rerank everything against original query
        # Limit to 10 for token optimization
        top_docs = rerank_documents(state["original_query"], all_docs, top_n=10)
        
        # Format context
        context_str = ""
        for i, doc in enumerate(top_docs, 1):
            source_type = doc.get('source', 'Web')
            title = doc.get('title', 'Untitled')
            snippet = doc.get('search_hit') or doc.get('snippet') or doc.get('text', '')
            context_str += f"[{i}] {title} ({doc.get('url')}) [{source_type}]:\n{snippet}\n\n"
            
        prompt = ChatPromptTemplate.from_template("""
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
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "query": state["original_query"]})
        
        return {"final_answer": answer}

manager_agent = ManagerAgent()

