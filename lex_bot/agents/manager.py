import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from .base_agent import BaseAgent
from ..tools.reranker import rerank_documents
from ..core.router import ROUTER_PROMPT  # Use enhanced router prompt


class ManagerAgent(BaseAgent):
    def classify_and_route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        First-stage router: Classifies query and assigns specific tasks to agents.
        
        Returns:
            State update with complexity, agent_tasks, synthesis_instruction, etc.
        """
        original_query = state.get("original_query")
        print(f"🧭 Router Analyzing: {original_query}")
        
        prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({"query": original_query})
            complexity = result.get("complexity", "simple")
            
            # Extract agent_tasks (new format with task_id, instruction, expected_output, dependencies)
            agent_tasks_raw = result.get("agent_tasks", [])
            
            # Build structured task data
            valid_agents = {"research", "explainer", "law", "case", "citation", "strategy"}
            selected_agents = []
            agent_tasks = {}  # agent -> full task object
            
            for task_item in agent_tasks_raw:
                agent_name = task_item.get("agent", "")
                
                if agent_name in valid_agents:
                    full_agent_name = f"{agent_name}_agent"
                    selected_agents.append(full_agent_name)
                    agent_tasks[full_agent_name] = {
                        "task_id": task_item.get("task_id", agent_name),
                        "instruction": task_item.get("instruction", "Perform research"),
                        "expected_output": task_item.get("expected_output", "Summary"),
                        "dependencies": task_item.get("dependencies", [])
                    }
            
            # Fallback for complex with no agents
            if complexity == "complex" and not selected_agents:
                selected_agents = ["law_agent", "case_agent"]
                agent_tasks["law_agent"] = {"task_id": "law_fallback", "instruction": "Find relevant statutes", "expected_output": "Statute list", "dependencies": []}
                agent_tasks["case_agent"] = {"task_id": "case_fallback", "instruction": "Find relevant cases", "expected_output": "Case list", "dependencies": []}
            
            print(f"   Complexity: {complexity.upper()}")
            if selected_agents:
                print(f"   Agents & Tasks:")
                for agent in selected_agents:
                    task = agent_tasks.get(agent, {})
                    print(f"      • {agent} [{task.get('task_id')}]: {task.get('instruction', '')[:60]}...")
            
            return {
                "complexity": complexity,
                "selected_agents": selected_agents,
                "agent_tasks": agent_tasks,
                "synthesis_instruction": result.get("synthesis_instruction", "Combine all agent outputs into cohesive response"),
                "synthesis_strategy": result.get("synthesis_strategy", "equal_weight"),
                "router_metadata": {
                    "reasoning": result.get("reasoning"),
                    "domain_tags": result.get("domain_tags", [])
                }
            }
        except Exception as e:
            print(f"❌ Router Failed: {e}")
            return {
                "complexity": "simple", 
                "selected_agents": [], 
                "agent_tasks": {},
                "synthesis_instruction": "Provide helpful response",
                "synthesis_strategy": "equal_weight"
            }


    # decompose_query is now obsolete - router assigns specific tasks directly via agent_tasks



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

    def check_needs_clarification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the query is ambiguous and needs clarification from user.
        Only called for COMPLEX queries.
        """
        original_query = state.get("original_query", "")
        
        clarification_prompt = ChatPromptTemplate.from_template("""
        You are a legal research assistant. Analyze if the following query is clear enough to proceed, 
        or if you need more information from the user.
        
        Query: {query}
        
        Check for:
        1. Missing jurisdiction (which state/court?)
        2. Missing context (civil/criminal? plaintiff/defendant?)
        3. Vague terms that could mean multiple things
        4. Missing facts essential for legal analysis
        
        Respond in JSON:
        {{
            "needs_clarification": true or false,
            "clarifying_questions": ["question1", "question2"] // Only if needs_clarification is true
        }}
        """)
        
        chain = clarification_prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({"query": original_query})
            
            if result.get("needs_clarification", False):
                questions = result.get("clarifying_questions", [])
                if questions:
                    # Format as a polite request
                    clarification_msg = "I need a bit more information to provide the best answer:\n\n"
                    for i, q in enumerate(questions[:3], 1):  # Max 3 questions
                        clarification_msg += f"{i}. {q}\n"
                    
                    return {
                        "needs_clarification": True,
                        "final_answer": clarification_msg
                    }
            
            return {"needs_clarification": False}
            
        except Exception as e:
            print(f"⚠️ Clarification check failed: {e}")
            return {"needs_clarification": False}

    def generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates context and generates final answer.
        Supports both normal and reasoning (CoT) modes.
        """
        print("📝 Generating Final Response...")
        law_ctx = state.get("law_context", [])
        case_ctx = state.get("case_context", [])
        llm_mode = state.get("llm_mode", "fast")
        
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
            
        # Add tool results (e.g. from Explainer or Research agent in complex mode)
        tool_results = state.get("tool_results", [])
        if tool_results:
            context_str += "\n\n=== AGENT REPORTS ===\n"
            for res in tool_results:
                agent = res.get("agent", "Unknown Agent")
                content = res.get("content", "")
                if content:
                    context_str += f"\n--- Report from {agent} ---\n{content}\n"
        
        # Choose prompt based on mode
        if llm_mode == "reasoning":
            # Chain-of-Thought prompt for reasoning mode
            prompt = ChatPromptTemplate.from_template("""
            You are a Senior Legal Research Assistant specializing in Indian Law.
            
            ## Task
            Analyze the legal query and provide a comprehensive, well-reasoned answer.
            
            ## Context (Retrieved Documents)
            {context}
            
            ## Query
            {query}
            
            ## Instructions
            Use Chain-of-Thought reasoning:
            
            **Step 1: Understand the Query**
            - What is the user really asking?
            - What are the key legal issues involved?
            
            **Step 2: Identify Relevant Law**
            - Which statutes, sections, or acts apply?
            - Cite specific provisions from the context.
            
            **Step 3: Analyze Case Law**
            - What precedents are relevant?
            - How have courts interpreted this?
            
            **Step 4: Synthesize**
            - Combine statutory and case law analysis.
            - Address any conflicts or nuances.
            
            **Step 5: Conclude**
            - Provide a clear, actionable answer.
            - Note any caveats or limitations.
            
            ## Format
            - Use clear headings
            - Cite sources as [1], [2], etc.
            - Distinguish between statutes (Law) and precedents (Cases)
            - Be professional and legally precise
            """)
        else:
            # Standard prompt for fast mode
            prompt = ChatPromptTemplate.from_template("""
            You are an Assistant of a Legal Advocate, you expertize in Indian Laws and Cases. 
            Answer the user's legal research query based on the provided context.
            
            Context: {context}
            
            Query: {query}
            
            Instructions:
            - Breakdown the query into aspects and answer each from the context.
            - Cite your sources using [Number] format.
            - Differentiate between Statutes (Law) and Precedents (Cases).
            - If context is insufficient, say so but infer from general legal knowledge if safe.
            - Be professional, precise, and legally sound.
            """)
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "query": state["original_query"]})
        
        # For reasoning mode, extract the reasoning trace
        result = {"final_answer": answer}
        if llm_mode == "reasoning":
            result["reasoning_trace"] = answer  # Full CoT is the reasoning trace
        
        return result

manager_agent = ManagerAgent()


