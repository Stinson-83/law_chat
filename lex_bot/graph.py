"""
Lex Bot v2 - LangGraph Workflow

Flow:
1. Memory Recall - Fetch relevant user memories (if enabled)
2. Manager Decompose - Analyze and decompose query
3. Law Agent & Case Agent - Parallel search (fan-out)
4. Manager Aggregate - Synthesize final response (fan-in)
5. Memory Store - Save key facts from conversation

Architecture:
    ┌─────────────────┐
    │  Memory Recall  │ (optional - if user_id provided)
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │Manager Decompose│
    └────────┬────────┘
             ▼
    ┌────────┴────────┐
    │    (Fan-Out)    │
    ▼                 ▼
┌─────────┐     ┌──────────┐
│Law Agent│     │Case Agent│
└────┬────┘     └────┬─────┘
     │               │
     └───────┬───────┘
             ▼
    ┌─────────────────┐
    │Manager Aggregate│
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  Memory Store   │ (optional - saves key facts)
    └────────┬────────┘
             ▼
           [END]
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from .state import AgentState
from .agents.manager import manager_agent
from .agents.law_agent import law_agent
from .agents.case_agent import case_agent
from .agents.document_agent import document_agent
from .agents.strategy_agent import strategy_agent
from .agents.citation_agent import citation_agent
from .agents.explainer_agent import explainer_agent
from .memory import UserMemoryManager
from .memory import UserMemoryManager
from .config import MEM0_ENABLED

# Import tools to ensure registration
import lex_bot.tools.ecourts
import lex_bot.tools.web_search
import lex_bot.tools.db_search
import lex_bot.tools.indian_kanoon
import lex_bot.tools.latin_phrases
import lex_bot.tools.penal_code_lookup


def memory_recall_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch relevant memories for context enrichment.
    """
    user_id = state.get("user_id")
    if not user_id or not MEM0_ENABLED:
        return {"memory_context": []}
    
    try:
        memory_manager = UserMemoryManager(user_id=user_id)
        query = state.get("original_query", "")
        memories = memory_manager.search(query, limit=5)
        
        if memories:
            print(f"📚 Retrieved {len(memories)} relevant memories for user {user_id}")
        
        return {"memory_context": memories}
    except Exception as e:
        print(f"⚠️ Memory recall failed: {e}")
        return {"memory_context": []}


def memory_store_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store key facts from the conversation for future reference.
    """
    user_id = state.get("user_id")
    if not user_id or not MEM0_ENABLED:
        return {}
    
    try:
        memory_manager = UserMemoryManager(user_id=user_id)
        
        # Create conversation context to store
        messages = [
            {"role": "user", "content": state.get("original_query", "")},
            {"role": "assistant", "content": state.get("final_answer", "")[:1000]}  # Truncate
        ]
        
        memory_manager.add(messages)
        print(f"💾 Stored conversation to memory for user {user_id}")
        
        return {}
    except Exception as e:
        print(f"⚠️ Memory store failed: {e}")
        return {}


from .agents.strategy_agent import strategy_agent
from .agents.citation_agent import citation_agent
from .agents.explainer_agent import explainer_agent

def define_graph():
    """
    Build and compile the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # === NODES ===
    workflow.add_node("memory_recall", memory_recall_node)
    workflow.add_node("manager_decompose", manager_agent.decompose_query)
    workflow.add_node("law_agent", law_agent.run)
    workflow.add_node("case_agent", case_agent.run)
    workflow.add_node("document_agent", document_agent.run)
    
    # New Agents
    workflow.add_node("strategy_agent", strategy_agent.run)
    workflow.add_node("citation_agent", citation_agent.run)
    workflow.add_node("explainer_agent", explainer_agent.run)
    
    workflow.add_node("manager_aggregate", manager_agent.generate_response)
    workflow.add_node("memory_store", memory_store_node)
    
    # Router Node
    def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify query complexity."""
        from lex_bot.core.router import get_router
        
        query = state.get("original_query", "")
        router = get_router()
        classification = router.classify(query)
        
        complexity = classification.get("complexity", "simple")
        suggested_agents = classification.get("suggested_agents", ["research"])
        
        print(f"🚦 Router Classification: {complexity.upper()}")
        print(f"   Suggested Agents: {suggested_agents}")
        
        # Set LLM mode based on complexity
        llm_mode = "reasoning" if complexity == "complex" else "fast"
        
        return {
            "complexity": complexity,
            "llm_mode": llm_mode, # Override initial mode
            "router_reasoning": classification.get("reasoning"),
            "suggested_agents": suggested_agents
        }

    workflow.add_node("router", router_node)
    
    # === EDGES ===
    # Entry point: Start with memory recall
    workflow.set_entry_point("memory_recall")
    
    # Memory recall -> Router -> Manager decompose
    workflow.add_edge("memory_recall", "router")
    workflow.add_edge("router", "manager_decompose")
    
    # Conditional routing based on decomposed queries AND router suggestions
    def route_agents(state: AgentState) -> List[str]:
        """Route to appropriate agents based on query decomposition and router suggestions."""
        # Priority: If file is uploaded, go to document agent
        if state.get("uploaded_file_path"):
            return ["document_agent"]

        routes = set()
        
        # 1. Check Decomposed Queries (Legacy/Fallback)
        if state.get("law_query"):
            routes.add("law_agent")
        if state.get("case_query"):
            routes.add("case_agent")
            
        # 2. Check Router Suggestions
        suggested = state.get("suggested_agents", [])
        
        # Map router output to graph node names
        # Router outputs: ["research", "law", "case", "strategy", "citation", "explainer"]
        if "strategy" in suggested:
            routes.add("strategy_agent")
        if "citation" in suggested:
            routes.add("citation_agent")
        if "explainer" in suggested:
            routes.add("explainer_agent")
        if "law" in suggested:
            routes.add("law_agent")
        if "case" in suggested:
            routes.add("case_agent")
            
        # If "research" is the only suggestion, default to Law + Case
        if not routes or (len(routes) == 0 and "research" in suggested):
             routes.add("law_agent")
             routes.add("case_agent")
        
        return list(routes)
    
    # Fan-out: Manager decompose -> All Agents (parallel)
    workflow.add_conditional_edges(
        "manager_decompose",
        route_agents,
        ["law_agent", "case_agent", "document_agent", "strategy_agent", "citation_agent", "explainer_agent"]
    )
    
    # Fan-in: All agents -> Manager aggregate
    workflow.add_edge("law_agent", "manager_aggregate")
    workflow.add_edge("case_agent", "manager_aggregate")
    workflow.add_edge("strategy_agent", "manager_aggregate")
    workflow.add_edge("citation_agent", "manager_aggregate")
    workflow.add_edge("explainer_agent", "manager_aggregate")
    
    # Document Agent -> Memory Store (Skip aggregate as it generates its own answer)
    workflow.add_edge("document_agent", "memory_store")
    
    # Final aggregation -> Memory store -> END
    workflow.add_edge("manager_aggregate", "memory_store")
    workflow.add_edge("memory_store", END)
    
    return workflow.compile()


# Compile the graph
app = define_graph()


# Convenience function for direct invocation
def run_query(
    query: str,
    user_id: str = None,
    session_id: str = None,
    file_path: str = None
) -> Dict[str, Any]:
    """
    Run a legal research query through the agent workflow.
    
    Args:
        query: User's legal research query
        user_id: Optional user ID for memory personalization
        session_id: Optional session ID for conversation tracking
    
    Returns:
        Final state with answer and context
    """
    initial_state = {
        "messages": [],
        "original_query": query,
        "user_id": user_id,
        "session_id": session_id,
        "llm_mode": "fast", # Default, will be updated by Router
        "uploaded_file_path": file_path,
        "law_context": [],
        "case_context": [],
        "tool_results": [],
        "errors": [],
    }
    
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Quick test
    print("🚀 Testing Lex Bot v2 Graph...")
    result = run_query("What is Section 302 IPC?")
    print("\n📝 Answer:")
    print(result.get("final_answer", "No answer generated"))
