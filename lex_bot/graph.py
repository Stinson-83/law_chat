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
from .memory import UserMemoryManager
from .config import MEM0_ENABLED


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
    workflow.add_node("manager_aggregate", manager_agent.generate_response)
    workflow.add_node("memory_store", memory_store_node)
    
    # === EDGES ===
    # Entry point: Start with memory recall
    workflow.set_entry_point("memory_recall")
    
    # Memory recall -> Manager decompose
    workflow.add_edge("memory_recall", "manager_decompose")
    
    # Conditional routing based on decomposed queries
    def route_agents(state: AgentState) -> List[str]:
        """Route to appropriate agents based on query decomposition."""
        routes = []
        
        if state.get("law_query"):
            routes.append("law_agent")
        if state.get("case_query"):
            routes.append("case_agent")
        
        # Fallback: if no specific routing, use both agents
        if not routes:
            return ["law_agent", "case_agent"]
        
        return routes
    
    # Fan-out: Manager decompose -> Law Agent & Case Agent (parallel)
    workflow.add_conditional_edges(
        "manager_decompose",
        route_agents,
        ["law_agent", "case_agent"]
    )
    
    # Fan-in: Both agents -> Manager aggregate
    workflow.add_edge("law_agent", "manager_aggregate")
    workflow.add_edge("case_agent", "manager_aggregate")
    
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
    llm_mode: str = "fast"
) -> Dict[str, Any]:
    """
    Run a legal research query through the agent workflow.
    
    Args:
        query: User's legal research query
        user_id: Optional user ID for memory personalization
        session_id: Optional session ID for conversation tracking
        llm_mode: "fast" or "reasoning"
    
    Returns:
        Final state with answer and context
    """
    initial_state = {
        "messages": [],
        "original_query": query,
        "user_id": user_id,
        "session_id": session_id,
        "llm_mode": llm_mode,
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
