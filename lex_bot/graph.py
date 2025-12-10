from langgraph.graph import StateGraph, END
from .state import AgentState
from .agents.manager import manager_agent
from .agents.law_agent import law_agent
from .agents.case_agent import case_agent

def define_graph():
    workflow = StateGraph(AgentState)
    
    # 1. Nodes
    workflow.add_node("manager_decompose", manager_agent.decompose_query)
    workflow.add_node("law_agent", law_agent.run)
    workflow.add_node("case_agent", case_agent.run)
    workflow.add_node("manager_aggregate", manager_agent.generate_response)
    
    # 2. Edges
    workflow.set_entry_point("manager_decompose")
    
    # Conditional Routing Logic
    def route_agents(state: AgentState):
        routes = []
        if state.get("law_query"):
            routes.append("law_agent")
        if state.get("case_query"):
            routes.append("case_agent")
        
        # If both are null/empty (fallback), defaulting to both usually safer, 
        # or handle gracefully. Manager fallback sets both to original query.
        if not routes:
             return ["law_agent", "case_agent"]
             
        return routes

    # Fan-out
    workflow.add_conditional_edges(
        "manager_decompose",
        route_agents,
        ["law_agent", "case_agent"]
    )
    
    # Fan-in
    workflow.add_edge("law_agent", "manager_aggregate")
    workflow.add_edge("case_agent", "manager_aggregate")
    
    workflow.add_edge("manager_aggregate", END)
    
    return workflow.compile()

app = define_graph()
