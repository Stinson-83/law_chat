from langgraph.graph import StateGraph, END
from .state import AgentState
from .agents.manager import manager_agent
from .agents.law_agent import law_agent
from .agents.case_agent import case_agent

class LegalWorkflow:
    def __init__(self):
        """Initializes the workflow and compiles the graph."""
        self.workflow = StateGraph(AgentState)
        self.app = self._build_graph()

    def _route_agents(self, state: AgentState):
        """
        Determines which agents to call based on the decomposed query.
        Returns a list of node names to execute (allowing for parallel execution).
        """
        routes = []
        if state.get("law_query"):
            routes.append("law_agent")
        if state.get("case_query"):
            routes.append("case_agent")
        
        # Fallback: if no specific routes found, default to both or handle error
        if not routes:
             return ["law_agent", "case_agent"]
             
        return routes

    def _build_graph(self):
        """Defines the nodes, edges, and compilation logic."""
        
        # 1. Add Nodes
        # assuming agent.run or agent.decompose_query are the callables
        self.workflow.add_node("manager_decompose", manager_agent.decompose_query)
        self.workflow.add_node("law_agent", law_agent.run)
        self.workflow.add_node("case_agent", case_agent.run)
        self.workflow.add_node("manager_aggregate", manager_agent.generate_response)
        
        # 2. Set Entry Point
        self.workflow.set_entry_point("manager_decompose")
        
        # 3. Conditional Routing (Fan-out)
        self.workflow.add_conditional_edges(
            "manager_decompose",
            self._route_agents,
            ["law_agent", "case_agent"]
        )
        
        # 4. Fan-in (Aggregation)
        self.workflow.add_edge("law_agent", "manager_aggregate")
        self.workflow.add_edge("case_agent", "manager_aggregate")
        
        # 5. End
        self.workflow.add_edge("manager_aggregate", END)
        
        return self.workflow.compile()

    def run(self, input_data: dict):
        """
        Convenience method to invoke the graph.
        """
        return self.app.invoke(input_data)