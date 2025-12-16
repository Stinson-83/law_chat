from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator

class AgentState(TypedDict):
    """
    State dictionary for the LangGraph workflow.
    """
    # Helper to merge lists (append) instead of overwrite
    messages: Annotated[List[Dict[str, str]], operator.add] 
    
    # Original user query
    original_query: str
    
    # Decomposed queries
    law_query: Optional[str]
    case_query: Optional[str]
    
    # Collected contexts from agents
    # We use a list to allow multiple agents to append their findings
    law_context: Annotated[List[Dict], operator.add]
    case_context: Annotated[List[Dict], operator.add]
    
    # Outline
    #outline: Optional[str]

    # Final answer
    final_answer: Optional[str]
    
    # Error tracking
    errors: Annotated[List[str], operator.add]
