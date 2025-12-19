from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
import operator


class AgentState(TypedDict):
    """
    State dictionary for the LangGraph workflow.
    
    Enhanced for Lex Bot v2 with:
    - User memory integration
    - Dual LLM modes
    - Query complexity routing
    """
    # Helper to merge lists (append) instead of overwrite
    messages: Annotated[List[Dict[str, str]], operator.add]
    
    # Original user query
    original_query: str
    
    # --- New v2 fields ---
    # User and session tracking
    user_id: Optional[str]
    session_id: Optional[str]
    
    # LLM mode for this query
    llm_mode: Literal["fast", "reasoning"]
    
    # Query classification
    query_complexity: Literal["simple", "complex"]
    
    # Decomposed queries for complex routing
    sub_queries: Optional[List[str]]
    law_query: Optional[str]
    case_query: Optional[str]
    
    # Collected contexts from agents
    law_context: Annotated[List[Dict], operator.add]
    case_context: Annotated[List[Dict], operator.add]
    
    # Tool results from various agents
    tool_results: Annotated[List[Dict], operator.add]
    
    # --- PDF Integration ---
    uploaded_file_path: Optional[str]
    document_context: Annotated[List[Dict], operator.add]
    
    # Memory context retrieved for this user
    memory_context: Optional[List[Dict]]
    
    # Final answer
    final_answer: Optional[str]
    
    # Error tracking
    errors: Annotated[List[str], operator.add]

