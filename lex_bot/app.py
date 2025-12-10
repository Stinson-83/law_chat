import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load env relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, ".env"))

# Import Graph (Ensure path is correct)
# Since we are inside lex_bot, we can import relative or absolute if in path
# Assuming running via uvicorn lex_bot.app:app from one level up OR python lex_bot/app.py
import sys
if current_dir not in sys.path:
    sys.path.append(os.path.dirname(current_dir)) # Add parent to path to allow 'lex_bot.graph' imports if needed
    sys.path.append(current_dir) # Add self

from lex_bot.graph import app as agent_app
from lex_bot.state import AgentState

app = FastAPI(title="Lex Bot API", description="Advanced Agentic Indian Law Research Bot")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    law_query: Optional[str] = None
    case_query: Optional[str] = None
    final_state: Optional[dict] = None

@app.get("/")
def health_check():
    return {"status": "active", "system": "Lex Bot"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Main endpoint to interact with the bot.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    print(f"📨 API Request: {request.query}")
    
    initial_state = AgentState(
        messages=[],
        original_query=request.query,
        law_context=[],
        case_context=[],
        errors=[]
    )
    
    try:
        # Invoke the graph
        result = agent_app.invoke(initial_state)
        
        return QueryResponse(
            answer=result.get("final_answer", "No answer generated."),
            law_query=result.get("law_query"),
            case_query=result.get("case_query"),
            # Exclude large context from response to save bandwidth, or include if needed
            # final_state=result 
        )
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Server on Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
