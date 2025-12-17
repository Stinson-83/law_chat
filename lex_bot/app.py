"""
Lex Bot v2 - FastAPI Application

Endpoints:
- GET  /          : Health check
- POST /chat      : Main chat endpoint
- POST /memory    : User memory operations
"""

import os
import uuid
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load env
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, ".env"))

# Path setup for imports
import sys
if current_dir not in sys.path:
    sys.path.append(os.path.dirname(current_dir))
    sys.path.append(current_dir)

from lex_bot.graph import run_query
from lex_bot.memory import UserMemoryManager
from lex_bot.config import MEM0_ENABLED

# ============ FastAPI App ============
app = FastAPI(
    title="Lex Bot v2 API",
    description="Advanced Agentic Indian Law Research Bot with Memory",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Request/Response Models ============
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Legal research query")
    user_id: Optional[str] = Field(None, description="User ID for memory personalization")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    llm_mode: Optional[str] = Field("fast", description="LLM mode: 'fast' or 'reasoning'")


class QueryResponse(BaseModel):
    answer: str
    law_query: Optional[str] = None
    case_query: Optional[str] = None
    memory_used: bool = False
    processing_time_ms: Optional[int] = None


class MemoryRequest(BaseModel):
    user_id: str
    action: str = Field(..., description="Action: 'get', 'search', 'clear'")
    query: Optional[str] = None


class MemoryResponse(BaseModel):
    success: bool
    memories: list = []
    message: str = ""


# ============ Endpoints ============
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "active",
        "system": "Lex Bot v2",
        "version": "2.0.0",
        "memory_enabled": MEM0_ENABLED,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Main chat endpoint for legal research queries.
    
    Supports:
    - Query decomposition and parallel agent search
    - User memory for personalization (if user_id provided)
    - Fast/Reasoning LLM modes
    """
    import time
    start_time = time.time()
    
    print(f"📨 API Request: {request.query}")
    if request.user_id:
        print(f"   User: {request.user_id}")
    
    try:
        # Run the query through the graph
        result = run_query(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id or str(uuid.uuid4()),
            llm_mode=request.llm_mode or "fast"
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer=result.get("final_answer", "No answer generated."),
            law_query=result.get("law_query"),
            case_query=result.get("case_query"),
            memory_used=bool(result.get("memory_context")),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory", response_model=MemoryResponse)
async def memory_endpoint(request: MemoryRequest):
    """
    User memory operations.
    
    Actions:
    - get: Get all memories for user
    - search: Search memories with query
    - clear: Clear all memories for user
    """
    if not MEM0_ENABLED:
        return MemoryResponse(
            success=False,
            message="Memory is disabled in configuration"
        )
    
    try:
        memory_manager = UserMemoryManager(user_id=request.user_id)
        
        if request.action == "get":
            memories = memory_manager.get_all()
            return MemoryResponse(
                success=True,
                memories=memories,
                message=f"Retrieved {len(memories)} memories"
            )
        
        elif request.action == "search":
            if not request.query:
                raise HTTPException(status_code=400, detail="Query required for search action")
            memories = memory_manager.search(request.query)
            return MemoryResponse(
                success=True,
                memories=memories,
                message=f"Found {len(memories)} relevant memories"
            )
        
        elif request.action == "clear":
            success = memory_manager.clear_all()
            return MemoryResponse(
                success=success,
                message="Memories cleared" if success else "Failed to clear memories"
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Memory Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Main ============
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Lex Bot v2 Server on Port 8000...")
    print("📚 Memory enabled:", MEM0_ENABLED)
    uvicorn.run(app, host="0.0.0.0", port=8000)
