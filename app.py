import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any

# Import Agents
from agents.manager import ManagerAgent
from agents.law_agent import LawAgent
from agents.case_agent import CaseAgent

# Import legacy components for raw search endpoint (debugging)
from search import hybrid_search
from rerank import rerank

# Load Environment
from dotenv import load_dotenv
load_dotenv()

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- AGENT INITIALIZATION ---
try:
    law_agent = LawAgent()
    case_agent = CaseAgent()
    manager_agent = ManagerAgent(law_agent=law_agent, case_agent=case_agent)
    logger.info("🤖 Agents Initialized Successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize agents: {e}")
    raise e


app = FastAPI(
    title="Legal RAG API (Agentic)", 
    description="Advanced Agentic RAG for Indian Law. Routes queries to specialized agents."
)

# --- DATA MODELS ---

class FilterParams(BaseModel):
    year: Optional[int] = None
    category: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    filters: Optional[FilterParams] = None
    top_n: int = 10

class AnswerRequest(BaseModel):
    query: str
    filters: Optional[FilterParams] = None
    top_n: int = 5

class Source(BaseModel):
    id: Any
    title: str
    text: str
    heading: Optional[str] = None
    score: Optional[float] = None
    url: Optional[str] = None
    type: str 

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]
    source_type: str 
    confidence: float

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "mode": "Agentic"}

@app.post("/search")
async def search_endpoint(payload: SearchRequest):
    """
    Raw Search Endpoint: Keeps the legacy directly calling search/rerank for specific debugging.
    """
    # 1. Hybrid Search (Database)
    candidates = hybrid_search(
        payload.query, 
        filters=payload.filters.dict() if payload.filters else None,
        pre_k=100
    )
    
    # 2. Cross-Encoder Reranking
    ranked = rerank(payload.query, candidates, top_n=payload.top_n)
    
    return {"results": ranked}

@app.post("/answer", response_model=AnswerResponse)
async def answer_endpoint(payload: AnswerRequest):
    """
    Agentic RAG Endpoint:
    Routes query via ManagerAgent -> LawAgent/CaseAgent.
    """
    logger.info(f"📝 Agentic Processing: {payload.query}")
    
    try:
        # Delegate to Manager Agent
        result = manager_agent.process(
            payload.query, 
            filters=payload.filters.dict() if payload.filters else None, 
            top_n=payload.top_n
        )
        
        # Convert dictionary result to Pydantic Model
        sources = []
        for s in result.get('sources', []):
            sources.append(Source(
                id=s.get('id', 'unknown'),
                title=s.get('title', 'Unknown'),
                text=s.get('text', ''),
                heading=s.get('heading'),
                score=s.get('score', 0.0),
                url=s.get('url'),
                type=s.get('type', 'unknown')
            ))
            
        return AnswerResponse(
            answer=result.get('answer', "No answer generated."),
            sources=sources,
            source_type=result.get('source_type', 'agent'),
            confidence=result.get('confidence', 0.0)
        )
        
    except Exception as e:
        logger.error(f"Agent Execution Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)