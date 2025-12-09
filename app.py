import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any

# Import our custom modules
from search import hybrid_search
from rerank import rerank
from web_search import web_searcher

# Load Environment
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# Use the PRO model for better legal reasoning and reduced hallucinations
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-1.5-pro")

# Threshold: If the top Database result has a rerank score below this, trigger Web Search.
# Based on BGE-M3 sigmoid normalization:
# > 0.7 = Very High Confidence (Exact match)
# > 0.45 = Moderate Confidence (Likely relevant)
# < 0.45 = Low Confidence (Probably irrelevant)
SEARCH_THRESHOLD = 0.45

app = FastAPI(
    title="Legal RAG API", 
    description="Advanced RAG for Indian Constitutional Law with Hybrid Search, Reranking, and Web Fallback."
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
    type: str  # "db" or "web"

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]
    source_type: str  # "database", "web", or "hybrid"
    confidence: float

# --- CORE LOGIC ---

def get_gemini_response(query: str, context: str, source_type: str) -> str:
    """
    Constructs the specialized legal prompt and calls Google Gemini.
    """
    model = genai.GenerativeModel(LLM_MODEL_NAME)
    
    system_prompt = f"""
    You are an expert Legal AI Assistant for Indian Constitutional Law.
    
    CONTEXT SOURCE: {source_type.upper()}
    
    INSTRUCTIONS:
    1. Answer the USER QUERY specifically using the provided CONTEXT.
    2. CITATIONS:
       - If context is from 'DATABASE': Cite specific Sections, Articles, or Case Names.
       - If context is from 'WEB': Cite the website/domain provided.
    3. TONE: Professional, objective, and legally precise.
    4. FALLBACK: If the answer is not in the context, clearly state: "I cannot find specific legal provisions for this in my current knowledge base."
    
    CONTEXT:
    {context}
    
    USER QUERY: {query}
    """
    
# def gemni_enhance_query(query: str) -> Tuple[str, List[str]]:
#     """
#     Takes in the User query and gives out filter parameters for DB search.
#     """
#     try:
#         response = model.generate_content(system_prompt)
#         return response.text
#     except Exception as e:
#         print(f"Gemini API Error: {e}")
#         return "I encountered an error while generating the answer. Please try again."

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "model": LLM_MODEL_NAME}

@app.post("/search")
async def search_endpoint(payload: SearchRequest):
    """
    Raw Search Endpoint: Returns ranked documents without LLM generation.
    Useful for debugging the retrieval pipeline.
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
    Production RAG Endpoint:
    1. Searches Database.
    2. Reranks results.
    3. Checks Confidence Score.
    4. Falls back to Web Search if confidence is low.
    5. Generates Answer via LLM.
    """
    print(f"📝 Processing Query: {payload.query}")
    
    # --- PHASE 1: DATABASE RETRIEVAL ---
    # Fetch broad candidates
    candidates = hybrid_search(
        payload.query, 
        filters=payload.filters.dict() if payload.filters else None,
        pre_k=200 
    )
    
    # Rerank top N
    ranked_results = rerank(payload.query, candidates, top_n=payload.top_n)
    
    # Determine Confidence
    top_score = ranked_results[0]['rerank'] if ranked_results else 0.0
    print(f"📊 Top DB Score: {top_score:.4f} (Threshold: {SEARCH_THRESHOLD})")
    
    final_context = ""
    final_sources = []
    source_origin = "database"

    # --- PHASE 2: ROUTING LOGIC ---
    
    if top_score >= SEARCH_THRESHOLD:
        # CASE A: Database is confident -> Use DB results
        source_origin = "database"
        
        context_blocks = []
        for r in ranked_results:
            # Logic from search.py: r['text'] is the PARENT (Full Context)
            # r['search_hit'] is the CHILD (Specific Clause)
            
            # Context for LLM (Parent)
            context_text = r.get('text', '') 
            meta = f"{r['title']} - {r['heading']}"
            context_blocks.append(f"DOCUMENT: {meta}\nTEXT: {context_text}")
            
            # Source for UI (Child Preview)
            preview_text = r.get('search_hit', context_text)
            
            final_sources.append(Source(
                id=r['id'],
                title=r['title'],
                heading=r['heading'],
                text=preview_text[:300] + "...", # UI Preview
                score=r['rerank'],
                type="db"
            ))
            
        final_context = "\n\n".join(context_blocks)
        
    else:
        # CASE B: Database is unsure -> Fallback to Web
        print("⚠️ Low confidence in DB. Triggering Web Search.")
        source_origin = "web"
        
        web_context, web_hits = web_searcher.search(payload.query)
        
        if not web_context:
            # Worst case: DB failed AND Web failed. 
            # Fallback to whatever weak DB results we had, if any.
            if ranked_results:
                print("⚠️ Web search failed. Falling back to weak DB results.")
                source_origin = "database (low confidence)"
                final_context = "\n".join([r['text'] for r in ranked_results])
                final_sources = [
                    Source(
                        id=r['id'], 
                        title=r['title'], 
                        text=r.get('search_hit', '')[:200], 
                        type="db", 
                        score=r['rerank']
                    ) for r in ranked_results
                ]
            else:
                # Absolute failure
                return AnswerResponse(
                    answer="I could not find relevant legal information in the database or reliable web sources.",
                    sources=[],
                    source_type="none",
                    confidence=0.0
                )
        else:
            # Success: Use Web Context
            final_context = web_context
            for hit in web_hits:
                final_sources.append(Source(
                    id="web",
                    title=hit['title'],
                    url=hit['url'],
                    text=hit['text'], # Snippet
                    type="web"
                ))

    # --- PHASE 3: GENERATION ---
    try:
        answer_text = get_gemini_response(payload.query, final_context, source_origin)
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        raise HTTPException(status_code=500, detail="Error generating answer from LLM")

    return AnswerResponse(
        answer=answer_text,
        sources=final_sources,
        source_type=source_origin,
        confidence=top_score
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)