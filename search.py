import os
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text as sql
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import hashlib

load_dotenv()
DB_URL = os.getenv('DATABASE_URL')
TEST_MODE = os.getenv('TEST_MODE', '0') == '1'

# --- PRODUCTION CONFIGURATION ---
# Using BGE-M3 (State-of-the-Art)
MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
# Updated dimension to 1024
EMB_DIM = 1024

st_model = None
if not TEST_MODE:
    print(f"🔍 Loading Search Model: {MODEL_NAME}...")
    st_model = SentenceTransformer(MODEL_NAME)

engine = create_engine(DB_URL)

# --- UTILITIES ---

def _local_embed(s: str, dim: int = EMB_DIM) -> List[float]:
    """Mock embedding for testing without GPU."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.normal(size=dim)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.astype(float).tolist()

def qembed(q: str) -> List[float]:
    """Generate embedding for the query."""
    if TEST_MODE:
        return _local_embed(q, dim=EMB_DIM)
    # BGE-M3 works best with a specific instruction for queries, though not strictly required.
    # We stick to standard encoding here.
    emb = st_model.encode([q], normalize_embeddings=True)[0]
    return emb.tolist()

# --- HYBRID SEARCH SQL ---
# 1. Utilizes HNSW index via <=> operator
# 2. Utilizes GIN index via search_vector
# 3. Fetches parent_text for the LLM
HYBRID_SQL = sql("""
WITH q AS (
  SELECT
    -- Use websearch_to_tsquery for advanced operators (quotes, -exclude)
    websearch_to_tsquery('english', :qtext)      AS qtsv,
    CAST(:qemb AS vector)                        AS qemb
)
SELECT 
    p.id, 
    p.doc_id, 
    p.heading, 
    p.text,          -- Child chunk (matched)
    p.parent_text,   -- Parent chunk (context for LLM)
    p.year, 
    p.category,
    r.title,
    
    -- Lexical Score: Calculated against the pre-computed 'search_vector'
    -- This respects the Weight A (Heading) vs Weight B (Text) logic
    ts_rank(p.search_vector, (SELECT qtsv FROM q)) AS lex,
    
    -- Vector Distance: Cosine distance using pgvector
    (p.embedding <=> (SELECT qemb FROM q)) AS distance

FROM passages p
JOIN docs_raw r ON r.id = p.doc_id

-- PRIMARY FILTER: HNSW Vector Search
-- We retrieve more candidates (pre_k * 2) to re-rank with lexical scores later
ORDER BY p.embedding <=> (SELECT qemb FROM q)
LIMIT :pre_k
""")

def hybrid_search(query: str, filters: Dict = None, pre_k=200, mmr_k=20) -> List[Dict]:
    """
    Performs Hybrid Search:
    1. Vector Search (HNSW) to get candidates.
    2. Lexical Scoring (TS_RANK) on candidates.
    3. Z-Score Fusion to combine scores.
    """
    q_emb = qembed(query)
    params = {
        'qtext': query, 
        'qemb': q_emb, 
        'pre_k': pre_k
    }
    
    # --- Dynamic Filtering ---
    # We construct the SQL dynamically to handle optional filters efficiently
    base_sql = HYBRID_SQL.text
    where_conditions = []
    
    if filters:
        if 'year' in filters and filters['year']:
            where_conditions.append("p.year = :f_year")
            params['f_year'] = int(filters['year'])
        if 'category' in filters and filters['category']:
            where_conditions.append("p.category = :f_cat")
            params['f_cat'] = filters['category']

    # Inject WHERE clause into the query before ORDER BY
    if where_conditions:
        # Finding the insertion point before ORDER BY
        split_point = "ORDER BY"
        parts = base_sql.split(split_point)
        # Reconstruct: Part1 + WHERE + AND (if needed) + Part2
        # Note: We append WHERE conditions. 
        # Since the original query has no WHERE, we start with WHERE.
        sql_final = sql(f"{parts[0]} WHERE {' AND '.join(where_conditions)} ORDER BY {parts[1]}")
    else:
        sql_final = HYBRID_SQL

    with Session(engine) as ses:
        rows = ses.execute(sql_final, params).mappings().all()

    if not rows:
        return []

    # --- Score Fusion ---
    # Convert to numpy for fast math
    lex_scores = np.array([float(r['lex']) for r in rows], dtype=np.float32)
    dist_scores = np.array([float(r['distance']) for r in rows], dtype=np.float32)
    
    # Invert distance to similarity (negate) for z-score (smaller distance = higher score)
    sem_scores = -dist_scores 

    def zscore(x: np.ndarray) -> np.ndarray:
        if len(x) < 2: return np.zeros_like(x)
        mu = x.mean()
        sigma = x.std()
        if sigma < 1e-9: # Avoid division by zero
            return np.zeros_like(x)
        return (x - mu) / sigma

    lex_z = zscore(lex_scores)
    sem_z = zscore(sem_scores)

    # Fusion Weight (Alpha). 
    # 0.4 implies slight bias towards Vector search (better for concept matching)
    # Adjust to 0.6 if specific keywords (Section numbers) are more important.
    alpha = 0.4 

    cands: List[Dict] = []
    for i, r in enumerate(rows):
        score = alpha * lex_z[i] + (1.0 - alpha) * sem_z[i]
        
        # KEY LOGIC: Parent-Child resolution
        # We return the Parent Text if available, otherwise the Child Text
        final_context = r['parent_text'] if r['parent_text'] else r['text']
        
        cands.append({
            'id': r['id'],
            'doc_id': r['doc_id'],
            'title': r['title'],
            'heading': r['heading'],
            'text': final_context,     # Return the FULL context for the LLM
            'search_hit': r['text'],   # Debug: The specific chunk that matched
            'year': r['year'],
            'category': r['category'],
            'lex_score': float(lex_scores[i]),
            'sem_score': float(sem_scores[i]),
            'score': float(score),     # Fusion score
        })

    # Sort by Fusion Score
    cands.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top K (MMR logic can be applied here if needed, but Fusion is usually sufficient)
    return cands[:mmr_k]

if __name__ == '__main__':
    # Simple test
    print("Testing Search...")
    results = hybrid_search("rights", pre_k=5, mmr_k=2)
    for r in results:
        print(f"[{r['score']:.4f}] {r['title']} - {r['heading']}")

