import os
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text as sql
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import hashlib

load_dotenv()
DB_URL = os.getenv('DATABASE_URL')
TEST_MODE = os.getenv('TEST_MODE', '0') == '1'

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMB_DIM = 768
st_model = None
if not TEST_MODE:
    st_model = SentenceTransformer(MODEL_NAME)

engine = create_engine(DB_URL)

# Calculate embedding for query

def _local_embed(s: str, dim: int = EMB_DIM) -> List[float]:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.normal(size=dim)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.astype(float).tolist()


def qembed(q: str) -> List[float]:
    if TEST_MODE:
        return _local_embed(q, dim=EMB_DIM)
    emb = st_model.encode([q], normalize_embeddings=True)[0]
    return emb.tolist()

# Hybrid search SQL: filter lexically, order by vector distance; return lex and sem scores
HYBRID_SQL = sql("""
WITH q AS (
  SELECT
    to_tsvector('english', :qtext)               AS qtsv,
    :qemb::vector                                AS qemb
), cand AS (
  SELECT p.id, p.doc_id, p.heading, p.text, p.year, p.category,
         r.title,
         -- lexical score
         ts_rank_cd(to_tsvector('english', p.text), plainto_tsquery(:qtext)) AS lex,
         -- vector distance
         (p.embedding <=> (SELECT qemb FROM q)) AS distance
  FROM passages p
  JOIN docs_raw r ON r.id = p.doc_id
  WHERE to_tsvector('english', p.text) @@ plainto_tsquery(:qtext)
  ORDER BY p.embedding <=> (SELECT qemb FROM q)
  LIMIT :pre_k
)
SELECT * FROM cand;
""")

def hybrid_search(query: str, filters: Dict = None, pre_k=200, mmr_k=20) -> List[Dict]:
    q_emb = qembed(query)
    params = {'qtext': query, 'qemb': np.array(q_emb), 'pre_k': pre_k}
    where_extra = []
    if filters:
        if 'year' in filters:
            where_extra.append('year = :f_year')
            params['f_year'] = int(filters['year'])
        if 'category' in filters:
            where_extra.append('category = :f_cat')
            params['f_cat'] = filters['category']

    sql_stmt = HYBRID_SQL
    if where_extra:
        sql_str = HYBRID_SQL.text.replace(
            'WHERE to_tsvector',
            'WHERE ' + ' AND '.join(where_extra) + ' AND to_tsvector'
        )
        sql_stmt = sql(sql_str)

    with Session(engine) as ses:
        rows = ses.execute(sql_stmt, params).mappings().all()

    if not rows:
        return []

    # raw scores
    lex_scores = np.array([float(r['lex']) for r in rows], dtype=np.float32)
    # pgvector <=> is a distance. smaller is better. convert to similarity proxy
    dist_scores = np.array([float(r['distance']) for r in rows], dtype=np.float32)
    sem_scores = -dist_scores  # monotonic, scale will be normalised anyway

    def zscore(x: np.ndarray) -> np.ndarray:
        mu = x.mean()
        sigma = x.std()
        if sigma == 0:
            return np.zeros_like(x)
        return (x - mu) / sigma

    lex_z = zscore(lex_scores)
    sem_z = zscore(sem_scores)

    alpha = 0.5  # tune this

    cands: List[Dict] = []
    for i, r in enumerate(rows):
        score = alpha * lex_z[i] + (1.0 - alpha) * sem_z[i]
        cands.append({
            'id': r['id'],
            'doc_id': r['doc_id'],
            'title': r['title'],
            'heading': r['heading'],
            'text': r['text'],
            'year': r['year'],
            'category': r['category'],
            'lex': float(lex_scores[i]),
            'distance': float(dist_scores[i]),
            'sem': float(sem_scores[i]),
            'score': float(score),
        })

    cands.sort(key=lambda x: x['score'], reverse=True)
    return cands[:mmr_k]

