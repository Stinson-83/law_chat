import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
RERANK_MODEL = os.getenv('RERANK_MODEL', 'BAAI/bge-reranker-base')
TEST_MODE = os.getenv('TEST_MODE', '0') == '1'

_reranker = None


def get_reranker():
    """
    Lazy-load CrossEncoder reranker.
    Model name comes from RERANK_MODEL env.
    """
    global _reranker
    if TEST_MODE:
        return None
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        # BGE reranker exposes CrossEncoder-compatible interface
        _reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True)
    return _reranker


def _local_score(q: str, t: str) -> float:
    """
    Simple Jaccard over tokens for TEST_MODE / CI.
    """
    qs = set(q.lower().split())
    ts = set(t.lower().split())
    if not qs or not ts:
        return 0.0
    inter = len(qs & ts)
    union = len(qs | ts)
    return inter / union


def _build_text_for_rerank(c: Dict) -> str:
    """
    For legal domain, combine title + heading + passage text.
    """
    title = c.get("title") or ""
    heading = c.get("heading") or ""
    body = c.get("text") or ""
    combined = f"{title}\n\n{heading}\n\n{body}".strip()
    return combined


def rerank(
    query: str,
    candidates: List[Dict],
    top_n: int = 10,
    threshold: Optional[float] = None
) -> List[Dict]:
    """
    Add 'rerank' score using BGE cross-encoder and return top_n.
    Optionally filter by threshold.
    """
    if not candidates:
        return []

    rr = get_reranker()

    if rr is None:  # TEST_MODE
        for c in candidates:
            text_for_score = _build_text_for_rerank(c)
            c['rerank'] = _local_score(query, text_for_score)
    else:
        pairs = [(query, _build_text_for_rerank(c)) for c in candidates]
        scores = rr.predict(pairs)
        scores = scores.tolist() if hasattr(scores, "tolist") else scores
        for c, s in zip(candidates, scores):
            c['rerank'] = float(s)

    candidates.sort(key=lambda x: x['rerank'], reverse=True)
    out = candidates[:top_n]

    if threshold is not None:
        out = [c for c in out if c['rerank'] >= threshold]

    return out

