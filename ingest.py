import os, json, hashlib
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models import Base, DocRaw, Passage
from pgvector.sqlalchemy import register_vector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

TEST_MODE = os.getenv('TEST_MODE', '0') == '1'
DB_URL = os.getenv('DATABASE_URL')
MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMB_DIM = 768

# Create model only if not TEST_MODE
st_model = None
if not TEST_MODE:
    st_model = SentenceTransformer(MODEL_NAME)

engine = create_engine(DB_URL)
register_vector(engine)
Base.metadata.create_all(engine)

# --- chunker ---
def simple_chunk(text: str, max_tokens: int = 600) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=st_model,
        chunk_size=max_tokens,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks

def _local_embed(s: str, dim: int = EMB_DIM) -> List[float]:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.normal(size=dim)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.astype(float).tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    if TEST_MODE:
        return [_local_embed(t) for t in texts]

    # self-hosted encoder
    embs = st_model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return embs.tolist()


# --- helpers ---
def make_checksum(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

# --- main ingest ---
def ingest_jsonl(path: str, title_key='title', year_key='year', category_key='category', text_key='text'):
    with engine.begin() as conn:
        Base.metadata.create_all(conn)
    with Session(engine) as ses:
        with open(path, 'r', encoding='utf-8') as f:
            batch_passages: List[Tuple[DocRaw, List[Dict]]] = []
            for line in f:
                rec = json.loads(line)
                title = rec.get(title_key)
                year = rec.get(year_key)
                category = rec.get(category_key)
                full_text = rec.get(text_key) or ''

                doc = DocRaw(filename=os.path.basename(path), title=title, year=year, category=category, data=rec)
                ses.add(doc)
                ses.flush()  # get doc.id

                units = []
                chunks = simple_chunk(full_text, max_tokens=600)
                for ch in chunks:
                    heading = None
                    inp = f"{title or ''}\n{heading or ''}\n{ch}".strip()
                    units.append({
                        'doc_id': doc.id,
                        'section_no': None,
                        'heading': heading,
                        'text': ch,
                        'year': year,
                        'category': category,
                        'token_count': len(ch.split()),
                        'checksum': make_checksum(inp)
                    })
                batch_passages.append((doc, units))

            # Flatten for embeddings
            flat_texts = [f"{doc.title or ''}\n{u['heading'] or ''}\n{u['text']}".strip()
                           for _, units in batch_passages for u in units]
            embs = embed_texts(flat_texts)
            i = 0
            for _, units in batch_passages:
                for u in units:
                    u['embedding'] = np.array(embs[i]).tolist(); i += 1
                    ses.add(Passage(**u))
            ses.commit()

if __name__ == '__main__':
    # Example: python ingest.py /path/to/file.jsonl
    import sys
    ingest_jsonl(sys.argv[1])