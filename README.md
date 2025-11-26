# RAG System with Hybrid Search & Reranking

A production-ready Retrieval-Augmented Generation (RAG) system that combines hybrid search (lexical + semantic), cross-encoder reranking, and LLM-powered question answering for document retrieval and analysis.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Workflow](#workflow)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Hybrid Search Engine**: Combines BM25-style lexical search with dense vector similarity using pgvector
- **Cross-Encoder Reranking**: BGE reranker for improved result quality and relevance
- **LLM-Powered Q&A**: Google Gemini integration for answer generation with source citations
- **Scalable Vector Storage**: PostgreSQL with pgvector extension and IVFFLAT indexing
- **Flexible Document Ingestion**: Support for JSONL format with metadata (title, year, category)
- **REST API**: FastAPI-based async API with Pydantic validation
- **Smart Chunking**: Recursive character text splitting with configurable overlap
- **Filter Support**: Query-time filtering by year, category, or custom fields

## 🏗️ Architecture

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   Hybrid Search             │
│  ┌──────────┐ ┌──────────┐ │
│  │ Lexical  │ │ Semantic │ │
│  │ (BM25)   │ │ (Vector) │ │
│  └────┬─────┘ └─────┬────┘ │
│       └───────┬─────┘       │
│           Fusion (α=0.5)    │
└───────────┬─────────────────┘
            │ (200 candidates)
            ▼
┌─────────────────────┐
│  MMR Diversity      │
│  (removes duplicates)│
└──────────┬──────────┘
           │ (20 diverse)
           ▼
┌─────────────────────┐
│  Cross-Encoder      │
│  Reranking (BGE)    │
└──────────┬──────────┘
           │ (top 8)
           ▼
┌─────────────────────┐
│  LLM Answer Gen     │
│  (Google Gemini)    │
└─────────────────────┘
```

## 📁 Project Structure

```
rag-system/
│
├── app.py                  # FastAPI application with /search and /answer endpoints
├── models.py               # SQLAlchemy ORM models (DocRaw, Passage)
├── search.py               # Hybrid search implementation (lexical + semantic)
├── rerank.py               # Cross-encoder reranking logic
├── ingest.py               # Document ingestion pipeline
├── schema.sql              # PostgreSQL schema with pgvector setup
├── requirements.txt        # Python dependencies
├── .env                    # Environment configuration (not in repo)
│
├── data/                   # Data directory (create this)
│   └── documents.jsonl     # Your input documents
│
└── tests/                  # Test files (optional)
    └── test_api.py
```

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **PostgreSQL**: 14+ with pgvector extension
- **API Keys**: Google API key for Gemini LLM
- **RAM**: 4GB+ recommended for embedding models
- **Disk**: ~2GB for models (BGE embedder + reranker)

## 🚀 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd rag-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up PostgreSQL

Install PostgreSQL and pgvector:

```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-14-pgvector

# macOS (Homebrew)
brew install postgresql pgvector

# Start PostgreSQL
sudo service postgresql start  # Linux
brew services start postgresql  # macOS
```

Create database and apply schema:

```bash
createdb rag_db
psql -U postgres -d rag_db -f schema.sql
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db

# Google Gemini API
GOOGLE_API_KEY=your_gemini_api_key_here

# Embedding Model (768-dim)
EMBED_MODEL=BAAI/bge-base-en-v1.5

# Reranking Model
RERANK_MODEL=BAAI/bge-reranker-base

# LLM Model
LLM_MODEL=gemini-2.0-flash-exp

# Test Mode (set to 1 for testing without models)
TEST_MODE=0
```

### Getting a Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste into your `.env` file

## 📝 Usage

### Step 1: Prepare Your Data

Create a JSONL file (`data/documents.jsonl`) where each line is a JSON object:

```jsonl
{"title": "Contract Law Basics", "year": 2024, "category": "legal", "text": "A contract is a legally binding agreement..."}
{"title": "Property Rights Guide", "year": 2023, "category": "legal", "text": "Property rights define ownership..."}
{"title": "Tax Regulations 2024", "year": 2024, "category": "finance", "text": "The following tax provisions apply..."}
```

**Required fields:**
- `text`: Full document content
- `title`: Document title (optional but recommended)
- `year`: Publication year (optional, for filtering)
- `category`: Document category (optional, for filtering)

### Step 2: Ingest Documents

```bash
python ingest.py data/documents.jsonl
```

**What happens during ingestion:**
1. Documents are split into 512-token chunks (100 overlap)
2. Each chunk gets embedded using BGE-base-en-v1.5 (768 dimensions)
3. Chunks stored in PostgreSQL with vector indexes
4. Full-text search indexes created automatically

**Progress output:**
```
Processing documents...
✓ Embedded 150 passages
✓ Created vector indexes
✓ Ingestion complete: 25 documents, 150 passages
```

### Step 3: Start the API Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Server will start at:** `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs` (Swagger UI)

### Step 4: Query the System

#### Using cURL

**Search endpoint:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are property rights?",
    "filters": {"category": "legal"},
    "top_n": 5
  }'
```

**Answer endpoint (with LLM):**
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain contract law basics",
    "filters": {"year": 2024},
    "top_n": 8
  }'
```

#### Using Python

```python
import requests

# Search for relevant passages
response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "tax regulations",
        "filters": {"category": "finance"},
        "pre_k": 200,
        "mmr_k": 20,
        "top_n": 8
    }
)
results = response.json()['results']

# Get LLM-generated answer
response = requests.post(
    "http://localhost:8000/answer",
    json={
        "query": "What are the key tax regulations for 2024?",
        "filters": {"year": 2024},
        "top_n": 8
    }
)
answer_data = response.json()
print(answer_data['answer'])
print(answer_data['citations'])
```

## 🔌 API Reference

### POST `/search`

Retrieve relevant passages using hybrid search and reranking.

**Request Body:**
```json
{
  "query": "string (required)",
  "filters": {
    "year": 2024,
    "category": "legal"
  },
  "pre_k": 200,        // Initial candidates from hybrid search
  "mmr_k": 20,         // Diverse candidates after MMR
  "top_n": 8,          // Final reranked results
  "threshold": 0.5     // Optional minimum rerank score
}
```

**Response:**
```json
{
  "results": [
    {
      "id": 123,
      "doc_id": 45,
      "title": "Document Title",
      "heading": "Section 2.1",
      "text": "Relevant passage text...",
      "year": 2024,
      "category": "legal",
      "lex": 0.85,       // Lexical score
      "distance": 0.23,  // Vector distance
      "sem": -0.23,      // Semantic score
      "score": 0.72,     // Hybrid fusion score
      "rerank": 0.89     // Cross-encoder score
    }
  ]
}
```

### POST `/answer`

Generate an answer using retrieved context and Google Gemini.

**Request Body:**
```json
{
  "query": "string (required)",
  "filters": {
    "year": 2024
  },
  "top_n": 8
}
```

**Response:**
```json
{
  "answer": "Based on the provided context, property rights define... [Source 1] [Source 3]",
  "citations": [
    {
      "id": 123,
      "title": "Property Rights Guide",
      "text": "Property rights define...",
      "rerank": 0.92
    }
  ],
  "model": "gemini-2.0-flash-exp"
}
```

## 🔄 Workflow

### Document Processing Pipeline

```
Input JSONL → Parse Records → Chunk Text (512 tokens) → Generate Embeddings
                                                               ↓
                                                        Store in DB
                                                               ↓
                                                    Create Vector Index
```

### Query Pipeline

```
User Query → Generate Query Embedding
                    ↓
    ┌───────────────┴──────────────┐
    ▼                              ▼
Lexical Search              Vector Search
(PostgreSQL FTS)           (pgvector <=>)
    │                              │
    └───────────┬──────────────────┘
                ▼
        Hybrid Fusion (α=0.5)
                ↓
         Z-score normalize
                ↓
        Top 200 candidates
                ↓
        MMR diversity (20)
                ↓
    Cross-encoder reranking
                ↓
          Top N results
                ↓
      [Optional] LLM Answer
```

### Scoring Details

1. **Lexical Score**: BM25-style ranking using PostgreSQL `ts_rank_cd`
2. **Semantic Score**: Negative cosine distance from pgvector
3. **Hybrid Score**: `α * lex_zscore + (1-α) * sem_zscore` where α=0.5
4. **Rerank Score**: BGE cross-encoder confidence (higher = more relevant)

## 🧪 Testing

### Run Tests

```bash
pytest tests/
```

### Test Mode

For CI/CD or environments without GPU:

```bash
export TEST_MODE=1
python ingest.py data/test.jsonl
uvicorn app:app --port 8000
```

In TEST_MODE:
- Embeddings use deterministic hashing (no model loading)
- Reranking uses simple Jaccard similarity
- LLM returns mock responses

### Manual Testing

```bash
# Test ingestion
python ingest.py data/sample.jsonl

# Test search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_n": 3}'

# Test answer generation
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "explain this topic", "top_n": 5}'
```

## 🔧 Troubleshooting

### Common Issues

**1. pgvector extension not found**
```sql
-- Connect to your database and run:
CREATE EXTENSION vector;
```

**2. Out of memory during ingestion**
```python
# Reduce batch size in ingest.py
embs = st_model.encode(texts, batch_size=32)  # default: 64
```

**3. Slow vector search**
```sql
-- Rebuild index with more lists for larger datasets
DROP INDEX passages_vec_idx;
CREATE INDEX passages_vec_idx ON passages
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 2000);

-- Run ANALYZE
ANALYZE passages;
```

**4. Google API key errors**
```bash
# Verify key is set
echo $GOOGLE_API_KEY

# Check quota at: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com
```

**5. Model download issues**
```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
```

### Performance Tuning

**For large datasets (100K+ passages):**

1. Increase IVFFLAT lists:
```sql
CREATE INDEX passages_vec_idx ON passages
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 5000);
```

2. Adjust search parameters:
```python
# In API calls
pre_k = 500  # More initial candidates
mmr_k = 50   # More diverse results
```

3. Enable connection pooling:
```python
# In search.py and app.py
engine = create_engine(DB_URL, pool_size=20, max_overflow=40)
```

## 📚 Additional Resources

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [Google Gemini API](https://ai.google.dev/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📄 License

[Add your license here]

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

**Questions?** Open an issue or contact [your-email@example.c