import os
from dotenv import load_dotenv

load_dotenv()

# --- API KEYS ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Or OPENAI_API_KEY
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FIRECRAWLER_API_KEY = os.getenv("FIRECRAWLER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- MODELS ---
# Using a generic model name constant to easily switch between providers if needed
# For now assuming Gemini or similar via LangChain
LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- SEARCH CONFIG ---
DB_SEARCH_LIMIT_PRE = 200
DB_SEARCH_LIMIT_FINAL = 20
WEB_SEARCH_MAX_RESULTS = 5

# --- TARGET WEBSITES ---
# Placeholder as requested - User to update
TARGET_CASE_SITE = "indiankanoon.org" 

PREFERRED_DOMAINS = [
    "indiankanoon.org",
    "legalserviceindia.com",
    "scconline.com",
    "livelaw.in",
    "barandbench.com",
    "sci.gov.in",
]
