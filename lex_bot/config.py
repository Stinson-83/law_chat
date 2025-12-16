import os
from dotenv import load_dotenv

load_dotenv()

# --- API KEYS ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FIRECRAWLER_API_KEY = os.getenv("FIRECRAWLER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
JHANA_API_KEY = os.getenv("JHANA_API_KEY")  # Jhana.ai (when available)

# --- LLM MODES ---
# "fast" = quick responses, lower cost
# "reasoning" = complex analysis, higher accuracy
LLM_MODE = os.getenv("LLM_MODE", "fast")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "openai"

# Fast Mode Models
GEMINI_FAST_MODEL = "gemini-1.5-flash"
OPENAI_FAST_MODEL = "gpt-4o-mini"

# Reasoning Mode Models
GEMINI_REASONING_MODEL = "gemini-1.5-pro"
OPENAI_REASONING_MODEL = "gpt-4o"

# Legacy compatibility
LLM_MODEL_NAME = GEMINI_FAST_MODEL

# --- EMBEDDING MODEL ---
EMBEDDING_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- SEARCH CONFIG ---
DB_SEARCH_LIMIT_PRE = 200
DB_SEARCH_LIMIT_FINAL = 20
WEB_SEARCH_MAX_RESULTS = 5

# --- RATE LIMITING ---
SCRAPE_DELAY_SECONDS = float(os.getenv("SCRAPE_DELAY", 2.5))

# --- MEMORY (mem0) ---
MEM0_ENABLED = os.getenv("MEM0_ENABLED", "true").lower() == "true"

# --- SESSION CACHE ---
SESSION_CACHE_TTL_MINUTES = int(os.getenv("SESSION_CACHE_TTL", 30))

# --- TARGET WEBSITES ---
TARGET_CASE_SITE = "indiankanoon.org"

PREFERRED_DOMAINS = [
    "indiankanoon.org",
    "legalserviceindia.com",
    "scconline.com",
    "livelaw.in",
    "barandbench.com",
    "sci.gov.in",
    "indiacode.nic.in",
    "ecourts.gov.in",
]

