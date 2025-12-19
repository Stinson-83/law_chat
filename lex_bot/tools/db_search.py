import os
import logging
from typing import List, Dict, Optional, Tuple
from ..config import DATABASE_URL, EMBEDDING_MODEL_NAME, DB_SEARCH_LIMIT_PRE
from .web_search import web_search_tool

# Configure logging
logger = logging.getLogger(__name__)

from lex_bot.core.tool_registry import register_tool

@register_tool(
    name="db_search",
    capabilities=["statute_lookup", "law_search"],
    description="Search local database for statutes and acts",
    requires_rate_limit=False
)
class SearchTool:
    def __init__(self):
        self.engine = None
        self.model = None
        
        # Lazy Import / Helper
        self._init_resources()

    def _init_resources(self):
        # 1. DB Engine - DISABLED
        # if DATABASE_URL:
        #     try:
        #         from sqlalchemy import create_engine
        #         self.engine = create_engine(DATABASE_URL)
        #         # Test connection logic could be here
        #     except ImportError:
        #         logger.error("SQLAlchemy not installed.")
        #     except Exception as e:
        #         logger.error(f"❌ DB Init Failed: {e}")
        
        # 2. Embedding Model - DISABLED
        # try:
        #     from sentence_transformers import SentenceTransformer
        #     print(f"🔍 Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        #     self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # except ImportError:
        #     logger.error("SentenceTransformers not installed.")
        # except Exception as e:
        #     logger.error(f"❌ Model Loading Failed: {e}")
        pass

    def _get_embedding(self, query: str) -> List[float]:
        if not self.model:
            return []
        return self.model.encode([query], normalize_embeddings=True)[0].tolist()

    def _hybrid_db_search(self, query: str) -> List[Dict]:
        # Disabled
        return []

    def run(self, query: str, domains: List[str] = None) -> Tuple[str, List[Dict]]:
        """
        Attempts DB Search. If fails/empty -> Web Search.
        """
        logger.info(f"🔎 SearchTool called for: {query}")
        
        # 1. Try DB Search - DISABLED
        # db_results = self._hybrid_db_search(query)
        
        # if db_results:
        #     logger.info(f"✅ DB Search returned {len(db_results)} results.")
        #     context = ""
        #     for r in db_results[:10]:
        #         context += f"Source: {r['title']} > {r['heading']}\n{r['text']}\n\n"
        #     return context, db_results
        
        logger.info("⚠️ DB Search disabled. Falling back to Web...")

        # 2. Fallback to Web Search
        return web_search_tool.run(query, domains)

search_tool = SearchTool()
