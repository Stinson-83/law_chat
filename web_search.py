import os
import logging
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import trafilatura
from dotenv import load_dotenv
from tavily import TavilyClient
from ddgs import DDGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('trafilatura').setLevel(logging.ERROR)

load_dotenv()

# Constants
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FIRECRAWLER_API_KEY = os.getenv("FIRECRAWLER_API_KEY")

PREFERRED_DOMAINS = [
    "indiankanoon.org",
    "legalserviceindia.com",
    "scconline.com",
    "livelaw.in",
    "barandbench.com",
    "sci.gov.in",
]

class WebSearcher:
    """
    A unified class for performing web searches and scraping content relative to Indian Law.
    """
    def __init__(self):
        self.tavily_client = None
        if TAVILY_API_KEY:
            self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        else:
            logger.warning("⚠️ TAVILY_API_KEY not found. Web search via Tavily will be disabled.")
        
        self.firecrawl = None
        if FIRECRAWLER_API_KEY:
            try:
                from firecrawl import FirecrawlApp
                self.firecrawl = FirecrawlApp(api_key=FIRECRAWLER_API_KEY)
            except ImportError:
                 try:
                    from firecrawl import Firecrawl
                    self.firecrawl = Firecrawl(api_key=FIRECRAWLER_API_KEY)
                 except ImportError:
                    logger.warning("Could not import Firecrawl. Fallback scraping disabled.")
            except Exception as e:
                logger.warning(f"Failed to initialize Firecrawl: {e}")

    @staticmethod
    def _merge_results(res1: List[Dict], res2: List[Dict]) -> List[Dict]:
        """Merge two lists of search results, removing duplicates based on URL."""
        seen = set()
        combined = []

        for item in res1 + res2:
            url = item.get("href") or item.get("url")
            if not url:
                continue

            if url not in seen:
                seen.add(url)
                combined.append(item)

        return combined

    def _ddgs_search(self, query: str, max_results: int = 5, domains: List[str] = None) -> List[Dict]:
        """Perform web search using DuckDuckGo."""
        try:
            target_domains = domains if domains else PREFERRED_DOMAINS
            domain_filter = " OR ".join(f"site:{d}" for d in target_domains)
            full_query = f"{query} ({domain_filter})"
            
            res = []
            with DDGS() as ddgs:
                results = ddgs.text(full_query, max_results=max_results)
                for r in results:
                    res.append({
                        "id": "web",
                        "title": r.get('title', 'Unknown Source'),
                        "url": r.get('href', '#'),
                        "text": r.get('body', '')[:200] + "..." # Preview
                    })
            return res
        except Exception as e:
            logger.error(f"❌ DDG Search Failed: {e}")
            return []

    def _tavily_search(self, query: str, max_results: int = 5, domains: List[str] = None) -> List[Dict]:
        """Perform web search using Tavily."""
        if not self.tavily_client:
            return []
        
        try:
            target_domains = domains if domains else PREFERRED_DOMAINS
            res = []
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=target_domains
            )
            for result in response.get('results', []):
                res.append({
                    "id": "web", 
                    "title": result.get('title', 'Unknown Source'),
                    "url": result.get('url', '#'),
                    "text": result.get('content', '')[:200] + "..." # Preview
                })
            return res
        except Exception as e:
            logger.error(f"❌ Tavily Search Failed: {e}")
            return []

    def _scrape_single_url(self, url: str) -> str:
        """Helper to scrape a single URL with fallback."""
        try:
            # 1. Try Trafilatura
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=True,
                    include_links=True,
                    include_formatting=False,
                    # include_metadata=True, # Removed causing TypeError
                    favor_precision=True,
                    url=url,
                )
                if text:
                    return f"Source: {url}\n\n{str(text)}\n\n"
            
            logger.warning(f"⚠️ Trafilatura failed/empty for {url}. Trying Firecrawl...")

            # 2. Fallback to Firecrawl
            if self.firecrawl:
                try:
                    # Note: handling different versions of firecrawl if method signatures vary
                    # Assuming v2 based on previous stack trace showing v2 path
                    # Trying to be robust to v1/v2 differences if possible, or sticking to what seemed to be used
                    # Original code used: firecrawl.scrape(url, formats=["markdown"])
                    # If using v2 client, it might be scrape_url or just scrape. 
                    # Let's inspect the original error again: 
                    # File "C:\Users\AYUSH\law_chat\venv\Lib\site-packages\firecrawl\v2\client.py", line 177, in scrape
                    # So 'scrape' exists on the client.
                    if hasattr(self.firecrawl, 'scrape'):
                         scrape_result = self.firecrawl.scrape(url, formats=["markdown"])
                    elif hasattr(self.firecrawl, 'scrape_url'):
                         scrape_result = self.firecrawl.scrape_url(url, params={"formats": ["markdown"]})
                    else:
                         raise AttributeError("Firecrawl client has no known scrape method")

                    # Check structure of response
                    if isinstance(scrape_result, dict) and 'markdown' in scrape_result:
                         return f"Source: {url}\n\n{scrape_result['markdown']}\n\n"
                    elif hasattr(scrape_result, 'markdown'):
                         return f"Source: {url}\n\n{scrape_result.markdown}\n\n"
                    else:
                         # last ditch attempt to just stringify
                         return f"Source: {url}\n\n{str(scrape_result)}\n\n"
                    
                except Exception as fe:
                    logger.warning(f"⚠️ Firecrawl failed for {url}: {fe}")
            
        except Exception as e:
            logger.error(f"❌ Scraping Failed for {url}: {e}")
        
        return ""

    def scrape_urls(self, urls: List[str]) -> str:
        """
        Scrape content from multiple URLs concurrently.
        """
        context = ""
        # Deduplicate URLs
        unique_urls = list(set(urls))
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self._scrape_single_url, url): url for url in unique_urls}
            for future in as_completed(future_to_url):
                try:
                    data = future.result()
                    if data:
                        context += data
                except Exception as exc:
                    url = future_to_url[future]
                    logger.error(f"Generated an exception for {url}: {exc}")
        
        return context

    def search(self, query: str, max_results: int = 5, domains: List[str] = None) -> Tuple[str, List[Dict]]:
        """
        Performs a smart search for Indian Legal context.
        Args:
            query: The search query.
            max_results: Max results per search engine.
            domains: Optional list of domains to restrict search to. If None, uses PREFERRED_DOMAINS.
        Returns:
            - context_text: String formatted for the LLM
            - sources: List of dicts with citations
        """
        logger.info(f"🌐 Searching Web for: {query} (Domains: {domains if domains else 'Default'})")

        # Run searches (could also be parallelized if needed, but they are fast enough usually)
        # We can run them in parallel for optimization
        res1 = []
        res2 = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_ddg = executor.submit(self._ddgs_search, query, max_results, domains)
            future_tavily = executor.submit(self._tavily_search, query, max_results, domains)
            
            res1 = future_ddg.result()
            res2 = future_tavily.result()

        sources = self._merge_results(res1, res2)
        
        urls = [src['url'] for src in sources if 'url' in src]
        full_context = self.scrape_urls(urls)
        
        return full_context, sources

# Singleton instance
web_searcher = WebSearcher()

if __name__ =="__main__":
    # Test call
    query = "What are the fundamental rights in india?"
    context, sources = web_searcher.search(query)
    
    print("\n--- SOURCES ---")
    for s in sources:
        print(f"- {s['title']} ({s['url']})")
    
    print("\n--- CONTEXT SNIPPET ---")
    print(len(context))