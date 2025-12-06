import os
from typing import List, Dict, Tuple
import trafilatura
from dotenv import load_dotenv
from tavily import TavilyClient
from ddgs import DDGS

load_dotenv()

# Get API Key: https://tavily.com/
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

preferred_domains = [
        "indiankanoon.org",
        "legalserviceindia.com",
        "scconline.com",
        "livelaw.in",
        "barandbench.com",
        "sci.gov.in",
    ]

def merge_results(res1, res2):
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

def ddgs_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform websearch from DuckDuckGo.
    """
    domain_filter = " OR ".join(f"site:{d}" for d in preferred_domains)

    query = f"{query} ({domain_filter})"

    res=[]
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        for r in results:
            res.append({
                "id":"web",
                "title": r.get('title', 'Unknown Source'),
                "url": r.get('href', '#'),
                "text": r.get('body', '')[:200] + "..." # Preview
            })
    return res

def tavily_search(client, query: str, max_results: int= 5) -> List[Dict]:
    """
    Perform websearch from Tavily.
    """
    res=[]
    response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=[ i for i in preferred_domains ]
            )
    for result in response.get('results', []):
        res.append({
                    "id": "web", 
                    "title": result.get('title', 'Unknown Source'),
                    "url": result.get('url', '#'),
                    "text": result.get('content', '')[:200] + "..." # Preview
                })
    return res
    
def scrape(urls: list[str]) -> str:
    """
    Scrape content from URL using #FISRT Trafilatura -> IF FAILED -> #SECOND Firecarwler.
    """
    context=""
    from firecrawl import Firecrawl
    firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWLER_API_KEY"))
    
    try:
        for url in urls:
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        include_links=True,          # captures citations & references
                        include_formatting=False,
                        include_metadata=True,       # titles/dates useful for RAG
                        favor_precision=True,        # cleaner extraction
                        url=url,
                    )
                    if text:
                        context += str(text) + "\n\n"
                    else:
                        print(f"⚠️ Trafilatura returned empty content for {url}. Falling back to Firecrawl...")
                        raise ValueError("Empty extraction")
                else:
                    print(f"⚠️ Trafilatura could not download {url}. Falling back to Firecrawl...")
                    raise ValueError("Download failed")

            except Exception as te:
                # Fallback: Firecrawl
                try:
                    scrape_result = firecrawl.scrape(url, formats=["markdown"])
                    context += str(scrape_result.markdown) + "\n\n"
                except Exception as fe:
                    print(f"⚠️ Firecrawl failed for {url}: {fe}")
                    continue

        return context

    except Exception as e:
        print(f"❌ Scraping Failed: {e}")
        return ""

class WebSearcher:
    def __init__(self):
        self.client = None
        if TAVILY_API_KEY:
            self.client = TavilyClient(api_key=TAVILY_API_KEY)
        else:
            print("⚠️ TAVILY_API_KEY not found. Web search will be disabled.")

    def search(self, query: str, max_results: int = 5) -> Tuple[str, List[Dict]]:
        """
        Performs a smart search for Indian Legal context.
        Returns:
            - context_text: String formatted for the LLM
            - sources: List of dicts with citations
        """
        

        print(f"🌐 Searching Web for: {query}...")

        res1 = []
        res2 = []
        query= query
        try:
            try: 
                res1=ddgs_search(query=query)
            except Exception as e:
                print(f"❌ DDG Search Failed: {e}")
            
            try:
                res2=tavily_search(self.client, query=query)

                combined_result= merge_results(res1, res2)

                sources= combined_result
                urls=[]
                for src in sources:
                    if 'url' in src:
                        urls.append(src['url'])
                full_context= scrape(urls)
                
                return full_context, sources
            except Exception as e:
                print(f"❌ Web Search Failed: {e}")
                return "", []
                
        except Exception as e:
            print(f"{e}")

# Singleton instance
web_searcher = WebSearcher()

if __name__ =="__main__":
    a,b=web_searcher.search("Can a person be denied bail indefinitely in India?")  # Test call
    print(a)
    print(b)
    # # print(f"############: {b}")
    # print(type(b))
    # print(b[0])
    # # print(type(b[0]))
    # # print(type(a))
    # # print(a)
    # res=ddgs_search("Article 21 Indian Constitution", max_results=5)
    # # print(res)
    # print(type(res))
    # print(res[0])

    #text
#     from firecrawl import Firecrawl
#     firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWLER_API_KEY"))
#     scrape_result = firecrawl.scrape('https://www.barandbench.com/columns/hidden-evidence-imperiled-rights-the-case-for-fair-disclosure', formats=['markdown'])
#     print(scrape_result.markdown)
#     print(type(scrape_result))
#     print(type(scrape_result.markdown))

#     url= 'https://www.barandbench.com/columns/hidden-evidence-imperiled-rights-the-case-for-fair-disclosure'
#     downloaded = trafilatura.fetch_url(url)
#     if downloaded:
#         text = trafilatura.extract(
#         downloaded,
#         include_comments=False,
#         include_tables=True,
#         include_links=True,          # captures citations & references
#         include_formatting=False,
#         favor_precision=True,        # cleaner extraction
#         url=url
# )
#         print(text)
#         print(type(text))
#     else:
#         print("T thingy failed")