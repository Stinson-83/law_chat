import os
import logging
import google.generativeai as genai
from typing import Dict, Any, List
from .base_agent import BaseAgent
from dotenv import load_dotenv

# Load Environment
load_dotenv()
logger = logging.getLogger(__name__)

class ManagerAgent(BaseAgent):
    """
    The orchestrator agent.
    Classifies the user query and routes it to the LawAgent or CaseAgent.
    """
    
    def __init__(self, law_agent: BaseAgent = None, case_agent: BaseAgent = None):
        self.law_agent = law_agent
        self.case_agent = case_agent
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def _classify_query(self, query: str) -> str:
        """
        Determines if the query is a General Law query or a Specific Case query.
        Returns: 'LAW', 'CASE', or 'BOTH'
        """
        prompt = f"""
        Role: Legal Query Classifier
        
        Analyze the following user query and classify it into one of these categories:
        1. 'CASE': The user is asking about a specific legal case, judgment, or court verdict (e.g., "Kesavananda Bharati case", "facts of X vs Y").
        2. 'LAW': The user is asking about general legal principles, acts, sections, or constitutional articles (e.g., "What is Article 21?", "Rights of arrested person").
        3. 'BOTH': The query requires both general law and specific case precedents (e.g., "Explain Article 21 with landmark judgments").
        
        Query: "{query}"
        
        Return ONLY the category name (LAW, CASE, or BOTH). Do not add any explanation.
        """
        
        try:
            response = self.model.generate_content(prompt)
            classification = response.text.strip().upper()
            if classification not in ['LAW', 'CASE', 'BOTH']:
                return 'LAW' # Default fall back
            return classification
        except Exception as e:
            logger.error(f"Routing Error: {e}")
            return 'LAW' # Default fallback

    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Routes the query to the appropriate agent.
        """
        category = self._classify_query(query)
        logger.info(f"📋 Query Classified as: {category}")
        
        if category == 'CASE' and self.case_agent:
            return self.case_agent.process(query, **kwargs)
        
        elif category == 'LAW' and self.law_agent:
            return self.law_agent.process(query, **kwargs)
            
        elif category == 'BOTH':
            # Complex logic: For now, we prioritize Law Agent but maybe append Case Agent results?
            # Or perhaps we run both and merge?
            # For this MVP, let's stick to Law Agent as it has the Hybrid Search which is robust.
            # Ideally:
            # law_res = self.law_agent.process(query)
            # case_res = self.case_agent.process(query)
            # return merge(law_res, case_res)
            
            # Simple approach: Use Law Agent (it has broad DB access)
            if self.law_agent:
                 return self.law_agent.process(query, **kwargs)
        
        # Fallback
        if self.law_agent:
             return self.law_agent.process(query, **kwargs)
             
        return {"answer": "Agent system not fully initialized.", "sources": []}
