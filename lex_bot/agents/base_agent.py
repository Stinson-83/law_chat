import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from ..config import LLM_MODEL_NAME, GOOGLE_API_KEY

class BaseAgent:
    def __init__(self):
        self.llm = self._init_llm()

    def _init_llm(self):
        """
        Initialize the LLM. Defaults to Gemini if key exists, else OpenAI.
        """
        if GOOGLE_API_KEY:
            return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0)
        else:
            # Fallback or User preference
            return ChatOpenAI(temperature=0)

    def enhance_query(self, query: str, agent_type: str) -> str:
        """
        Enhance the user query for better search results.
        """
        system_prompt = ""
        if agent_type == "law":
            system_prompt = """You are a legal search optimizer.
            Input: A user query about Indian Law.
            Output: A single line of 5-8 relevant search keywords.
            - Include the specific act or section if relevant (e.g. "Article 21").
            - NO explanations, NO lists, NO markdown. Just keywords separated by spaces."""
        elif agent_type == "case":
            system_prompt = """You are a case law search optimizer.
            Input: A user query about Indian Case Law.
            Output: A single line of 5-8 relevant search keywords.
            - Focus on landmark case names or specific legal doctrines.
            - NO explanations, NO lists, NO markdown. Just keywords separated by spaces."""
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
