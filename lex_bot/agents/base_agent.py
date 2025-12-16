import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from lex_bot.config import LLM_MODEL_NAME, GOOGLE_API_KEY

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
            system_prompt = """You are a Researcher who specializes in legal case law.
            You want to research about specific case(s) and find out the relevant information about it
            so you have to search on the internet about it, so you want to structure your queries in way so that you get most .
            Input: An unstructured user query about Indian legal Case.
            Output: A single line of 5-8 relevant search keywords.
            Instructions about the output:
            - The query should roughly be structured this way: "Case Name + Court Name + Date of Judgment"
            - Case Name, Court Name, Date of judgement should be extracted from the user query only, if it can't be infered from the user query dont add it in the query.
            - Court Name can be among: Supreme Court or High court with state name or District Court
            - Focus on landmark case names or specific legal doctrines.
            - NO explanations, NO lists, NO markdown. Just keywords separated by spaces."""
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
