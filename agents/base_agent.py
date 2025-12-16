from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """
    Abstract Base Class for all agents in the system.
    """
    
    @abstractmethod
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process the user query and return a structured response.
        
        Args:
            query (str): The user's input query.
            **kwargs: Additional arguments (e.g., filters, context).
            
        Returns:
            Dict[str, Any]: A dictionary containing the answer, sources, and metadata.
        """
        pass
