import os
import sys
from dotenv import load_dotenv

# Ensure we can import from current directory
sys.path.append(os.getcwd())

load_dotenv()

from agents.manager import ManagerAgent
from agents.law_agent import LawAgent
from agents.case_agent import CaseAgent

def test_routing():
    print("🚀 Initializing Agents...")
    try:
        law = LawAgent()
        case = CaseAgent()
        manager = ManagerAgent(law_agent=law, case_agent=case)
        print("✅ Agents Initialized.")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    queries = [
        ("What is Article 21 of Indian Constitution?", "LAW"),
        ("Kesavananda Bharati vs State of Kerala case details", "CASE"),
        ("Explain the Basic Structure Doctrine", "LAW") # Could be BOTH, but likely Law or Case.
    ]
    
    for q, expected in queries:
        print(f"\n🧪 Testing Query: '{q}'")
        # specific access to private method for testing routing logic
        category = manager._classify_query(q)
        print(f"   -> Classified as: {category} (Expected: {expected} or similar)")
        
        # Run process
        try:
            result = manager.process(q)
            answer = result.get('answer', '')
            source_type = result.get('source_type', 'unknown')
            print(f"   -> Source Type: {source_type}")
            print(f"   -> Answer Preview: {answer[:100]}...")
        except Exception as e:
            print(f"   -> Processing Failed: {e}")

if __name__ == "__main__":
    test_routing()
