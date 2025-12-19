import sys
import os
from unittest.mock import MagicMock

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock PDF Processor BEFORE importing graph
from lex_bot.tools.pdf_processor import pdf_processor
pdf_processor.extract_text = MagicMock(return_value="""
AGREEMENT FOR SALE
This Agreement for Sale is made on this 19th day of December 2025.
BETWEEN
Mr. John Doe (Seller)
AND
Mr. Jane Smith (Buyer)

1. The Seller agrees to sell the property located at 123 Legal Lane, Mumbai for a consideration of Rs. 1 Crore.
2. The Buyer has paid an advance of Rs. 10 Lakhs.
3. The possession shall be handed over on 1st Jan 2026.
4. In case of dispute, courts in Mumbai shall have jurisdiction.
""")

from lex_bot.graph import app
from lex_bot.state import AgentState

def test_pdf_flow():
    print("🚀 Testing PDF Flow...")
    
    query = "What is the consideration amount and possession date?"
    file_path = "/tmp/dummy.pdf"
    
    initial_state = AgentState(
        messages=[],
        original_query=query,
        uploaded_file_path=file_path,
        law_context=[],
        case_context=[],
        errors=[],
        document_context=[]
    )
    
    try:
        print(f"Query: {query}")
        print(f"File: {file_path}")
        
        result = app.invoke(initial_state)
        
        print("\n✅ WORKFLOW SUCCESS")
        print("Final Answer:\n", result.get("final_answer"))
        
        # Verify document agent was used
        tool_results = result.get("tool_results", [])
        doc_agent_used = any(t.get("agent") == "document" for t in tool_results)
        
        if doc_agent_used:
            print("\n✅ Document Agent was triggered.")
        else:
            print("\n❌ Document Agent was NOT triggered.")
            
    except Exception as e:
        print(f"\n❌ WORKFLOW FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_flow()
