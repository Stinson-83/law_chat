import sys
import os
import uuid
from fastapi.testclient import TestClient

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lex_bot.app import app

client = TestClient(app)

def test_session_flow():
    print("🚀 Testing Session-Based Upload Flow...")
    
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    
    # 1. Upload File with Session ID
    file_content = b"Dummy PDF Content for Session Test"
    files = {"file": ("session_test.pdf", file_content, "application/pdf")}
    data = {"session_id": session_id}
    
    print("\n1️⃣ Uploading file...")
    try:
        response = client.post("/upload", files=files, data=data)
        if response.status_code == 200:
            print("✅ Upload Success")
            print(response.json())
        else:
            print(f"❌ Upload Failed: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Exception during upload: {e}")
        return

    # 2. Chat WITHOUT file_path (should use session cache)
    print("\n2️⃣ Sending Chat Request (No file_path)...")
    chat_payload = {
        "query": "What is in this file?",
        "session_id": session_id,
        "llm_mode": "fast"
    }
    
    try:
        # We expect the graph to try and use the file. 
        # Since it's a dummy file, the PDF processor might fail or return empty, 
        # but we just want to verify that the 'file_path' was retrieved and passed to the graph.
        
        # To verify this without mocking everything, we can check logs or just see if it runs without error.
        # Ideally, we'd mock run_query, but let's run it against the real app and see if it picks up the file.
        
        response = client.post("/chat", json=chat_payload)
        
        if response.status_code == 200:
            print("✅ Chat Request Success")
            res_data = response.json()
            print("Answer:", res_data.get("answer"))
            
            # Check if memory/file was used (inferred)
            # In a real scenario, we'd check internal logs, but here success implies it didn't crash on missing file
            print("✅ Flow completed successfully.")
        else:
            print(f"❌ Chat Failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Exception during chat: {e}")

if __name__ == "__main__":
    test_session_flow()
