import sys
import os
from fastapi.testclient import TestClient

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lex_bot.app import app

client = TestClient(app)

def test_upload():
    print("🚀 Testing Upload Endpoint...")
    
    # Create a dummy file
    file_content = b"Dummy PDF Content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    
    try:
        response = client.post("/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload Success")
            print(f"File Path: {data['file_path']}")
            
            # Now verify checking existence
            if os.path.exists(data['file_path']):
                print("✅ File exists on disk")
            else:
                print("❌ File not found on disk")
                
            # Clean up
            os.remove(data['file_path'])
            
        else:
            print(f"❌ Upload Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_upload()
