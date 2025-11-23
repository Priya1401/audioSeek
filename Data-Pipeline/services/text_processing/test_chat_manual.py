import requests
import json
import sys

BASE_URL = "http://localhost:8001"

def test_chat_session():
    print("="*50)
    print("Testing Chat Session Management")
    print("="*50)

    # 1. Start a new session
    print("\n1. Starting new session...")
    query1 = {
        "query": "What is the main theme of the book?",
        "book_id": "default",  # Ensure you have a book processed or use 'default'
        "top_k": 3
    }
    
    try:
        response1 = requests.post(f"{BASE_URL}/qa/ask", json=query1)
        response1.raise_for_status()
        data1 = response1.json()
        
        session_id = data1.get("session_id")
        answer1 = data1.get("answer")
        
        print(f"✓ Session Created: {session_id}")
        print(f"✓ Answer: {answer1[:100]}...")
        
        if not session_id:
            print("❌ Error: No session_id returned!")
            return

        # 2. Follow-up question (using session_id)
        print("\n2. Asking follow-up question...")
        query2 = {
            "query": "Tell me more about that.",
            "book_id": "default",
            "session_id": session_id,
            "top_k": 3
        }
        
        response2 = requests.post(f"{BASE_URL}/qa/ask", json=query2)
        response2.raise_for_status()
        data2 = response2.json()
        
        answer2 = data2.get("answer")
        session_id2 = data2.get("session_id")
        
        print(f"✓ Answer: {answer2[:100]}...")
        
        if session_id == session_id2:
            print(f"✓ Session ID maintained: {session_id2}")
        else:
            print(f"❌ Error: Session ID changed! {session_id} -> {session_id2}")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the service.")
        print("Make sure the service is running: python services/text_processing/main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_chat_session()
