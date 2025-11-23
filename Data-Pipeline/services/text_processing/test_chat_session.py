import sys
import os
import pytest

# Add current directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
from unittest.mock import MagicMock, patch

# Mock sentence_transformers before importing services to avoid model download
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers"].SentenceTransformer = MagicMock()

# Mock spacy and transformers to avoid loading large models
sys.modules["spacy"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Mock visualization and tracking libs
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["seaborn"] = MagicMock()
sys.modules["mlflow"] = MagicMock()

from services import MetadataDBService, QAService
from models import QueryRequest

# Mock settings
os.environ["GEMINI_API_KEY"] = "fake_key"
os.environ["VECTOR_DB_TYPE"] = "faiss"

@pytest.fixture
def metadata_db():
    db_path = "test_audiobook_metadata.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    service = MetadataDBService(db_path=db_path)
    yield service
    
    if os.path.exists(db_path):
        os.remove(db_path)

def test_session_creation(metadata_db):
    session_id = metadata_db.create_session()
    assert session_id is not None
    assert isinstance(session_id, str)
    
    # Verify in DB
    conn = sqlite3.connect(metadata_db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    assert row is not None
    assert row[0] == session_id

def test_chat_history(metadata_db):
    session_id = metadata_db.create_session()
    
    metadata_db.add_chat_message(session_id, "user", "Hello")
    metadata_db.add_chat_message(session_id, "assistant", "Hi there")
    
    history = metadata_db.get_chat_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there"

@patch("services.get_vector_db")
@patch("services.embedding_model")
@patch("google.generativeai.GenerativeModel")
def test_qa_service_session(mock_genai, mock_embed, mock_get_db, metadata_db):
    # Setup mocks
    mock_llm = MagicMock()
    mock_llm.generate_content.return_value.text = "This is the answer."
    mock_genai.return_value = mock_llm
    
    import numpy as np
    mock_embed.encode.return_value = np.array([[0.1] * 384]) # Mock embedding as numpy array
    
    mock_vector_db = MagicMock()
    mock_vector_db.search.return_value = [
        {"metadata": {"text": "Context text", "start_time": 0, "end_time": 10}, "score": 0.9}
    ]
    mock_get_db.return_value = mock_vector_db
    
    qa_service = QAService(metadata_db)
    qa_service.llm = mock_llm # Inject mock LLM
    
    # 1. First turn (new session)
    req1 = QueryRequest(query="What is this book about?", book_id="test_book")
    resp1 = qa_service.ask_question(req1)
    
    assert resp1.session_id is not None
    session_id = resp1.session_id
    assert resp1.answer == "This is the answer."
    
    # Verify history
    history = metadata_db.get_chat_history(session_id)
    assert len(history) == 2
    assert history[0]["content"] == "What is this book about?"
    
    # 2. Second turn (existing session)
    req2 = QueryRequest(query="Who is the author?", book_id="test_book", session_id=session_id)
    resp2 = qa_service.ask_question(req2)
    
    assert resp2.session_id == session_id
    
    # Verify history grew
    history = metadata_db.get_chat_history(session_id)
    assert len(history) == 4
    assert history[2]["content"] == "Who is the author?"

if __name__ == "__main__":
    # Manually run tests
    print("Running tests manually...")
    
    # Setup fixture
    db_path = "test_audiobook_metadata.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    service = MetadataDBService(db_path=db_path)
    
    try:
        test_session_creation(service)
        print("test_session_creation passed")
        
        test_chat_history(service)
        print("test_chat_history passed")
        
        # Mocking for qa_service test
        mock_genai = MagicMock()
        mock_embed = MagicMock()
        mock_get_db = MagicMock()
        test_qa_service_session(mock_genai, mock_embed, mock_get_db, service)
        print("test_qa_service_session passed")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

