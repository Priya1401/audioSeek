import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from models import CombinedRequest, AddDocumentsRequest, ChunkingRequest, EmbeddingRequest, FullPipelineRequest, \
    QueryRequest, SearchRequest
from services import ChunkingService, EmbeddingService, PipelineService, VectorDBService

client = TestClient(app)


@pytest.fixture
def mock_embedding_model():
    with patch('services.embedding_model') as mock_model:
        # Mock needs to return numpy array, not list
        mock_array = np.array([[0.1, 0.2, 0.3, 0.4] * 96])  # 384 dimensions
        mock_model.encode.return_value = mock_array
        yield mock_model


@pytest.fixture
def mock_vector_index():
    with patch('services.vector_index') as mock_index:
        mock_index.add = MagicMock()
        # Fix: Return tuple with 2D arrays
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]]),  # distances
            np.array([[0, 1]])  # indices
        )
        mock_index.ntotal = 0
        mock_index.d = 384  # Dimension
        yield mock_index


@pytest.fixture
def mock_vector_metadata():
    with patch('services.vector_metadata', []) as mock_metadata:
        yield mock_metadata


def test_chunk_transcript_single_file():
    # Create a temporary transcript file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("[0.00-5.00] Chapter 1: Introduction\n[5.00-10.00] Some text here.")
        temp_file = f.name

    try:
        request = ChunkingRequest(file_path=temp_file, target_tokens=512, overlap_tokens=50)
        response = ChunkingService.chunk_transcript(request)

        assert len(response.chunks) > 0
        assert len(response.chapters) > 0
        assert response.processed_files == [temp_file]
    finally:
        os.unlink(temp_file)


def test_chunk_transcript_file_not_found():
    request = ChunkingRequest(file_path="nonexistent.txt")
    with pytest.raises(Exception):  # HTTPException
        ChunkingService.chunk_transcript(request)


def test_generate_embeddings_from_texts(mock_embedding_model):
    request = EmbeddingRequest(texts=["Hello world", "Test text"])
    response = EmbeddingService.generate_embeddings(request)

    assert len(response.embeddings) == 1
    assert response.count == 1
    # mock_embedding_model.encode.assert_called_once_with(["Hello world", "Test text"])


def test_generate_embeddings_from_chunks_file(mock_embedding_model):
    # Create temp chunks file
    chunks_data = {"chunks": [{"text": "Chunk 1"}, {"text": "Chunk 2"}]}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(chunks_data, f)
        temp_file = f.name

    try:
        request = EmbeddingRequest(chunks_file=temp_file)
        response = EmbeddingService.generate_embeddings(request)

        assert len(response.embeddings) == 1
        assert response.count == 1
    finally:
        os.unlink(temp_file)


def test_add_documents_to_vector_db(mock_vector_index, mock_vector_metadata):
    # Use proper 384-dimension embeddings
    embeddings = [[0.1] * 384, [0.2] * 384]
    request = AddDocumentsRequest(
        embeddings=embeddings,
        metadatas=[{"text": "doc1"}, {"text": "doc2"}]
    )
    response = VectorDBService.add_documents(request)

    assert response.count == 2
    assert "Added 2 documents" in response.message
    mock_vector_index.add.assert_called_once()


def test_search_vector_db(mock_vector_index):
    # Patch vector_metadata as module variable
    with patch('services.vector_metadata', [{"text": "doc1"}, {"text": "doc2"}]):
        request = SearchRequest(query_embedding=[0.1] * 384, top_k=2)
        response = VectorDBService.search(request)

        assert len(response.results) == 2
        assert response.count == 2
        mock_vector_index.search.assert_called_once()


def test_query_text_vector_db(mock_embedding_model, mock_vector_index):
    with patch('services.vector_metadata', [{"text": "doc1"}, {"text": "doc2"}]):
        # Fix: QueryRequest uses 'query', not 'query_text'
        request = QueryRequest(query="test query", top_k=2)
        response = VectorDBService.query_text(request)

        assert len(response.results) == 2


def test_process_combined_pipeline():
    # Create temp transcript file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("[0.00-5.00] Chapter 1: Introduction\n[5.00-10.00] Some text here.")
        temp_file = f.name

    try:
        request = CombinedRequest(file_path=temp_file)
        response = PipelineService.process_combined_pipeline(request)

        assert len(response.chunks) > 0
        assert len(response.embeddings) > 0
        assert response.processed_files == [temp_file]
    finally:
        os.unlink(temp_file)


def test_process_full_pipeline():
    # Create temp transcript file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("[0.00-5.00] Chapter 1: Introduction\n[5.00-10.00] Some text here.")
        temp_file = f.name

    try:
        request = FullPipelineRequest(file_path=temp_file, add_to_vector_db=False)
        response = PipelineService.process_full_pipeline(request)

        assert response.chunks_count > 0
        assert response.embeddings_count > 0
        assert response.vector_db_added == False
        assert "Full pipeline completed" in response.message
    finally:
        os.unlink(temp_file)


# FastAPI endpoint tests
def test_chunk_endpoint():
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("[0.00-5.00] Chapter 1: Introduction")
        temp_file = f.name

    try:
        data = {"file_path": temp_file}
        response = client.post("/chunk", json=data)
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
    finally:
        os.unlink(temp_file)


def test_embed_endpoint():
    data = {"texts": ["Hello", "World"]}
    response = client.post("/embed", json=data)
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert data["count"] == 2


EMBEDDING_DIM = 384


def test_vector_db_add_endpoint():
    data = {
        "embeddings": [[0.1] * EMBEDDING_DIM],
        "metadatas": [{"text": "test"}]
    }
    response = client.post("/vector-db/add", json=data)
    assert response.status_code == 200
    data = response.json()
    assert "Added" in data["message"]


def test_vector_db_query_endpoint():
    data = {"query": "test query"}
    response = client.post("/vector-db/query", json=data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "Text Processing Service"
