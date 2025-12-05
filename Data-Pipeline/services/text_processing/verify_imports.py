import sys
import os

# Add the current directory to sys.path to simulate running from root
sys.path.append(os.getcwd())

print("Verifying imports...")

try:
    from core.model_loader import get_embedding_model
    print("model_loader imported successfully")
except ImportError as e:
    print(f"Failed to import model_loader: {e}")
    sys.exit(1)

try:
    from services.storage.vector_db_service import get_vector_db
    print("get_vector_db imported successfully")
except ImportError as e:
    print(f"Failed to import get_vector_db: {e}")
    sys.exit(1)

try:
    from services.nlp.qa_service import QAService
    print("QAService imported successfully")
except ImportError as e:
    print(f"Failed to import QAService: {e}")
    sys.exit(1)

# Check ChunkingService FIRST (User's error source)
try:
    from services.nlp.chunking_service import ChunkingService
    print("ChunkingService imported successfully")
except ImportError as e:
    print(f"Failed to import ChunkingService: {e}")
    sys.exit(1)

try:
    from api.main import app
    print("api.main imported successfully")
except ImportError as e:
    print(f"Failed to import api.main: {e}")
    sys.exit(1)

try:
    from services.storage.vector_db_service import VectorDBService
    print("VectorDBService imported successfully")
except ImportError as e:
    print(f"Failed to import VectorDBService: {e}")
    sys.exit(1)

print("All critical imports verified.")
