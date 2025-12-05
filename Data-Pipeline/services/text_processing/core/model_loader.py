import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer model...")
        _embedding_model = SentenceTransformer(
            "BAAI/bge-m3",
            trust_remote_code=True
        )
        logger.info("SentenceTransformer model loaded.")
    return _embedding_model
