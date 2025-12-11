import json
import logging
import os
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from google.cloud import storage

from services.storage.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


class FAISSVectorDB(VectorDBInterface):
    """
    Local FAISS vector store, with optional Google Cloud Storage sync.
    Book-aware: each book_id gets its own FAISS index and metadata file.
    """

    def __init__(self, book_id: str = "default",
        bucket_name: Optional[str] = None,
        project_id: Optional[str] = None):

        self.book_id = book_id
        self.bucket_name = bucket_name
        self.project_id = project_id

        # Local storage paths
        self.storage_folder = f"faiss_store/{self.book_id}"
        os.makedirs(self.storage_folder, exist_ok=True)

        self.index_file = os.path.join(self.storage_folder, "index.faiss")
        self.metadata_file = os.path.join(self.storage_folder, "metadata.json")

        # Load or create index
        self.index = None
        self.metadatas: List[Dict[str, Any]] = []

        self._load_local()

        # Sync from GCS (if enabled)  Completely skipped it
        if self.bucket_name:
           self._download_from_gcs()

    # --------------------------------------------------------
    # PRIVATE HELPERS
    # --------------------------------------------------------
    def _load_local(self):
        """Load FAISS index + metadata from local disk if present."""
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Loaded FAISS index for book_id={self.book_id}")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")

        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadatas = json.load(f)
                logger.info(f"Loaded metadata for book_id={self.book_id}")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")

    def _save_local(self):
        """Save FAISS index + metadata locally."""
        if self.index:
            faiss.write_index(self.index, self.index_file)

        with open(self.metadata_file, "w") as f:
            json.dump(self.metadatas, f, indent=2)

        logger.info(
            f"Saved FAISS & metadata locally for book_id={self.book_id}")

    # --------------------------------------------------------
    # GCS SYNC
    # --------------------------------------------------------

    def _download_from_gcs(self):
        """Download existing FAISS index + metadata from GCS if present."""
        try:
            client = storage.Client(project=self.project_id)
            bucket = client.bucket(self.bucket_name)

            gcs_index = f"vector-db/{self.book_id}/index.faiss"
            gcs_meta = f"vector-db/{self.book_id}/metadata.json"

            # Download index
            if bucket.blob(gcs_index).exists():
                bucket.blob(gcs_index).download_to_filename(self.index_file)
                self.index = faiss.read_index(self.index_file)

            # Download metadata
            if bucket.blob(gcs_meta).exists():
                bucket.blob(gcs_meta).download_to_filename(self.metadata_file)
                with open(self.metadata_file, "r") as f:
                    self.metadatas = json.load(f)

            logger.info(f"GCS sync completed for book_id={self.book_id}")

        except Exception as e:
            logger.warning(
                f"No GCS index found for book_id={self.book_id}: {e}")

    def _upload_to_gcs(self):
        """Upload FAISS index + metadata to GCS."""
        if not self.bucket_name:
            return

        try:
            client = storage.Client(project=self.project_id)
            bucket = client.bucket(self.bucket_name)

            gcs_index = f"vector-db/{self.book_id}/index.faiss"
            gcs_meta = f"vector-db/{self.book_id}/metadata.json"

            bucket.blob(gcs_index).upload_from_filename(self.index_file)
            bucket.blob(gcs_meta).upload_from_filename(self.metadata_file)

            logger.info(
                f"Uploaded FAISS & metadata to GCS for book_id={self.book_id}")

        except Exception as e:
            logger.error(f"Failed to upload FAISS to GCS: {e}")

    # --------------------------------------------------------
    # PUBLIC METHODS
    # --------------------------------------------------------
    def verify_connection(self):
        """
        Local FAISS always 'works', so return True.
        Exists only to satisfy VectorDBInterface requirements.
        """
        return True

    def add_documents(self, embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:

        vectors = np.array(embeddings).astype("float32")
        new_dim = vectors.shape[1]

        # Check if we need to create or recreate the index
        if self.index is None:
            logger.info(f"Creating new FAISS index with dimension {new_dim}")
            self.index = faiss.IndexFlatL2(new_dim)
        elif self.index.d != new_dim:
            logger.warning(
                f"Dimension mismatch: existing index has dim={self.index.d}, "
                f"new embeddings have dim={new_dim}. Recreating index."
            )
            self.index = faiss.IndexFlatL2(new_dim)

        self.index.add(vectors)

        # Replace metadata
        self.metadatas = metadatas

        # Save locally + GCS
        self._save_local()
        self._upload_to_gcs()

        return {
            "message": f"Added {len(embeddings)} vectors to FAISS for book_id={self.book_id}",
            "count": len(embeddings)
        }

    def search(self, query_vector: List[float], top_k: int = 5, max_chapter: int = None, seconds_listened: float = None):
        if self.index is None:
            return []

        q = np.array([query_vector]).astype("float32")
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue

            meta = self.metadatas[idx]
            score = float(distances[0][i])

            results.append({
                "metadata": meta,
                "score": score
            })

        # SPOILER-FREE: FILTER BY CHAPTER

        if seconds_listened is not None and max_chapter is not None:
            logger.info(f"Filtering until chapter {max_chapter}, time {seconds_listened}")

            filtered = []
            for r in results:
                chapter = r["metadata"].get("chapter_id") or r["metadata"].get("chapter_number") or 0
                start = r["metadata"].get("start_time", 0)

                # Case 1: Earlier chapters → always allowed
                if chapter < max_chapter:
                    filtered.append(r)
                    continue

                # Case 2: Same chapter as user → allow only earlier parts
                if chapter == max_chapter and start <= seconds_listened:
                    filtered.append(r)
                    continue

                # Case 3: Future chapters or future time → block
                # do nothing

            results = filtered


        elif max_chapter is not None:
            logger.info(f"Here , Filtering only until {max_chapter}")
            results = [
                r for r in results
                if (r["metadata"].get("chapter_id") or r["metadata"].get("chapter_number") or 0) <= max_chapter
            ]

        elif seconds_listened is not None:
            logger.info(f"Filtering only until time {seconds_listened}")
            results = [
                r for r in results
                if r["metadata"].get("start_time", 0) <= seconds_listened
            ]

        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "book_id": self.book_id,
            "vector_count": self.index.ntotal if self.index else 0,
            "metadata_count": len(self.metadatas),
            "local_index_path": self.index_file,
            "local_metadata_path": self.metadata_file,
            "gcs_bucket": self.bucket_name,
            "chapter_count": len({m.get("chapter_id") for m in self.metadatas if m.get("chapter_id") is not None})
        }

    def get_by_chapter(self, chapter_id: int) -> List[Dict[str, Any]]:
        """Retrieve all chunks belonging to a specific chapter_id."""
        return [
            m for m in self.metadatas
            if m.get("chapter_id") == chapter_id
        ]
