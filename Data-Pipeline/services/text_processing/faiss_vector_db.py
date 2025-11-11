import logging
import os
import json
from typing import List, Dict, Any
import faiss
import numpy as np
from google.cloud import storage

from vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

class FAISSVectorDB(VectorDBInterface):
    """FAISS-based vector database with GCS persistence for production"""
    
    def __init__(
        self, 
        bucket_name: str = "audioseek-bucket",
        storage_path: str = "vector-db",
        dimension: int = 384,
        project_id: str = None
    ):
        self.bucket_name = bucket_name
        self.storage_path = storage_path
        self.dimension = dimension
        self.local_cache = "/tmp/faiss_cache"
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID', 'ie7374-475102')
        
        # Create local cache directory
        os.makedirs(self.local_cache, exist_ok=True)
        
        # Initialize GCS client with explicit project
        try:
            self.storage_client = storage.Client(project=self.project_id)
            self.bucket = self.storage_client.bucket(bucket_name)
            logger.info(f"Connected to GCS: project={self.project_id}, bucket={bucket_name}")
        except Exception as e:
            logger.error(f"Failed to connect to GCS: {e}")
            raise
        
        # Initialize FAISS index
        self.vector_index = faiss.IndexFlatIP(dimension)
        self.vector_metadata = []
        
        # Load existing index from GCS if available
        self._load_from_gcs()
        
        logger.info(f"Initialized FAISS Vector DB with GCS persistence")
    
    def _load_from_gcs(self):
        """Download FAISS index and metadata from GCS"""
        try:
            index_blob_path = f"{self.storage_path}/faiss_index.bin"
            metadata_blob_path = f"{self.storage_path}/faiss_metadata.json"
            
            index_blob = self.bucket.blob(index_blob_path)
            metadata_blob = self.bucket.blob(metadata_blob_path)
            
            # Check if index exists in GCS
            if index_blob.exists() and metadata_blob.exists():
                logger.info(f"Downloading index from gs://{self.bucket_name}/{index_blob_path}")
                
                # Download index file
                local_index_path = os.path.join(self.local_cache, "faiss_index.bin")
                index_blob.download_to_filename(local_index_path)
                self.vector_index = faiss.read_index(local_index_path)
                
                # Download metadata file
                local_metadata_path = os.path.join(self.local_cache, "faiss_metadata.json")
                metadata_blob.download_to_filename(local_metadata_path)
                
                with open(local_metadata_path, 'r', encoding='utf-8') as f:
                    self.vector_metadata = json.load(f)
                
                logger.info(f"✓ Loaded {len(self.vector_metadata)} documents from GCS")
            else:
                logger.info("No existing index found in GCS, starting fresh")
                
        except Exception as e:
            logger.warning(f"Error loading index from GCS: {e}, starting fresh")
            self.vector_index = faiss.IndexFlatIP(self.dimension)
            self.vector_metadata = []
    
    def _save_to_gcs(self):
        """Upload FAISS index and metadata to GCS"""
        try:
            # Save to local cache first
            local_index_path = os.path.join(self.local_cache, "faiss_index.bin")
            local_metadata_path = os.path.join(self.local_cache, "faiss_metadata.json")
            
            logger.info(f"Saving index with {len(self.vector_metadata)} documents")
            
            # Write index to local file
            faiss.write_index(self.vector_index, local_index_path)
            
            # Write metadata to local file
            with open(local_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.vector_metadata, f, indent=2)
            
            # Upload to GCS
            index_blob_path = f"{self.storage_path}/faiss_index.bin"
            metadata_blob_path = f"{self.storage_path}/faiss_metadata.json"
            
            logger.info(f"Uploading to gs://{self.bucket_name}/{index_blob_path}")
            
            index_blob = self.bucket.blob(index_blob_path)
            index_blob.upload_from_filename(local_index_path)
            
            metadata_blob = self.bucket.blob(metadata_blob_path)
            metadata_blob.upload_from_filename(local_metadata_path)
            
            logger.info(f"✓ Saved {len(self.vector_metadata)} documents to GCS")
            
        except Exception as e:
            logger.error(f"Error saving index to GCS: {e}")
    
    def add_documents(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to FAISS index and save to GCS"""
        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadatas length mismatch")
        
        logger.info(f"Adding {len(embeddings)} documents to FAISS")
        
        vectors = np.array(embeddings, dtype=np.float32)
        self.vector_index.add(vectors)
        self.vector_metadata.extend(metadatas)
        
        # Auto-save to GCS after adding
        self._save_to_gcs()
        
        logger.info(f"Documents added successfully. Total: {len(self.vector_metadata)}")
        
        return {
            "message": f"Added {len(embeddings)} documents to FAISS (stored in GCS)",
            "count": len(embeddings),
            "total_documents": len(self.vector_metadata)
        }
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        logger.info(f"Searching for top {top_k} results in FAISS")
        
        if len(self.vector_metadata) == 0:
            logger.warning("Vector DB is empty")
            return []
        
        query = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, len(self.vector_metadata))
        distances, indices = self.vector_index.search(query, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.vector_metadata):
                results.append({
                    "metadata": self.vector_metadata[idx],
                    "score": float(distances[0][i])
                })
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        return {
            "service": "FAISS Vector DB with GCS Persistence",
            "status": "healthy",
            "documents_count": len(self.vector_metadata),
            "index_total": self.vector_index.ntotal,
            "dimension": self.dimension,
            "storage": {
                "type": "Google Cloud Storage",
                "bucket": self.bucket_name,
                "path": self.storage_path
            }
        }
    
    def verify_connection(self) -> bool:
        """Verify connections"""
        try:
            _ = self.vector_index.ntotal
            self.bucket.exists()
            logger.info("✓ FAISS and GCS verified")
            return True
        except Exception as e:
            logger.error(f"✗ Verification failed: {e}")
            return False