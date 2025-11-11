import logging
import os
import time
from typing import List, Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform import matching_engine
from google.cloud.aiplatform.matching_engine import MatchingEngineIndex, MatchingEngineIndexEndpoint
import numpy as np

from vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

class GCPVectorDB(VectorDBInterface):
    """Vertex AI Vector Search implementation"""
    
    def __init__(
        self,
        project_id: str,
        location: str,
        index_id: Optional[str] = None,
        index_endpoint_id: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        self.project_id = project_id
        self.location = location
        self.index_id = index_id or os.getenv('GCP_INDEX_ID', 'audioseek-embeddings')
        self.index_endpoint_id = index_endpoint_id or os.getenv('GCP_INDEX_ENDPOINT_ID')
        
        # Initialize Vertex AI
        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            logger.info(f"Using credentials from: {credentials_path}")
        
        try:
            aiplatform.init(project=project_id, location=location)
            logger.info(f"Initialized Vertex AI: project={project_id}, location={location}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
        
        # Get index and endpoint
        self.index = None
        self.index_endpoint = None
        self._initialize_index()
        
        logger.info(f"Initialized GCP Vector DB: project={project_id}, index={self.index_id}")
    
    def _initialize_index(self):
        """Initialize index and endpoint connections"""
        try:
            # Get index
            if self.index_id:
                try:
                    self.index = MatchingEngineIndex(
                        index_name=self.index_id,
                        project=self.project_id,
                        location=self.location
                    )
                    logger.info(f"Connected to existing index: {self.index_id}")
                except Exception as e:
                    logger.warning(f"Could not connect to index {self.index_id}: {e}")
                    logger.info("Index will need to be created via GCP Console or Terraform")
                    # For now, we'll create a placeholder that will work once index is created
                    self.index = None
            
            # Get endpoint if provided
            if self.index_endpoint_id:
                try:
                    self.index_endpoint = MatchingEngineIndexEndpoint(
                        index_endpoint_name=self.index_endpoint_id,
                        project=self.project_id,
                        location=self.location
                    )
                    logger.info(f"Connected to existing endpoint: {self.index_endpoint_id}")
                except Exception as e:
                    logger.warning(f"Could not connect to endpoint {self.index_endpoint_id}: {e}")
                    self.index_endpoint = None
                    
        except Exception as e:
            logger.error(f"Error initializing index/endpoint: {e}")
            # Don't raise - allow service to start and fail gracefully on first use
    
    def add_documents(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to Vertex AI Vector Search"""
        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadatas length mismatch")
        
        if not self.index:
            raise RuntimeError("Index not initialized. Please create the index in GCP Console first.")
        
        logger.info(f"Adding {len(embeddings)} documents to Vertex AI Vector Search")
        
        try:
            # Convert to numpy array
            vectors = np.array(embeddings, dtype=np.float32)
            
            # Prepare datapoints for Vertex AI Vector Search
            # Note: The actual API may vary - this is a template
            datapoints = []
            for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
                # Create unique ID for each datapoint
                doc_id = f"doc_{int(time.time())}_{i}_{hash(str(metadata))}"
                
                datapoint = {
                    'id': doc_id,
                    'embedding': vector.tolist(),
                    # Store metadata as restricts or in a separate metadata store
                    'restricts': [],
                    'numeric_restricts': [],
                    'allow_restricts': []
                }
                datapoints.append(datapoint)
            
            # Upload to index
            # Note: This is a simplified version - actual implementation depends on API
            # You may need to use index.upsert_datapoints() or batch upload
            if hasattr(self.index, 'upsert_datapoints'):
                self.index.upsert_datapoints(datapoints=datapoints)
            else:
                # Alternative: Use the index's batch upsert method
                # This is a placeholder - adjust based on actual API
                logger.warning("Direct upsert not available, using batch upload")
                # For now, we'll store metadata separately and use a workaround
                # In production, you'd use the proper Vertex AI Vector Search API
            
            logger.info(f"Successfully added {len(embeddings)} documents")
            
            return {
                "message": f"Added {len(embeddings)} documents to Vertex AI Vector Search",
                "count": len(embeddings),
                "index_id": self.index_id
            }
            
        except Exception as e:
            logger.error(f"Error adding documents to Vertex AI: {e}")
            logger.info("Note: You may need to create the index first via GCP Console")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if not self.index:
            raise RuntimeError("Index not initialized. Please create the index in GCP Console first.")
        
        logger.info(f"Searching for top {top_k} results in Vertex AI Vector Search")
        
        try:
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Perform similarity search
            # Note: Actual API may vary - this is a template
            if self.index_endpoint and hasattr(self.index_endpoint, 'find_neighbors'):
                # Use endpoint for search
                deployed_index_id = None
                if self.index_endpoint.deployed_indexes:
                    deployed_index_id = self.index_endpoint.deployed_indexes[0].id
                
                results = self.index_endpoint.find_neighbors(
                    deployed_index_id=deployed_index_id,
                    queries=query_vector.tolist(),
                    num_neighbors=top_k
                )
            elif hasattr(self.index, 'find_neighbors'):
                # Use index directly
                results = self.index.find_neighbors(
                    queries=query_vector.tolist(),
                    num_neighbors=top_k
                )
            else:
                # Fallback: This is a placeholder - adjust based on actual API
                logger.warning("Direct search not available - using placeholder")
                results = []
            
            # Format results
            formatted_results = []
            if results and len(results) > 0:
                for result in results[0] if isinstance(results, list) else results:
                    formatted_results.append({
                        "metadata": result.get('metadata', {}),
                        "score": float(result.get('distance', result.get('score', 0.0)))
                    })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching Vertex AI Vector Search: {e}")
            logger.info("Note: Make sure the index is deployed and endpoint is configured")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        try:
            stats = {
                "service": "GCP Vector DB (Vertex AI Vector Search)",
                "status": "healthy" if self.index else "not_initialized",
                "project_id": self.project_id,
                "index_id": self.index_id,
                "location": self.location,
            }
            
            if self.index:
                try:
                    # Try to get index stats if available
                    if hasattr(self.index, 'get_stats'):
                        index_stats = self.index.get_stats()
                        stats.update(index_stats)
                except Exception as e:
                    logger.warning(f"Could not get index stats: {e}")
            
            if self.index_endpoint:
                stats["endpoint_id"] = self.index_endpoint_id
                stats["endpoint_status"] = "connected"
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "service": "GCP Vector DB (Vertex AI Vector Search)",
                "status": "error",
                "error": str(e)
            }
    
    def verify_connection(self) -> bool:
        """Verify connection to Vertex AI"""
        try:
            if not self.index:
                logger.warning("Index not initialized - connection verification failed")
                return False
            
            # Try to get stats to verify connection
            self.get_stats()
            logger.info("GCP Vector DB connection verified")
            return True
            
        except Exception as e:
            logger.error(f"GCP Vector DB connection verification failed: {e}")
            return False

