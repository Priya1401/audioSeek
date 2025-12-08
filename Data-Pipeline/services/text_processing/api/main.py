import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from core.config import settings
from api.controllers import router
from services.storage.vector_db_service import get_vector_db

# Configure logging with explicit stdout handler and force override
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # Override any existing configuration
)

# Configure specific module loggers
for module in ['services', 'utils', 'controllers', 'faiss_vector_db',
               'gcp_vector_db', '__main__']:
    module_logger = logging.getLogger(module)
    module_logger.setLevel(logging.INFO)
    module_logger.propagate = True

# Reduce noise from uvicorn
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Processing Service",
    description="Chunk transcripts, generate embeddings, and manage vector database",
    version="2.0.0"
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Verify connections on startup"""
    logger.info("=" * 70)
    logger.info("Starting Text Processing Service...")
    logger.info(f"Vector DB Type: {settings.vector_db_type}")

    if settings.vector_db_type.lower() == 'gcp':
        logger.info(f"GCP Project: {settings.gcp_project_id}")
        logger.info(f"GCP Region: {settings.gcp_region}")
        logger.info(f"Index ID: {settings.gcp_index_id}")

    try:
        # Initialize and verify vector DB connection
        vector_db = get_vector_db()
        if vector_db.verify_connection():
            logger.info("✓ Vector DB connection verified")
        else:
            logger.warning("⚠ Vector DB connection verification failed")
            
        # Sync metadata from GCS if configured
        if settings.gcp_bucket_name and settings.gcp_project_id:
            logger.info("Syncing metadata from GCS...")
            from services.storage.metadata_db_service import metadata_db
            metadata_db.sync_from_gcs(settings.gcp_project_id, settings.gcp_bucket_name)

        # Check for stale jobs (interrupted by previous crashes)
        from services.jobs.job_service import job_service
        job_service.check_stale_jobs()
            
    except Exception as e:
        logger.error(f"✗ Failed to initialize: {e}")
        logger.warning("Service will start but operations may fail")

    logger.info("=" * 70)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        vector_db = get_vector_db()
        stats = vector_db.get_stats()
        return JSONResponse({
            "status": "healthy",
            "vector_db": stats.get("status", "unknown"),
            "vector_db_type": settings.vector_db_type
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e)
        }, status_code=503)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True
    )
