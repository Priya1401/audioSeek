import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
  """Application settings loaded from environment variables"""

  # GCP Configuration
  gcp_project_id: str = os.getenv('GCP_PROJECT_ID', '')
  gcp_region: str = os.getenv('GCP_REGION', 'us-east1')
  gcp_index_id: str = os.getenv('GCP_INDEX_ID', 'audioseek-embeddings')
  gcp_index_endpoint_id: Optional[str] = os.getenv('GCP_INDEX_ENDPOINT_ID')
  gcp_credentials_path: str = os.getenv('GCP_CREDENTIALS_PATH',
                                        '/app/gcp-credentials.json')

  # Vector DB Type
  vector_db_type: str = os.getenv('VECTOR_DB_TYPE', 'local')

  # Gemini
  gemini_api_key: str = os.getenv('GEMINI_API_KEY', '')

  class Config:
    env_file = '.env'
    case_sensitive = False


settings = Settings()