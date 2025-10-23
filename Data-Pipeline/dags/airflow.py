#!/usr/bin/env python3
"""
AudioSEEK Main DAG - FIXED VERSION
Pipeline: Validate Model → Transcription → Cross-Model Eval → Chunking → Embedding

File: dags/__init__.py or dags/airflow.py
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ===========================
# Path Configuration - FIXED
# ===========================

# Get project root (DATA-PIPELINE folder)
DAG_FOLDER = Path(__file__).parent  # dags/
PROJECT_ROOT = DAG_FOLDER.parent    # DATA-PIPELINE/

# Define data paths
DATA_DIR = PROJECT_ROOT / "data"

# ===========================
# Task 1: Validate Model - FIXED
# ===========================

def task_validate(**context):
    """
    Validate that all required models are available
    """
    logger.info("=" * 60)
    logger.info("TASK 1: VALIDATE MODEL")
    logger.info("=" * 60)
    
    # FIXED: Import directly here, not at module level
    try:
        # Add validation path
        validation_path = PROJECT_ROOT / "scripts" / "validation" / "model_validation"
        sys.path.insert(0, str(validation_path))
        
        logger.info(f"Looking for validate_model in: {validation_path}")
        
        # Check if file exists
        validate_file = validation_path / "validate_model.py"
        if not validate_file.exists():
            logger.error(f"File not found: {validate_file}")
            logger.info("Creating a basic validation function...")
            
            # If file doesn't exist, create inline validation
            from faster_whisper import WhisperModel
            
            logger.info("Testing Faster-Whisper import...")
            logger.info("✓ Faster-Whisper imported successfully")
            
            logger.info("Loading base model...")
            model = WhisperModel("base", device="cpu", compute_type="float32")
            logger.info("✓ Model loaded successfully")
            
            del model  # Free memory
            
            return {
                "status": "passed",
                "model": "base",
                "message": "Model validation successful (inline)"
            }
        
        else:
            # File exists, try to import it
            logger.info(f"✓ Found validate_model.py")
            from validate_model import validate_whisper_model
            
            logger.info("Running model validation...")
            result = validate_whisper_model()
            
            logger.info("✓ Model validation passed")
            return result
        
    except Exception as e:
        logger.error(f"✗ Model validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ===========================
# Task 2: Transcription - FIXED
# ===========================

def task_transcribe(**context):
    """
    Transcribe audio files
    """
    logger.info("=" * 60)
    logger.info("TASK 2: TRANSCRIPTION")
    logger.info("=" * 60)
    
    try:
        # Add transcription path
        transcription_path = PROJECT_ROOT / "scripts" / "transcription"
        sys.path.insert(0, str(transcription_path))
        
        logger.info(f"Looking for transcription module in: {transcription_path}")
        
        # Get DAG run configuration
        dag_run = context['dag_run']
        audio_file = dag_run.conf.get('audio_file', 'sample.mp3')
        
        logger.info(f"Audio file from config: {audio_file}")
        
        # Check if transcription.py has the function
        transcription_file = transcription_path / "transcription.py"
        if not transcription_file.exists():
            logger.error(f"Transcription script not found: {transcription_file}")
            raise FileNotFoundError(f"Missing: {transcription_file}")
        
        logger.info(f"✓ Found transcription.py")
        
        # For now, return mock data until you implement transcription function
        logger.warning("⚠️ Using mock transcription data")
        logger.info("TODO: Implement actual transcription logic")
        
        return {
            "status": "success",
            "transcript_file": str(DATA_DIR / "transcription_results" / "sample_transcript.txt"),
            "message": "Mock transcription (replace with actual implementation)"
        }
        
    except Exception as e:
        logger.error(f"✗ Transcription failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ===========================
# Task 3: Cross-Model Evaluation - FIXED
# ===========================

def task_eval(**context):
    """
    Evaluate transcription quality
    """
    logger.info("=" * 60)
    logger.info("TASK 3: CROSS-MODEL EVALUATION")
    logger.info("=" * 60)
    
    try:
        # Add evaluation path
        eval_path = PROJECT_ROOT / "scripts" / "validation" / "cross_model_evaluation"
        sys.path.insert(0, str(eval_path))
        
        logger.info(f"Looking for evaluation module in: {eval_path}")
        
        # Get transcription results
        ti = context['ti']
        transcribe_data = ti.xcom_pull(task_ids='task_transcribe')
        
        logger.info(f"Transcription data: {transcribe_data}")
        
        # For now, return mock data
        logger.warning("⚠️ Using mock evaluation data")
        logger.info("TODO: Implement actual evaluation logic")
        
        return {
            "status": "success",
            "evaluation": "passed",
            "message": "Mock evaluation (replace with actual implementation)"
        }
        
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ===========================
# Task 4: Chunking - FIXED
# ===========================

def task_chunk(**context):
    """
    Chunk transcripts
    """
    logger.info("=" * 60)
    logger.info("TASK 4: CHUNKING")
    logger.info("=" * 60)
    
    try:
        # Add chunking path
        chunking_path = PROJECT_ROOT / "scripts" / "chunking"
        sys.path.insert(0, str(chunking_path))
        
        logger.info(f"Looking for chunking module in: {chunking_path}")
        
        # Get transcription results
        ti = context['ti']
        transcribe_data = ti.xcom_pull(task_ids='task_transcribe')
        
        logger.info(f"Transcription data: {transcribe_data}")
        
        # For now, return mock data
        logger.warning("⚠️ Using mock chunking data")
        logger.info("TODO: Implement actual chunking logic")
        
        return {
            "status": "success",
            "chunks_file": str(DATA_DIR / "chunking_results" / "sample_chunks.json"),
            "message": "Mock chunking (replace with actual implementation)"
        }
        
    except Exception as e:
        logger.error(f"✗ Chunking failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ===========================
# Task 5: Embedding - FIXED
# ===========================

def task_embed(**context):
    """
    Generate embeddings
    """
    logger.info("=" * 60)
    logger.info("TASK 5: EMBEDDING")
    logger.info("=" * 60)
    
    try:
        # Add embedding path
        embedding_path = PROJECT_ROOT / "scripts" / "embedding"
        sys.path.insert(0, str(embedding_path))
        
        logger.info(f"Looking for embedding module in: {embedding_path}")
        
        # Get chunking results
        ti = context['ti']
        chunk_data = ti.xcom_pull(task_ids='task_chunk')
        
        logger.info(f"Chunking data: {chunk_data}")
        
        # For now, return mock data
        logger.warning("⚠️ Using mock embedding data")
        logger.info("TODO: Implement actual embedding logic")
        
        logger.info("=" * 60)
        logger.info("✓ PIPELINE COMPLETE!")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "embeddings_file": "sample_embeddings.npy",
            "message": "Mock embedding (replace with actual implementation)"
        }
        
    except Exception as e:
        logger.error(f"✗ Embedding failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# ===========================
# DAG Definition
# ===========================

default_args = {
    'owner': 'audioseek',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,  # Reduced to 1 for faster debugging
    'retry_delay': timedelta(minutes=2),  # Reduced to 2 minutes
}

# Create the DAG
dag = DAG(
    dag_id='audioseek_main_pipeline',
    default_args=default_args,
    description='AudioSEEK: Validate → Transcribe → Evaluate → Chunk → Embed (Fixed)',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['audioseek', 'mlops', 'fixed'],
)

# ===========================
# Define Tasks
# ===========================

validate = PythonOperator(
    task_id='task_validate',
    python_callable=task_validate,
    provide_context=True,
    dag=dag,
)

transcribe = PythonOperator(
    task_id='task_transcribe',
    python_callable=task_transcribe,
    provide_context=True,
    dag=dag,
)

evaluate = PythonOperator(
    task_id='task_eval',
    python_callable=task_eval,
    provide_context=True,
    dag=dag,
)

chunk = PythonOperator(
    task_id='task_chunk',
    python_callable=task_chunk,
    provide_context=True,
    dag=dag,
)

embed = PythonOperator(
    task_id='task_embed',
    python_callable=task_embed,
    provide_context=True,
    dag=dag,
)

# ===========================
# Define Pipeline Flow
# ===========================

validate >> transcribe >> evaluate >> chunk >> embed