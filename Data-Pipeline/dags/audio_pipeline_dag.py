from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# # Add the project root to Python path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# Import your script main() entrypoints
from scripts.validation.model_validation.validate_model import main as validate_model_main
from scripts.transcription.transcription import main as transcribe_audio_main
from scripts.validation.cross_model_evaluation.cross_model_evaluation import main as cross_model_eval_main
from scripts.chunking.chunking import main as chunking_main
from scripts.embedding.embedding import main as embedding_main

# Default arguments for the DAG
default_args = {
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 24),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'audio_processing_pipeline',
    default_args=default_args,
    description='DAG for audio processing: validate → transcribe → cross-validate → chunk → embed',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['audio-processing', 'audioseek', 'mlops'],
)

def run_model_validation(**context):
    """Task to run model validation script."""
    print("Starting model validation...")
    # If your validate_model_main needs args, pass them here:
    # validate_model_main(zip_path='...', reference='...', out_csv='...')
    validate_model_main()
    print("Model validation completed!")
    return "Model validation finished successfully"

def run_transcription(**context):
    """Task to run transcription script."""
    print("Starting transcription...")
    # transcribe_audio_main(zipfile='...')
    transcribe_audio_main()
    print("Transcription completed!")
    return "Transcription finished successfully"

def run_cross_model_validation(**context):
    """Task to run cross-model validation script."""
    print("Starting cross-model validation...")
    # cross_model_eval_main(zipfile='...')
    cross_model_eval_main()
    print("Cross-model validation completed!")
    return "Cross-model validation finished successfully"

def run_chunking(**context):
    """Task to run chunking script."""
    print("Starting chunking process...")
    # chunking_main(input_dir='data/transcription_results/')
    chunking_main()
    print("Chunking completed!")
    return "Chunking process finished successfully"

def run_generate_embeddings(**context):
    """Task to run embedding generation script."""
    print("Starting embedding generation...")
    # embedding_main(input_dir='data/chunking_results/')
    embedding_main()
    print("Embedding generation completed!")
    return "Embedding generation finished successfully"

# Define tasks (using your exact task_ids)
model_validation = PythonOperator(
    task_id='model_validation',
    python_callable=run_model_validation,
    provide_context=True,
    dag=dag,
)

transcribe_audio = PythonOperator(
    task_id='transcribe_audio',
    python_callable=run_transcription,
    provide_context=True,
    dag=dag,
)

cross_model_validation = PythonOperator(
    task_id='cross_model_validation',
    python_callable=run_cross_model_validation,
    provide_context=True,
    dag=dag,
)

chunk_text = PythonOperator(
    task_id='chunk_text',
    python_callable=run_chunking,
    provide_context=True,
    dag=dag,
)

generate_embeddings = PythonOperator(
    task_id='generate_embeddings',
    python_callable=run_generate_embeddings,
    provide_context=True,
    dag=dag,
)

# Set pipeline dependencies
model_validation >> transcribe_audio >> cross_model_validation >> chunk_text >> generate_embeddings
# chunk_text >> generate_embeddings

