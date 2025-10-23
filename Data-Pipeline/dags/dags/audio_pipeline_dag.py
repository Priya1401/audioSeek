from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os, sys

from scripts.validation.model_validation.validate_model import main as validate_model
from scripts.validation.cross_model_evaluation.cross_model_evaluation import main as cross_model_eval
from scripts.transcription.transcription import main as transcribe_audio
from scripts.chunking.chunking import main as chunk_text
from scripts.embedding.embedding import main as generate_embeddings

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email': ['your_email@outlook.com'],  # optional
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'audio_processing_pipeline',
    default_args=default_args,
    description='Audio processing with validation, transcription, chunking, and embedding',
    schedule_interval=None,
    catchup=False,
) as dag:

    model_validation_task = PythonOperator(
        task_id='model_validation',
        python_callable=validate_model,
        op_kwargs={
            'zip_path': 'data/validation/model_validation/sample_audio.zip',
            'reference': 'data/validation/model_validation/sample_script.txt',
            'out_csv': 'data/validation/model_validation/sample_validation_summary.csv',
        },
    )

    transcription_task = PythonOperator(
        task_id='transcribe_audio',
        python_callable=transcribe_audio,
        op_kwargs={'zipfile': 'data/validation/model_validation/sample_audio.zip'},
    )

    cross_validation_task = PythonOperator(
        task_id='cross_model_validation',
        python_callable=cross_model_eval,
        op_kwargs={'zipfile': 'data/validation/model_validation/sample_audio.zip'},
    )

    chunking_task = PythonOperator(
        task_id='chunk_text',
        python_callable=chunk_text,
        op_kwargs={'input_dir': 'data/transcription_results/'},
    )

    embedding_task = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings,
        op_kwargs={'input_dir': 'data/chunking_results/'},
    )

    model_validation_task >> transcription_task >> cross_validation_task >> chunking_task >> embedding_task
