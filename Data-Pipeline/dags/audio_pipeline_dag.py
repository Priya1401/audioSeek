from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../scripts'))

from scripts.validation.model_validation import validate_model
from scripts.validation.cross_model_evaluation import validate_transcription
from scripts.transcription.utils import audio_utils
from scripts.chunking import chunking
from scripts.embedding import embedding

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'audio_processing_pipeline',
    default_args=default_args,
    description='Audio processing with validation, transcription, chunking, and embedding',
    schedule_interval=None,
    catchup=False,
)

model_validation_task = PythonOperator(
    task_id='model_validation',
    python_callable=validate_model,
    op_kwargs={'audio_file': 'audio.wav'},
    dag=dag,
)

transcription_task = PythonOperator(
    task_id='transcribe_audio',
    python_callable=audio_utils,
    op_kwargs={'validated_audio_b64': "{{ task_instance.xcom_pull(task_ids='model_validation') }}"},
    dag=dag,
)

cross_validation_task = PythonOperator(
    task_id='cross_model_validation',
    python_callable=validate_transcription,
    op_kwargs={'transcription_b64': "{{ task_instance.xcom_pull(task_ids='transcribe_audio') }}"},
    dag=dag,
)

chunking_task = PythonOperator(
    task_id='chunk_text',
    python_callable=chunking,
    op_kwargs={'validated_transcription_b64': "{{ task_instance.xcom_pull(task_ids='cross_model_validation') }}"},
    dag=dag,
)

embedding_task = PythonOperator(
    task_id='generate_embeddings',
    python_callable=embedding,
    op_kwargs={
        'chunks_b64': "{{ task_instance.xcom_pull(task_ids='chunk_text') }}",
        'filename': 'embeddings.pkl'
    },
    dag=dag,
)

model_validation_task >> transcription_task >> cross_validation_task >> chunking_task >> embedding_task