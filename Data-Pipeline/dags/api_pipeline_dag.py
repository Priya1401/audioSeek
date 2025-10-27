from airflow import DAG
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import json

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'api_processing_pipeline',
    default_args=default_args,
    description='API-based pipeline: fetch data → chunk → embed → store',
    schedule_interval=None,
    catchup=False,
    tags=['api-processing', 'audioseek'],
)

def prepare_chunk_request(**context):
    transcript_path = context['params'].get('transcript_file', 'data/faster_whisper.txt')
    # Return just the dict, not wrapped in anything
    return {
        'file_path': transcript_path,
        'target_tokens': 512,
        'overlap_tokens': 50,
        'output_file': 'data/api_chunks.json'
    }

def prepare_embed_request(**context):
    return {
        'chunks_file': 'data/api_chunks.json',
        'output_file': 'data/api_embeddings.json'
    }

fetch_data = PythonOperator(
    task_id='fetch_data',
    python_callable=lambda: "Data fetched",
    dag=dag,
)

prepare_chunk = PythonOperator(
    task_id='prepare_chunk_request',
    python_callable=prepare_chunk_request,
    provide_context=True,
    dag=dag,
)

# Fix: Use a proper callable to build the request body
def build_chunk_request(**context):
    ti = context['ti']
    chunk_data = ti.xcom_pull(task_ids='prepare_chunk_request')
    return json.dumps(chunk_data)

chunk_text = SimpleHttpOperator(
    task_id='chunk_text',
    http_conn_id='text_processing_api',
    endpoint='chunk',
    method='POST',
    data="{{ ti.xcom_pull(task_ids='prepare_chunk_request') | tojson }}",
    headers={'Content-Type': 'application/json'},
    response_check=lambda response: response.status_code == 200,
    dag=dag,
)

prepare_embed = PythonOperator(
    task_id='prepare_embed_request',
    python_callable=prepare_embed_request,
    dag=dag,
)

generate_embeddings = SimpleHttpOperator(
    task_id='generate_embeddings',
    http_conn_id='text_processing_api',
    endpoint='embed',
    method='POST',
    data="{{ ti.xcom_pull(task_ids='prepare_embed_request') | tojson }}",
    headers={'Content-Type': 'application/json'},
    response_check=lambda response: response.status_code == 200,
    dag=dag,
)

store_results = PythonOperator(
    task_id='store_results',
    python_callable=lambda: "Results stored",
    dag=dag,
)

# Set up dependencies
fetch_data >> prepare_chunk >> chunk_text >> prepare_embed >> generate_embeddings >> store_results