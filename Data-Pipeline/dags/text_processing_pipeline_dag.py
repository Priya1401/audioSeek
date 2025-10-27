from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import json
import os

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

TEXT_PROCESSING_URL = 'http://transcription-textprocessing:8001'

dag = DAG(
    'text_processing_pipeline',
    default_args=default_args,
    description='Pipeline calling text-processing service endpoints: chunk → embed → add to vector DB → query → QA',
    schedule_interval='@once',  # Run once on startup
    catchup=False,
    tags=['text-processing', 'audioseek'],
)

def chunk_transcript(**context):
    url = f"{TEXT_PROCESSING_URL}/chunk"
    data = {
        'file_path': '/opt/airflow/data/transcription_results/edison_test/audiobook_edison_test_chapter_01.txt',
        'target_tokens': 512,
        'overlap_tokens': 50,
        'output_file': '/opt/airflow/data/chunking_results/chunks.json'
    }
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()

def generate_embeddings(**context):
    url = f"{TEXT_PROCESSING_URL}/embed"
    data = {
        'chunks_file': '/opt/airflow/data/chunking_results/chunks.json',
        'output_file': '/opt/airflow/data/chunking_results/embeddings.json'
    }
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()

def add_to_vector_db(**context):
    url = f"{TEXT_PROCESSING_URL}/vector-db/add"
    # Load embeddings from file
    with open('/opt/airflow/data/chunking_results/embeddings.json', 'r') as f:
        embeddings_data = json.load(f)
    embeddings = embeddings_data['embeddings']
    # Assume metadatas are empty for simplicity
    metadatas = [{} for _ in embeddings]
    data = {
        'embeddings': embeddings,
        'metadatas': metadatas
    }
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()

def query_vector_db(**context):
    url = f"{TEXT_PROCESSING_URL}/vector-db/query"
    data = {
        'query_text': 'What is the main topic of the chapter?',
        'top_k': 5
    }
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()

def ask_qa_question(**context):
    url = f"{TEXT_PROCESSING_URL}/qa/ask"
    data = {
        'query': 'Summarize the key inventions discussed in this chapter.',
        'top_k': 5
    }
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()

# Tasks
chunk_task = PythonOperator(
    task_id='chunk_transcript',
    python_callable=chunk_transcript,
    provide_context=True,
    dag=dag,
)

embed_task = PythonOperator(
    task_id='generate_embeddings',
    python_callable=generate_embeddings,
    provide_context=True,
    dag=dag,
)

add_to_db_task = PythonOperator(
    task_id='add_to_vector_db',
    python_callable=add_to_vector_db,
    provide_context=True,
    dag=dag,
)

query_task = PythonOperator(
    task_id='query_vector_db',
    python_callable=query_vector_db,
    provide_context=True,
    dag=dag,
)

qa_task = PythonOperator(
    task_id='ask_qa_question',
    python_callable=ask_qa_question,
    provide_context=True,
    dag=dag,
)

# Set dependencies
chunk_task >> embed_task >> add_to_db_task >> query_task >> qa_task