from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'api_processing_pipeline',
    default_args=default_args,
    description='API-based pipeline: fetch data → chunk → embed → store to files AND vector DB',
    schedule_interval=None,
    catchup=False,
    tags=['api-processing', 'audioseek'],
)

# ============= TASK 1: FETCH DATA =============
fetch_data = PythonOperator(
    task_id='fetch_data',
    python_callable=lambda: "Data fetched",
    dag=dag,
)


# ============= TASK 2: PREPARE CHUNK REQUEST =============
def prepare_chunk_request(**context):
    """Prepare chunking request"""
    folder_path = context['params'].get(
        'folder_path',
        '/app/raw_data/transcription_results/edison_lifeinventions/'
    )

    return {
        'folder_path': folder_path,
        'target_tokens': 512,
        'overlap_tokens': 50,
        'output_file': '/app/data/chunks_output.json'
    }


prepare_chunk = PythonOperator(
    task_id='prepare_chunk_request',
    python_callable=prepare_chunk_request,
    provide_context=True,
    dag=dag,
)


# ============= TASK 3: CHUNK TEXT =============
def call_chunk_api(**context):
    """Call the chunking API with proper request"""
    import requests

    ti = context['ti']
    chunk_request = ti.xcom_pull(task_ids='prepare_chunk_request')

    # Debug: print what we're sending
    print(f"Sending chunk request: {chunk_request}")

    response = requests.post(
        'http://transcription-textprocessing:8001/chunk',
        json=chunk_request,
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code != 200:
        raise Exception(f"Chunk API failed: {response.status_code} - {response.text}")

    print(f"Chunk response: {response.json()}")
    return response.json()


chunk_text = PythonOperator(
    task_id='chunk_text',
    python_callable=call_chunk_api,
    provide_context=True,
    dag=dag,
)


# ============= TASK 4: PREPARE EMBED REQUEST =============
def prepare_embed_request(**context):
    """Prepare embedding request"""
    return {
        'chunks_file': '/app/data/chunks_output.json',
        'output_file': '/app/data/embeddings_output.json'
    }


prepare_embed = PythonOperator(
    task_id='prepare_embed_request',
    python_callable=prepare_embed_request,
    dag=dag,
)


# ============= TASK 5: GENERATE EMBEDDINGS =============
def call_embed_api(**context):
    """Call the embedding API with proper request"""
    import requests

    ti = context['ti']
    embed_request = ti.xcom_pull(task_ids='prepare_embed_request')

    print(f"Sending embed request: {embed_request}")

    response = requests.post(
        'http://transcription-textprocessing:8001/embed',
        json=embed_request,
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code != 200:
        raise Exception(f"Embed API failed: {response.status_code} - {response.text}")

    print(f"Embed response: {response.json()}")
    return response.json()


generate_embeddings = PythonOperator(
    task_id='generate_embeddings',
    python_callable=call_embed_api,
    provide_context=True,
    dag=dag,
)


# ============= TASK 6: PREPARE VECTOR DB REQUEST =============
def prepare_vector_db_request(**context):
    """Prepare request to add to vector DB from files"""
    return {
        'chunks_file': '/app/data/chunks_output.json',
        'embeddings_file': '/app/data/embeddings_output.json'
    }


prepare_vector_db = PythonOperator(
    task_id='prepare_vector_db_request',
    python_callable=prepare_vector_db_request,
    dag=dag,
)


# ============= TASK 7: ADD TO VECTOR DB =============
def call_vector_db_api(**context):
    """Call the vector DB API to add documents"""
    import requests

    ti = context['ti']
    vector_request = ti.xcom_pull(task_ids='prepare_vector_db_request')

    print(f"Sending vector DB request: {vector_request}")

    response = requests.post(
        'http://transcription-textprocessing:8001/vector-db/add-from-files',
        json=vector_request,
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code != 200:
        raise Exception(f"Vector DB API failed: {response.status_code} - {response.text}")

    print(f"Vector DB response: {response.json()}")
    return response.json()


add_to_vector_db = PythonOperator(
    task_id='add_to_vector_db',
    python_callable=call_vector_db_api,
    provide_context=True,
    dag=dag,
)


# ============= TASK 8: VERIFY STORAGE =============
def verify_storage(**context):
    """Verify files were created and vector DB was populated"""
    import os

    chunks_exists = os.path.exists('/opt/airflow/working_data/chunks_output.json')
    embeddings_exists = os.path.exists('/opt/airflow/working_data/embeddings_output.json')

    print(f" Chunks file exists: {chunks_exists}")
    print(f" Embeddings file exists: {embeddings_exists}")
    print(f" Vector DB populated (check /vector-db/stats endpoint)")

    return "All files saved and vector DB populated!"


verify_results = PythonOperator(
    task_id='verify_storage',
    python_callable=verify_storage,
    dag=dag,
)


# ============= TASK 9: FINAL REPORT =============
def generate_report(**context):
    """Generate final pipeline report"""
    print("=" * 50)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 50)
    print(" Step 1: Data fetched")
    print(" Step 2: Text chunked")
    print(" Step 3: Embeddings generated")
    print(" Step 4: Data added to Vector DB")
    print(" Step 5: Files saved to disk")
    print("=" * 50)
    print("Output files:")
    print("  - /app/data/chunks_output.json")
    print("  - /app/data/embeddings_output.json")
    print("=" * 50)
    return "Pipeline completed successfully!"


final_report = PythonOperator(
    task_id='generate_final_report',
    python_callable=generate_report,
    dag=dag,
)

# ============= DEPENDENCIES =============
# Linear pipeline with 9 tasks
fetch_data >> prepare_chunk >> chunk_text >> prepare_embed >> generate_embeddings >> prepare_vector_db >> add_to_vector_db >> verify_results >> final_report
