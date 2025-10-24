from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.chunking.chunking import main as chunking_main

# Default arguments for the DAG
default_args = {
    'owner': 'mirudula',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 24),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'chunking_pipeline',
    default_args=default_args,
    description='DAG for audio chunking process',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['chunking', 'audio-processing'],
)

def run_chunking(**context):
    """
    Task to run chunking script from chunking.py
    """
    print("Starting chunking process...")
    chunking_main()
    print("Chunking completed!")
    return "Chunking process finished successfully"

# Define task
task_chunking = PythonOperator(
    task_id='run_chunking',
    python_callable=run_chunking,
    provide_context=True,
    dag=dag,
)

# Single task - no dependencies needed
task_chunking