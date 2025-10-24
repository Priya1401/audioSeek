from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import sys
import os

sys.path.insert(0, '/opt/airflow/scripts')


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email': ['your_email@outlook.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# ----------------------------------------------------
# Function Wrappers for Each Stage
# ----------------------------------------------------

def run_model_validation():
    """Step 1: Model Validation"""
    logging.info("=" * 60)
    logging.info("STEP 1: MODEL VALIDATION")
    logging.info("=" * 60)

    try:
        from validation.model_validation.validate_model import main as validate_model_main
        validate_model_main()
        logging.info(" Model validation completed successfully.")
    except Exception as e:
        logging.error(f" Model validation failed: {str(e)}")
        raise




# ----------------------------------------------------
# DAG Definition
# ----------------------------------------------------
with DAG(
    dag_id='audio_processing_pipeline',
    default_args=default_args,
    description='Audio processing pipeline with validation, transcription, chunking, and embedding',
    schedule_interval=None,
    catchup=False,
    tags=['audioseek', 'mlops', 'audio'],
) as dag:

    model_validation_task = PythonOperator(
        task_id='model_validation',
        python_callable=run_model_validation,
    )

    # Task Dependencies
    # model_validation_task >> transcription_task >> cross_validation_task >> chunking_task >> embedding_task >> summary_task
