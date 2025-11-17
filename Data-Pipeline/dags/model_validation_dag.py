import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.email import send_email

# Add /opt/airflow to path
# sys.path.insert(0, '/opt/airflow')

# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ALERT_EMAILS = os.getenv('ALERT_EMAILS', 'default@example.com').split(',')

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 24),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ALERT_EMAILS,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_validation_pipeline',
    default_args=default_args,
    description='Validate embeddings, Q&A performance, and detect bias',
    schedule_interval=None,
    catchup=False,
    tags=['validation', 'qa', 'embeddings', 'bias', 'mlops'],
    params={
        'book_id': 'romeo_and_juliet',
        'min_rouge_score': 0.4,
        'min_citation_count': 1,
        'max_response_time': 15.0,
        'bias_threshold': 0.20,
    },
)

# Base URL for API calls
API_BASE_URL = 'http://transcription-textprocessing:8001'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _gp(context, key, default=None):
    """Get param from dag_run.conf first, else DAG params, else default."""
    conf = context.get('dag_run').conf if context.get('dag_run') else {}
    if key in conf:
        return conf[key]
    return context.get('params', {}).get(key, default)


def send_validation_alert(context, message, stage):
    """Send email alert for validation issues"""
    task_instance = context['task_instance']
    dag_run = context['dag_run']

    subject = f"[VALIDATION ALERT] {stage}"

    html_content = f"""
    <html>
    <body>
        <h2 style="color: #f0ad4e;">Validation Alert</h2>
        <p><strong>DAG:</strong> {task_instance.dag_id}</p>
        <p><strong>Task:</strong> {task_instance.task_id}</p>
        <p><strong>Stage:</strong> {stage}</p>

        <h3>Details:</h3>
        <p>{message}</p>

        <p><a href="http://localhost:8080/dags/{task_instance.dag_id}/grid">View in Airflow</a></p>
    </body>
    </html>
    """

    try:
        send_email(to=ALERT_EMAILS, subject=subject, html_content=html_content)
        logger.info(f"Alert sent to {ALERT_EMAILS}")
    except Exception as e:
        logger.exception(f"Failed to send alert: {e}")


# ============================================================================
# VALIDATION TASKS
# ============================================================================

def load_data_for_validation(**context):
    """Load vector DB stats to verify data exists"""
    import requests

    book_id = _gp(context, 'book_id', 'romeo_and_juliet')

    logger.info("=" * 70)
    logger.info(f"LOADING DATA FOR VALIDATION: {book_id}")
    logger.info("=" * 70)

    # API Endpoint: GET /vector-db/stats
    response = requests.get(
        f'{API_BASE_URL}/vector-db/stats',
        params={'book_id': book_id},
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(
            f"Could not load vector DB stats for {book_id}: {response.text}")

    stats = response.json()

    logger.info(f"  book_id: {stats.get('book_id')}")
    logger.info(f"  Vector count: {stats.get('vector_count')}")
    logger.info(f"  Metadata count: {stats.get('metadata_count')}")

    if stats.get('vector_count', 0) == 0:
        raise Exception(
            f"No data for {book_id}. Run audio_processing_pipeline first.")

    context['ti'].xcom_push(key='book_id', value=book_id)
    context['ti'].xcom_push(key='vector_stats', value=stats)

    logger.info(f"✓ Data loaded for {book_id}")
    logger.info("=" * 70)

    return stats


def validate_embeddings_task(**context):
    """Validate embedding quality"""
    import requests
    from scripts.validation.model_validation.QA.embedding_validation import EmbeddingValidator

    ti = context['ti']
    book_id = ti.xcom_pull(task_ids='load_data_for_validation', key='book_id')

    logger.info(f"Running embedding validation for {book_id}")

    # API Endpoint: POST /vector-db/query
    # Get sample embeddings/chunks from vector DB
    response = requests.post(
        f'{API_BASE_URL}/vector-db/query',
        json={'book_id': book_id, 'query': 'sample validation query',
              'top_k': 200},
        headers={'Content-Type': 'application/json'},
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"Could not retrieve data: {response.text}")

    results = response.json().get('results', [])
    chunks = [r['metadata'] for r in results]

    logger.info(f"Retrieved {len(chunks)} chunks for validation")

    # Try to load full embeddings from file (optional)
    embeddings = []
    possible_paths = [
        f'/opt/airflow/data/embedding_results/{book_id}_embedding.json',
        f'/opt/airflow/working_data/embeddings_output.json',
    ]

    for path in possible_paths:
        if Path(path).exists():
            try:
                with open(path, 'r') as f:
                    embeddings_data = json.load(f)
                embeddings = embeddings_data.get('embeddings', [])
                logger.info(f"Loaded {len(embeddings)} embeddings from {path}")
                break
            except Exception as e:
                logger.warning(f"Could not load from {path}: {e}")

    # Run validation
    validator = EmbeddingValidator()

    if embeddings:
        validation_result = validator.validate_embeddings(
            embeddings=embeddings,
            chunks=chunks,
            book_id=book_id
        )
    else:
        logger.warning("Running limited validation without embedding file")
        validation_result = {
            'validation_passed': len(chunks) > 0,
            'metrics': {
                'chunks_count': len(chunks),
                'status': 'limited_validation'
            }
        }

    # Save report
    report_path = f"data/validation/embedding_validation_{book_id}.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(validation_result, f, indent=2)

    context['ti'].xcom_push(key='embedding_validation', value=validation_result)

    if not validation_result['validation_passed']:
        send_validation_alert(
            context,
            validation_result.get('failure_reason', 'Validation failed'),
            "Embedding Validation"
        )
        raise Exception("Embedding validation failed")

    logger.info(f"✓ Embedding validation passed")
    return validation_result


def validate_qa_task(**context):
    """Validate Q&A performance"""
    import requests
    from scripts.validation.model_validation.QA.qa_validation import QAValidator

    ti = context['ti']
    book_id = ti.xcom_pull(task_ids='load_data_for_validation', key='book_id')

    logger.info(f"Running Q&A validation for {book_id}")

    # Create validator
    validator = QAValidator(
        min_rouge_score=float(_gp(context, 'min_rouge_score', 0.4)),
        min_citation_count=int(_gp(context, 'min_citation_count', 1)),
        max_response_time=float(_gp(context, 'max_response_time', 15.0))
    )

    # Get test cases
    test_cases = validator.create_test_cases(book_id)

    if not test_cases:
        logger.warning(f"No test cases for {book_id}, using generic tests")
        test_cases = [
            {
                'query': 'What is this story about?',
                'expected_answer': 'A story',
                'query_type': 'general'
            },
            {
                'query': 'Who are the main characters?',
                'expected_answer': 'The main characters',
                'query_type': 'factual'
            }
        ]

    # Run tests via API
    logger.info(f"Running {len(test_cases)} test queries...")
    results = []

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}/{len(test_cases)}: {test_case['query']}")

        start_time = time.time()

        try:
            # API Endpoint: POST /qa/ask
            response = requests.post(
                f'{API_BASE_URL}/qa/ask',
                json={
                    'book_id': book_id,
                    'query': test_case['query'],
                    'top_k': 5
                },
                headers={'Content-Type': 'application/json'},
                timeout=30
            )

            response_time = time.time() - start_time

            if response.status_code != 200:
                logger.error(f"  API error: {response.status_code}")
                results.append({
                    'query': test_case['query'],
                    'test_passed': False,
                    'error': f"API error: {response.status_code}"
                })
                continue

            qa_response = response.json()

            # Calculate ROUGE-L
            rouge_score = validator.scorer.score(
                test_case['expected_answer'],
                qa_response['answer']
            )
            rouge_l = rouge_score['rougeL'].fmeasure

            # Check criteria
            rouge_passed = rouge_l >= validator.min_rouge_score
            citations_passed = len(
                qa_response['citations']) >= validator.min_citation_count
            time_passed = response_time <= validator.max_response_time

            result = {
                'query': test_case['query'],
                'query_type': test_case.get('query_type'),
                'expected_answer': test_case['expected_answer'],
                'actual_answer': qa_response['answer'],
                'rouge_l': round(rouge_l, 4),
                'citations_count': len(qa_response['citations']),
                'response_time': round(response_time, 3),
                'rouge_passed': rouge_passed,
                'citations_passed': citations_passed,
                'time_passed': time_passed,
                'test_passed': rouge_passed and citations_passed and time_passed
            }

            logger.info(
                f"  ROUGE-L: {rouge_l:.3f} ({'✓' if rouge_passed else '✗'})")
            logger.info(
                f"  Citations: {len(qa_response['citations'])} ({'✓' if citations_passed else '✗'})")
            logger.info(
                f"  Time: {response_time:.2f}s ({'✓' if time_passed else '✗'})")

        except Exception as e:
            logger.error(f"  Error: {e}")
            result = {'query': test_case['query'], 'test_passed': False,
                      'error': str(e)}

        results.append(result)

    # Calculate summary
    total = len(results)
    passed = sum(1 for r in results if r.get('test_passed', False))
    pass_rate = passed / total if total > 0 else 0

    validation_result = {
        'book_id': book_id,
        'validation_passed': pass_rate >= 0.7,
        'individual_results': results,
        'summary': {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': pass_rate
        }
    }

    # Save report
    report_path = f"data/validation/qa_validation_{book_id}.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(validation_result, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Q&A VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total tests: {total}")
    logger.info(f"Passed: {passed} ({pass_rate:.1%})")
    logger.info(f"Failed: {total - passed}")
    logger.info(
        f"Overall: {'✓ PASSED' if validation_result['validation_passed'] else '✗ FAILED'}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 70)

    context['ti'].xcom_push(key='qa_validation', value=validation_result)

    if not validation_result['validation_passed']:
        send_validation_alert(
            context,
            f"Q&A validation failed: {pass_rate:.1%} pass rate (threshold: 70%)",
            "Q&A Validation"
        )

    return validation_result


def detect_bias_task(**context):
    """Run bias detection across chapters"""
    import requests

    ti = context['ti']
    book_id = ti.xcom_pull(task_ids='load_data_for_validation', key='book_id')

    logger.info("=" * 70)
    logger.info(f"BIAS DETECTION for {book_id}")
    logger.info("=" * 70)

    # Run queries across different chapters
    test_queries = [
        {'query': 'What happened in chapter 1?', 'chapter': 1},
        {'query': 'What happened in chapter 2?', 'chapter': 2},
        {'query': 'What happened in chapter 3?', 'chapter': 3},
        {'query': 'What happened in chapter 4?', 'chapter': 4},
        {'query': 'What happened in chapter 5?', 'chapter': 5},
    ]

    results_by_chapter = {}

    for test in test_queries:
        logger.info(f"Testing chapter {test['chapter']}...")

        try:
            # API Endpoint: POST /qa/ask
            response = requests.post(
                f'{API_BASE_URL}/qa/ask',
                json={
                    'book_id': book_id,
                    'query': test['query'],
                    'top_k': 5
                },
                headers={'Content-Type': 'application/json'},
                timeout=30
            )

            if response.status_code == 200:
                qa_response = response.json()
                results_by_chapter[test['chapter']] = {
                    'citations_count': len(qa_response['citations']),
                    'answer_length': len(qa_response['answer']),
                    'has_answer': len(qa_response['answer'].strip()) > 20
                }
                logger.info(
                    f"  Chapter {test['chapter']}: {len(qa_response['citations'])} citations, answer length: {len(qa_response['answer'])} chars")
            else:
                logger.warning(
                    f"  Chapter {test['chapter']}: API error {response.status_code}")

        except Exception as e:
            logger.error(f"  Chapter {test['chapter']}: Error - {e}")

    # Calculate bias metrics
    if len(results_by_chapter) >= 3:
        citation_counts = [r['citations_count'] for r in
                           results_by_chapter.values()]
        avg_citations = np.mean(citation_counts)
        std_citations = np.std(citation_counts)
        max_citations = max(citation_counts)
        min_citations = min(citation_counts)

        citation_range = max_citations - min_citations

        # Bias detection criteria
        bias_detected = (std_citations > avg_citations * 0.5) or (
                citation_range > 3)

        bias_report = {
            'book_id': book_id,
            'bias_detected': bias_detected,
            'bias_severity': 'high' if citation_range > 5 else (
                'medium' if bias_detected else 'none'),
            'results_by_chapter': results_by_chapter,
            'metrics': {
                'avg_citations': float(avg_citations),
                'std_citations': float(std_citations),
                'min_citations': int(min_citations),
                'max_citations': int(max_citations),
                'citation_range': int(citation_range),
                'coefficient_of_variation': float(
                    std_citations / avg_citations) if avg_citations > 0 else 0
            },
            'recommendations': []
        }

        if bias_detected:
            threshold = avg_citations * 0.5
            poor_chapters = [
                ch for ch, res in results_by_chapter.items()
                if res['citations_count'] < threshold
            ]

            bias_report['recommendations'] = [
                f"Chapters with low citations: {poor_chapters}",
                f"Citation range is {citation_range} (high variation detected)",
                "Review chunking parameters for affected chapters"
            ]
    else:
        bias_report = {
            'book_id': book_id,
            'bias_detected': False,
            'status': 'insufficient_data',
            'message': f'Only tested {len(results_by_chapter)} chapters (need 3+ for bias detection)'
        }

    # Save report
    report_path = f"data/validation/bias_detection_{book_id}.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(bias_report, f, indent=2)

    logger.info(f"Bias detected: {bias_report.get('bias_detected', False)}")
    logger.info(f"Severity: {bias_report.get('bias_severity', 'none')}")

    if bias_report.get('recommendations'):
        logger.info("Recommendations:")
        for rec in bias_report['recommendations']:
            logger.info(f"  - {rec}")

    logger.info(f"Report: {report_path}")
    logger.info("=" * 70)

    context['ti'].xcom_push(key='bias_report', value=bias_report)

    if bias_report.get('bias_detected') and bias_report.get(
        'bias_severity') == 'high':
        send_validation_alert(
            context,
            f"High bias detected: {', '.join(bias_report.get('recommendations', []))}",
            "Bias Detection"
        )

    return bias_report


def generate_validation_summary_task(**context):
    """Generate comprehensive validation summary"""
    ti = context['ti']

    book_id = ti.xcom_pull(task_ids='load_data_for_validation', key='book_id')
    vector_stats = ti.xcom_pull(task_ids='load_data_for_validation')
    embedding_validation = ti.xcom_pull(task_ids='validate_embeddings_task',
                                        key='embedding_validation')
    qa_validation = ti.xcom_pull(task_ids='validate_qa_task',
                                 key='qa_validation')
    bias_report = ti.xcom_pull(task_ids='detect_bias_task', key='bias_report')

    logger.info("=" * 70)
    logger.info("COMPREHENSIVE VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"book_id: {book_id}")
    logger.info(f"Validation timestamp: {datetime.now().isoformat()}")
    logger.info(f"Total vectors in DB: {vector_stats.get('vector_count')}")
    logger.info("")

    # Embedding validation
    logger.info("1. Embedding Validation:")
    logger.info(
        f"   Status: {'✓ PASSED' if embedding_validation.get('validation_passed') else '✗ FAILED'}")
    if not embedding_validation.get('validation_passed'):
        logger.info(f"   Reason: {embedding_validation.get('failure_reason')}")

    # Q&A validation
    logger.info("\n2. Q&A Validation:")
    logger.info(
        f"   Status: {'✓ PASSED' if qa_validation.get('validation_passed') else '✗ FAILED'}")
    logger.info(f"   Pass Rate: {qa_validation['summary']['pass_rate']:.1%}")
    logger.info(
        f"   Tests Passed: {qa_validation['summary']['passed']}/{qa_validation['summary']['total_tests']}")

    # Bias detection
    logger.info("\n3. Bias Detection:")
    logger.info(
        f"   Bias Detected: {'⚠ YES' if bias_report.get('bias_detected') else '✓ NO'}")
    logger.info(f"   Severity: {bias_report.get('bias_severity', 'none')}")

    if bias_report.get('recommendations'):
        logger.info("   Recommendations:")
        for rec in bias_report['recommendations']:
            logger.info(f"     - {rec}")

    # Overall status
    overall_passed = (
        embedding_validation.get('validation_passed', False) and
        qa_validation.get('validation_passed', False) and
        (not bias_report.get('bias_detected', True) or bias_report.get(
            'bias_severity') != 'high')
    )

    summary = {
        'book_id': book_id,
        'timestamp': datetime.now().isoformat(),
        'overall_validation_passed': overall_passed,
        'components': {
            'embeddings': {
                'passed': embedding_validation.get('validation_passed', False),
                'details': embedding_validation
            },
            'qa': {
                'passed': qa_validation.get('validation_passed', False),
                'pass_rate': qa_validation['summary']['pass_rate'],
                'details': qa_validation
            },
            'bias': {
                'detected': bias_report.get('bias_detected', False),
                'severity': bias_report.get('bias_severity', 'none'),
                'details': bias_report
            }
        },
        'data_stats': vector_stats
    }

    # Save comprehensive report
    report_path = f"data/validation/comprehensive_validation_{book_id}.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info(
        f"OVERALL VALIDATION: {'✓✓✓ PASSED ✓✓✓' if overall_passed else '✗✗✗ FAILED ✗✗✗'}")
    logger.info(f"Comprehensive report: {report_path}")
    logger.info("=" * 70)

    # Send alert if overall validation failed
    if not overall_passed:
        failures = []
        if not embedding_validation.get('validation_passed'):
            failures.append("Embeddings")
        if not qa_validation.get('validation_passed'):
            failures.append("Q&A")
        if bias_report.get('bias_detected') and bias_report.get(
            'bias_severity') == 'high':
            failures.append("High Bias")

        send_validation_alert(
            context,
            f"Validation failed: {', '.join(failures)}",
            "Overall Validation"
        )

    return summary


# ============================================================================
# BUILD THE DAG - OPERATORS
# ============================================================================

load_data = PythonOperator(
    task_id='load_data_for_validation',
    python_callable=load_data_for_validation,
    provide_context=True,
    dag=dag,
)

validate_embeddings = PythonOperator(
    task_id='validate_embeddings_task',
    python_callable=validate_embeddings_task,
    provide_context=True,
    dag=dag,
)

validate_qa = PythonOperator(
    task_id='validate_qa_task',
    python_callable=validate_qa_task,
    provide_context=True,
    dag=dag,
)

detect_bias = PythonOperator(
    task_id='detect_bias_task',
    python_callable=detect_bias_task,
    provide_context=True,
    dag=dag,
)

validation_summary = PythonOperator(
    task_id='generate_validation_summary_task',
    python_callable=generate_validation_summary_task,
    provide_context=True,
    dag=dag,
)

# ============================================================================
# TASK DEPENDENCIES
# ============================================================================

# Run all validations in parallel, then generate summary
load_data >> [validate_embeddings, validate_qa,
              detect_bias] >> validation_summary