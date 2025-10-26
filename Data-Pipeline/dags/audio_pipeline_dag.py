from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# # Add the project root to Python path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 24),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'audio_processing_pipeline',
    default_args=default_args,
    description='DAG for audio processing: validate → transcribe → cross-validate → chunk → embed',
    schedule_interval=None,  # manual trigger
    catchup=False,
    tags=['audio-processing', 'audioseek', 'mlops'],
    params={
        # -------- VALIDATE MODEL (only used by model_validation task) --------
        'validation_zipfile': 'data/validation/model_validation/sample_audio.zip',
        'validation_reference': 'data/validation/model_validation/sample_script.txt',
        'validation_out': 'data/validation/model_validation/sample_validation_summary.csv',
        'validation_model': 'base',
        'validation_beam_size': 5,
        'validation_compute_type': 'float32',

        # -------- TRANSCRIPTION (only used by transcribe_audio task) --------
        'transcription_zipfile': 'data/raw/edison_lifeinventions.zip',
        'transcription_outdir': 'data/transcription_results/edison_lifeinventions',
        'transcription_type': 'audiobook',
        'transcription_model': 'base',
        'transcription_beam_size': 5,
        'transcription_compute_type': 'float32',

        # -------- CROSS-MODEL EVAL (only used by cross_model_validation task) --------
        'cross_zipfile': 'data/raw/edison_lifeinventions.zip',
        'cross_type': 'audiobook',
        'cross_sample_size': 1,
    },
)


def _gp(context, key, default=None):
    """Get param from dag_run.conf first, else DAG params, else default."""
    conf = context.get('dag_run').conf if context.get('dag_run') else {}
    if key in conf:
        return conf[key]
    return context.get('params', {}).get(key, default)


def _ensure_hf_cache():
    """
    Force Hugging Face / Transformers / Torch caches to writable paths inside container.
    Override env (not setdefault) so we beat any preexisting values.
    """
    cache_root = "/opt/airflow/.cache"
    hf_cache = f"{cache_root}/huggingface"

    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_HUB_CACHE"] = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["TORCH_HOME"] = f"{cache_root}/torch"
    os.environ["XDG_CACHE_HOME"] = cache_root
    os.environ["HOME"] = "/opt/airflow"  # so "~/.cache" resolves under /opt/airflow

    Path(hf_cache).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    # Writable sanity check
    testfile = Path(hf_cache) / ".write_test"
    testfile.write_text("ok", encoding="utf-8")
    testfile.unlink(missing_ok=True)
    print(f"[HF_CACHE] HF_HOME={os.environ['HF_HOME']}  XDG_CACHE_HOME={os.environ['XDG_CACHE_HOME']}  HOME={os.environ['HOME']}")


def run_model_validation(**context):
    _ensure_hf_cache()
    from scripts.validation.model_validation.validate_model import main as validate_model_main

    print("Starting model validation...")

    zipfile = _gp(context, 'validation_zipfile')
    reference = _gp(context, 'validation_reference')
    out_csv = _gp(context, 'validation_out')
    model = _gp(context, 'validation_model', 'base')
    beam_size = _gp(context, 'validation_beam_size', 5)
    compute_type = _gp(context, 'validation_compute_type', 'float32')

    if not zipfile or not reference:
        raise ValueError("validation_zipfile and validation_reference are required.")

    argv = [
        "validate_model.py",
        "--zipfile", str(zipfile),
        "--reference", str(reference),
        "--model", str(model),
        "--beam-size", str(beam_size),
        "--compute-type", str(compute_type),
    ]
    if out_csv:
        argv += ["--out", str(out_csv)]

    old_argv = sys.argv
    try:
        sys.argv = argv
        try:
            validate_model_main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv

    context['ti'].xcom_push(key="validation_summary_csv", value=str(out_csv) if out_csv else "")
    print("Model validation completed!")
    return f"Model validation finished (summary: {out_csv})"


def run_transcription(**context):
    _ensure_hf_cache()
    from scripts.transcription.transcription import main as transcribe_audio_main

    print("Starting transcription...")

    zipfile = _gp(context, 'transcription_zipfile')
    outdir = _gp(context, 'transcription_outdir')
    type_name = _gp(context, 'transcription_type', 'audiobook')
    model = _gp(context, 'transcription_model', 'base')
    beam_size = _gp(context, 'transcription_beam_size', 5)
    compute_type = _gp(context, 'transcription_compute_type', 'float32')

    if not zipfile or not outdir:
        raise ValueError("transcription_zipfile and transcription_outdir are required.")

    argv = [
        "transcription.py",
        "--zipfile", str(zipfile),
        "--type", str(type_name),
        "--outdir", str(outdir),
        "--model", str(model),
        "--beam-size", str(beam_size),
        "--compute-type", str(compute_type),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        try:
            transcribe_audio_main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv

    zip_base = Path(zipfile).stem
    summary_csv = Path(outdir) / f"{type_name.lower()}_{zip_base}_summary.csv"
    context['ti'].xcom_push(key="transcription_summary_csv", value=str(summary_csv))
    print("Transcription completed!")
    return f"Transcription finished (summary: {summary_csv})"


def run_cross_model_validation(**context):
    _ensure_hf_cache()
    from scripts.validation.cross_model_evaluation.cross_model_evaluation import main as cross_model_eval_main

    print("Starting cross-model evaluation...")

    zipfile = _gp(context, 'cross_zipfile')
    type_name = _gp(context, 'cross_type', 'audiobook')
    sample_size = int(_gp(context, 'cross_sample_size', 3))

    if not zipfile:
        raise ValueError("cross_zipfile is required.")

    argv = [
        "cross_model_evaluation.py",
        "--zipfile", str(zipfile),
        "--type", str(type_name),
        "--sample-size", str(sample_size),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        try:
            cross_model_eval_main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv

    zip_base = Path(zipfile).stem
    out_csv = f"data/validation/cross_model_evaluation/{type_name}_{zip_base}_validation_summary.csv"
    context['ti'].xcom_push(key="cross_model_eval_summary_csv", value=out_csv)
    print("Cross-model evaluation completed!")
    return f"Cross-model evaluation finished (summary: {out_csv})"


def run_chunking(**context):
    _ensure_hf_cache()
    from scripts.chunking.chunking import main as chunking_main

    print("Starting chunking...")
    try:
        chunking_main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    print("Chunking completed!")
    return "Chunking finished"


def run_generate_embeddings(**context):
    _ensure_hf_cache()
    from scripts.embedding.embedding import main as embedding_main

    print("Starting embedding generation...")
    try:
        embedding_main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    print("Embedding generation completed!")
    return "Embedding finished"


# Operators
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

# Dependencies
model_validation >> transcribe_audio >> cross_model_validation >> chunk_text >> generate_embeddings
