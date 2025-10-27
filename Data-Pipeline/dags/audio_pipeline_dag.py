from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

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
    description='DAG for audio processing: validate → transcribe → cross-validate (parallel) → chunk → embed',
    schedule_interval=None,
    catchup=False,
    tags=['audio-processing', 'audioseek', 'mlops'],
    params={
        # -------- MODEL VALIDATION - TRANSCRIPTION STEP --------
        'validation_audio': 'data/validation/model_validation/sample_audio',
        'validation_model': 'base',
        'validation_beam_size': 5,
        'validation_compute_type': 'float32',

        # -------- MODEL VALIDATION - VALIDATION STEP --------
        'validation_reference': 'data/validation/model_validation/sample_script.txt',
        'validation_out': 'data/validation/model_validation/sample_validation_summary.csv',
        'validation_model_name': 'Faster-Whisper(base)',

        # -------- TRANSCRIPTION (only used by transcribe_audio task) --------
        'transcription_zipfile': 'data/raw/edison_lifeinventions.zip',
        'transcription_outdir': 'data/transcription_results/edison_lifeinventions',
        'transcription_type': 'audiobook',
        'transcription_model': 'base',
        'transcription_beam_size': 5,
        'transcription_compute_type': 'float32',

        # -------- CROSS-MODEL EVAL (used by cross_model_validation tasks) --------
        'cross_folder': 'data/raw/edison_lifeinventions',
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


# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def run_transcribe_reference(**context):
    """
    Step 1 of Model Validation: Transcribe reference audio using Faster-Whisper
    """
    _ensure_hf_cache()
    from scripts.validation.model_validation.transcribe_reference import main as transcribe_ref_main

    print("Starting reference audio transcription...")

    audio_path = _gp(context, 'validation_audio')
    model = _gp(context, 'validation_model', 'base')
    beam_size = _gp(context, 'validation_beam_size', 5)
    compute_type = _gp(context, 'validation_compute_type', 'float32')


    argv = [
        "transcribe_reference.py",
        "--audio", str(audio_path),
        "--model", str(model),
        "--beam-size", str(beam_size),
        "--compute-type", str(compute_type),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        try:
            transcribe_ref_main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv

    print(f"Reference transcription completed!")
    return f"Transcription finished"


def run_validate_transcription(**context):
    """
    Step 2 of Model Validation: Validate generated transcript against official reference
    """
    _ensure_hf_cache()
    from scripts.validation.model_validation.validate_transcription import main as validate_trans_main

    print("Starting transcription validation...")

    reference = _gp(context, 'validation_reference')
    out_csv = _gp(context, 'validation_out')
    model_name = _gp(context, 'validation_model_name', 'Faster-Whisper(base)')


    argv = [
        "validate_transcription.py",
        "--reference", str(reference),
        "--out", str(out_csv),
        "--model-name", str(model_name),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        try:
            validate_trans_main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv

    context['ti'].xcom_push(key="validation_summary_csv", value=str(out_csv))
    print(f"Validation completed! Summary saved to: {out_csv}")
    return f"Validation finished (summary: {out_csv})"

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


# ============================================================================
# NEW: CROSS-MODEL VALIDATION - SPLIT INTO 4 PARALLEL TASKS
# ============================================================================

def run_create_sample_zip(**context):
    """
    STEP 1: Create sample ZIP for cross-model evaluation
    This runs once, then feeds both parallel transcription tasks
    """
    _ensure_hf_cache()
    from scripts.transcription.utils.audio_utils import sample_zip_filtered
    
    print("=" * 70)
    print("STEP 1/3: Creating Sample ZIP for Cross-Model Evaluation")
    print("=" * 70)
    
    # Get parameters
    folder = _gp(context, 'cross_folder')
    type_name = _gp(context, 'cross_type', 'audiobook')
    sample_size = int(_gp(context, 'cross_sample_size', 1))
    
    if not folder:
        raise ValueError("cross_folder parameter is required.")
    
    print(f"Source Folder: {folder}")
    print(f"Content Type: {type_name}")
    print(f"Sample Size: {sample_size} file(s)")
    
    # Create paths
    source_path = Path(folder)
    sample_zip_path = Path("data/raw/sample_subset.zip")
    sample_zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create sample ZIP
    out_path = sample_zip_filtered(source_path, sample_size, sample_zip_path)
    print(f"Sample ZIP created: {out_path}")
    
    # Push to XCom for downstream parallel tasks
    context['ti'].xcom_push(key='sample_zip_path', value=str(out_path))
    context['ti'].xcom_push(key='content_type', value=type_name)
    context['ti'].xcom_push(key='folder', value=str(folder))
    
    print("=" * 70)
    print("STEP 1 COMPLETE - Ready for parallel transcription")
    print("=" * 70)
    
    return f"Sample ZIP created: {out_path}"


def run_transcribe_openai_whisper(**context):
    """
    STEP 2a: OpenAI Whisper transcription
    ⚡ Runs IN PARALLEL with Wav2Vec2 task
    """
    _ensure_hf_cache()
    from scripts.validation.cross_model_evaluation.cross_model_sample_openaiwhisper import transcribe_sample_openaiwhisper
    
    print("=" * 70)
    print("STEP 2a/3: OpenAI Whisper Transcription ⚡ PARALLEL")
    print("=" * 70)
    
    # Pull data from XCom
    ti = context['ti']
    sample_zip_path = ti.xcom_pull(task_ids='create_sample_zip', key='sample_zip_path')
    content_type = ti.xcom_pull(task_ids='create_sample_zip', key='content_type')

    if not sample_zip_path:
        raise ValueError("sample_zip_path not found in XCom. Ensure 'create_sample_zip' completed successfully.")
    
    print(f"Input: {sample_zip_path}")
    print(f"Content Type: {content_type}")
    print(f"Model: OpenAI Whisper (base)")
    
    # Create output directory
    output_dir = Path("data/validation/cross_model_evaluation/openaiwhisper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run transcription
    transcribe_sample_openaiwhisper(
        zipfile_path=sample_zip_path,
        outdir_path=str(output_dir),
        content_type=content_type,
        model_size="base"
    )
    
    print(f"OpenAI Whisper transcription complete!")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    return "OpenAI Whisper transcription finished"


def run_transcribe_wav2vec(**context):
    """
    STEP 2b: Wav2Vec2 transcription
    ⚡ Runs IN PARALLEL with OpenAI Whisper task
    """
    _ensure_hf_cache()
    from scripts.validation.cross_model_evaluation.cross_model_sample_wav2vec import transcribe_sample_wav2vec
    
    print("=" * 70)
    print("STEP 2b/3: Wav2Vec2 Transcription ⚡ PARALLEL")
    print("=" * 70)
    
    # Pull data from XCom
    ti = context['ti']
    sample_zip_path = ti.xcom_pull(task_ids='create_sample_zip', key='sample_zip_path')
    content_type = ti.xcom_pull(task_ids='create_sample_zip', key='content_type')
    
    if not sample_zip_path:
        raise ValueError("sample_zip_path not found in XCom. Ensure 'create_sample_zip' completed successfully.")
    
    print(f"Input: {sample_zip_path}")
    print(f"Content Type: {content_type}")
    print(f"Model: facebook/wav2vec2-base-960h")
    
    # Create output directory
    output_dir = Path("data/validation/cross_model_evaluation/wav2vec2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run transcription
    transcribe_sample_wav2vec(
        zipfile_path=sample_zip_path,
        outdir_path=str(output_dir),
        content_type=content_type
    )
    
    print(f"Wav2Vec2 transcription complete!")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    return "Wav2Vec2 transcription finished"


def run_validate_cross_models(**context):
    """
    STEP 3: Cross-model validation
    Runs AFTER both parallel transcription tasks complete
    Compares Faster-Whisper vs OpenAI Whisper vs Wav2Vec2
    """
    _ensure_hf_cache()
    from scripts.validation.cross_model_evaluation.validate_transcription import validate_models
    
    print("=" * 70)
    print("STEP 3/3: Cross-Model Validation (After Parallel Tasks)")
    print("=" * 70)
    
    # Pull data from XCom
    ti = context['ti']
    content_type = ti.xcom_pull(task_ids='create_sample_zip', key='content_type')
    folder = ti.xcom_pull(task_ids='create_sample_zip', key='folder')
    sample_zip_path = ti.xcom_pull(task_ids='create_sample_zip', key='sample_zip_path')
    sample_zip = Path(sample_zip_path)
    if sample_zip:
        sample_zip.unlink()
        print(f"Cleared Sample Zip - {sample_zip_path}")

    # Get transcription output directory
    transcription_outdir = _gp(context, 'transcription_outdir')
    
    print(f"Comparing three models:")
    print(f"   Faster-Whisper: {transcription_outdir}")
    print(f"   OpenAI Whisper: data/validation/cross_model_evaluation/openaiwhisper")
    print(f"   Wav2Vec2: data/validation/cross_model_evaluation/wav2vec2")
    
    # Prepare output CSV
    source_base = Path(folder).stem if folder else "unknown"
    out_csv = Path(f"data/validation/cross_model_evaluation/{content_type}_{source_base}_validation_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Validation summary: {out_csv}")
    print(f" Thresholds: WER < 45%, ROUGE-L > 60%")
    print("\n Running validation...")
    
    # Run validation (will sys.exit(1) if validation fails)
    validate_models(
        fw_dir=transcription_outdir,
        ow_dir="data/validation/cross_model_evaluation/openaiwhisper",
        w2v_dir="data/validation/cross_model_evaluation/wav2vec2",
        out_csv=out_csv,
        content_type=content_type,
        wer_threshold=0.45,
        rouge_threshold=0.60
    )
    
    # If we reach here, validation passed
    print("\n" + "=" * 70)
    print("VALIDATION PASSED!")
    print(f"Summary: {out_csv}")
    print("=" * 70)
    
    # Store in XCom
    context['ti'].xcom_push(key="cross_model_validation_summary_csv", value=str(out_csv))
    
    return f"Cross-model validation passed (summary: {out_csv})"


# ============================================================================
# REMAINING TASK FUNCTIONS
# ============================================================================

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


# ============================================================================
# OPERATORS
# ============================================================================

# Model Validation - Step 1: Transcribe Reference Audio
transcribe_reference_audio = PythonOperator(
    task_id='transcribe_reference_audio',
    python_callable=run_transcribe_reference,
    provide_context=True,
    dag=dag,
)

# Model Validation - Step 2: Validate Transcription
validate_transcription = PythonOperator(
    task_id='validate_transcription',
    python_callable=run_validate_transcription,
    provide_context=True,
    dag=dag,
)

transcribe_audio = PythonOperator(
    task_id='transcribe_audio',
    python_callable=run_transcription,
    provide_context=True,
    dag=dag,
)

# --- Cross-Model Validation: 4 Separate Tasks ---

create_sample_zip = PythonOperator(
    task_id='create_sample_zip',
    python_callable=run_create_sample_zip,
    provide_context=True,
    dag=dag,
)

transcribe_openai_whisper = PythonOperator(
    task_id='transcribe_openai_whisper',
    python_callable=run_transcribe_openai_whisper,
    provide_context=True,
    dag=dag,
)

transcribe_wav2vec = PythonOperator(
    task_id='transcribe_wav2vec',
    python_callable=run_transcribe_wav2vec,
    provide_context=True,
    dag=dag,
)

validate_cross_models = PythonOperator(
    task_id='validate_cross_models',
    python_callable=run_validate_cross_models,
    provide_context=True,
    dag=dag,
)

# --- Remaining Tasks ---

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


# ============================================================================
# TASK DEPENDENCIES - PARALLEL EXECUTION
# ============================================================================

# Linear flow
transcribe_reference_audio >> validate_transcription  >> transcribe_audio >> create_sample_zip

# PARALLEL BRANCHING: Both models transcribe simultaneously
create_sample_zip >> [transcribe_openai_whisper, transcribe_wav2vec]

# CONVERGENCE: Validation waits for BOTH parallel tasks
[transcribe_openai_whisper, transcribe_wav2vec] >> validate_cross_models

# Continue pipeline
validate_cross_models >> chunk_text >> generate_embeddings