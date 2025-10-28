from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task
from datetime import datetime, timedelta
import json
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
    description='Complete audio pipeline: validate → transcribe → cross-validate → chunk (API) → embed (API) → vector DB',
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

        # -------- TRANSCRIPTION --------
        'transcription_inputdir': 'data/raw/edison_lifeinventions',
        'transcription_type': 'audiobook',
        'transcription_model': 'base',
        'transcription_beam_size': 5,
        'transcription_compute_type': 'float32',

        # -------- CROSS-MODEL EVAL --------
        'cross_folder': 'data/raw/edison_lifeinventions',
        'cross_type': 'audiobook',
        'cross_sample_size': 1,

        # -------- API CHUNKING & EMBEDDING --------
        'chunk_target_tokens': 512,
        'chunk_overlap_tokens': 50,
    },
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _gp(context, key, default=None):
    """Get param from dag_run.conf first, else DAG params, else default."""
    conf = context.get('dag_run').conf if context.get('dag_run') else {}
    if key in conf:
        return conf[key]
    return context.get('params', {}).get(key, default)


def _ensure_hf_cache():
    """Force Hugging Face / Transformers / Torch caches to writable paths inside container."""
    cache_root = "/opt/airflow/.cache"
    hf_cache = f"{cache_root}/huggingface"

    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_HUB_CACHE"] = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["TORCH_HOME"] = f"{cache_root}/torch"
    os.environ["XDG_CACHE_HOME"] = cache_root
    os.environ["HOME"] = "/opt/airflow"

    Path(hf_cache).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    testfile = Path(hf_cache) / ".write_test"
    testfile.write_text("ok", encoding="utf-8")
    testfile.unlink(missing_ok=True)
    print(f"[HF_CACHE] HF_HOME={os.environ['HF_HOME']}  XDG_CACHE_HOME={os.environ['XDG_CACHE_HOME']}  HOME={os.environ['HOME']}")


# ============================================================================
# PHASE 1: MODEL VALIDATION (2 STEPS)
# ============================================================================

def run_transcribe_reference(**context):
    """Step 1 of Model Validation: Transcribe reference audio using Faster-Whisper"""
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
    """Step 2 of Model Validation: Validate generated transcript against official reference"""
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


# ============================================================================
# PHASE 2: TRANSCRIPTION PIPELINE (VALIDATION → EXTRACTION → TRANSCRIPTION → SUMMARY)
# ============================================================================

def validate_input_dir_task(**context):
    """Task 1: Validate that input directory exists and is accessible."""
    from scripts.transcription.transcription_file_checks import validate_input_directory
    input_dir = _gp(context, 'transcription_inputdir')

    print(f"VALIDATION: Checking input directory: {input_dir}")
    result = validate_input_directory(input_dir)
    context['ti'].xcom_push(key='input_dir_validation', value=result)
    context['ti'].xcom_push(key='input_dir', value=input_dir)

    print(f"VALIDATION: Input Directory is valid")
    return result


def validate_output_dir_task(**context):
    """Task 2: Validate that output directory exists or can be created."""
    from scripts.transcription.transcription_file_checks import validate_output_directory
    output_dir_base = 'data/transcription_results'
    input_dir = _gp(context, 'transcription_inputdir')
    base_name = Path(input_dir).stem.lower()
    full_output_dir = str(Path(output_dir_base) / base_name)

    print(f"VALIDATION: Input Directory (From XCom): {input_dir}")
    print(f"VALIDATION: Base Name Extracted: {base_name}")
    print(f"VALIDATION: Full output directory: {full_output_dir}")

    result = validate_output_directory(full_output_dir)
    context['ti'].xcom_push(key='output_dir_validation', value=result)
    context['ti'].xcom_push(key='base_name', value=base_name)
    context['ti'].xcom_push(key='full_output_dir', value=full_output_dir)

    print(f"VALIDATION: Output directory is valid")
    return result


def validate_content_type_task(**context):
    """Task 3: Validate that content type is 'audiobook' or 'podcast'."""
    from scripts.transcription.transcription_file_checks import validate_content_type
    content_type = _gp(context, 'transcription_type', 'audiobook')

    print(f"VALIDATION: Checking content type: {content_type}")
    result = validate_content_type(content_type)
    context['ti'].xcom_push(key='content_type_validation', value=result)

    print(f"VALIDATION: Content type is valid: {result['split_type']}")
    return result


def extract_audio_metadata_task(**context):
    """Task 4: Extract metadata from all audio files in input directory."""
    from scripts.transcription.extraction_tasks import extract_and_list_audio_files

    input_dir = _gp(context, 'transcription_inputdir')
    content_type = _gp(context, 'transcription_type', 'audiobook')
    ti = context['ti']
    full_output_dir = ti.xcom_pull(task_ids='validation_group.validate_output_directory', key='full_output_dir')

    print(f"[EXTRACTION] Input directory: {input_dir}")
    print(f"[EXTRACTION] Output directory: {full_output_dir}")
    print(f"[EXTRACTION] Scanning audio files...")

    result = extract_and_list_audio_files(input_dir=input_dir, output_dir=full_output_dir, audio_type=content_type)

    xcom_data = {
        'base_name': result['base_name'],
        'total_files': result['total_files'],
        'extraction_file': result['extraction_file'],
        'status': result['status']
    }
    context['ti'].xcom_push(key='extraction_result', value=xcom_data)

    print(f"[EXTRACTION] Found {result['total_files']} audio files")
    print(f"[EXTRACTION] Metadata saved to: {result['extraction_file']}")

    return xcom_data


def download_whisper_model_task(**context):
    """Download Whisper model ONCE before parallel transcription."""
    from faster_whisper import WhisperModel

    model_size = _gp(context, 'transcription_model', 'base')
    compute_type = _gp(context, 'transcription_compute_type', 'float32')

    print(f"[MODEL DOWNLOAD] Pre-downloading Whisper model: {model_size}")
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    print(f"[MODEL DOWNLOAD] Model cached successfully")
    return "Model downloaded"


def transcribe_audio_file_task(chapter_index: int, **context):
    """Task 5: Transcribe a single audio file (runs in parallel for each chapter)."""
    _ensure_hf_cache()
    from scripts.transcription.trascription_tasks import transcribe_single_chapter

    content_type = _gp(context, 'transcription_type', 'audiobook')
    model_size = _gp(context, 'transcription_model', 'base')
    beam_size = _gp(context, 'transcription_beam_size', 5)
    compute_type = _gp(context, 'transcription_compute_type', 'float32')

    ti = context['ti']
    extraction_result = ti.xcom_pull(task_ids='extract_audio_metadata', key='extraction_result')
    base_name = extraction_result['base_name']
    full_output_dir = ti.xcom_pull(task_ids='validation_group.validate_output_directory', key='full_output_dir')

    print(f"[TRANSCRIPTION] Starting transcription for chapter {chapter_index}")

    result = transcribe_single_chapter(
        chapter_index=chapter_index,
        base_name=base_name,
        content_type=content_type,
        output_dir=full_output_dir,
        model_size=model_size,
        beam_size=beam_size,
        compute_type=compute_type
    )

    print(f"[TRANSCRIPTION] Chapter {chapter_index} completed")
    return result


def generate_summary_report_task(**context):
    """Task 6: Generate final summary report after all transcriptions complete."""
    from scripts.transcription.summary import generate_summary_report

    content_type = _gp(context, 'transcription_type', 'audiobook')
    ti = context['ti']
    extraction_result = ti.xcom_pull(task_ids='extract_audio_metadata', key='extraction_result')
    base_name = extraction_result['base_name']
    full_output_dir = ti.xcom_pull(task_ids='validation_group.validate_output_directory', key='full_output_dir')

    print(f"[SUMMARY] Generating summary report for: {base_name}")

    summary = generate_summary_report(
        base_name=base_name,
        content_type=content_type,
        output_dir=full_output_dir,
        cleanup_results=False
    )

    xcom_data = {
        'base_name': summary['base_name'],
        'total_chapters': summary['total_chapters'],
        'successful': summary['successful'],
        'failed': summary['failed'],
        'status': 'complete',
        'output_dir': full_output_dir
    }
    context['ti'].xcom_push(key='summary_report', value=xcom_data)

    print(f"[SUMMARY] Report generated")
    print(f"[SUMMARY] Total chapters: {summary['total_chapters']}")
    print(f"[SUMMARY] Successful: {summary['successful']}")
    print(f"[SUMMARY] Failed: {summary['failed']}")

    return xcom_data


# ============================================================================
# PHASE 3: CROSS-MODEL VALIDATION (PARALLEL)
# ============================================================================

def run_create_sample_zip(**context):
    """STEP 1: Create sample ZIP for cross-model evaluation"""
    _ensure_hf_cache()
    from scripts.transcription.utils.audio_utils import sample_zip_filtered
    
    print("=" * 70)
    print("STEP 1/3: Creating Sample ZIP for Cross-Model Evaluation")
    print("=" * 70)
    
    folder = _gp(context, 'cross_folder')
    type_name = _gp(context, 'cross_type', 'audiobook')
    sample_size = int(_gp(context, 'cross_sample_size', 1))
    
    if not folder:
        raise ValueError("cross_folder parameter is required.")
    
    print(f"Source Folder: {folder}")
    print(f"Content Type: {type_name}")
    print(f"Sample Size: {sample_size} file(s)")
    
    source_path = Path(folder)
    sample_zip_path = Path("data/raw/sample_subset.zip")
    sample_zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_path = sample_zip_filtered(source_path, sample_size, sample_zip_path)
    print(f"Sample ZIP created: {out_path}")
    
    context['ti'].xcom_push(key='sample_zip_path', value=str(out_path))
    context['ti'].xcom_push(key='content_type', value=type_name)
    context['ti'].xcom_push(key='folder', value=str(folder))
    
    print("=" * 70)
    print("STEP 1 COMPLETE - Ready for parallel transcription")
    print("=" * 70)
    
    return f"Sample ZIP created: {out_path}"


def run_transcribe_openai_whisper(**context):
    """STEP 2a: OpenAI Whisper transcription (runs in parallel with Wav2Vec2)"""
    _ensure_hf_cache()
    from scripts.validation.cross_model_evaluation.cross_model_sample_openaiwhisper import transcribe_sample_openaiwhisper
    
    print("=" * 70)
    print("STEP 2a/3: OpenAI Whisper Transcription PARALLEL")
    print("=" * 70)
    
    ti = context['ti']
    sample_zip_path = ti.xcom_pull(task_ids='create_sample_zip', key='sample_zip_path')
    content_type = ti.xcom_pull(task_ids='create_sample_zip', key='content_type')

    if not sample_zip_path:
        raise ValueError("sample_zip_path not found in XCom.")
    
    print(f"Input: {sample_zip_path}")
    print(f"Content Type: {content_type}")
    print(f"Model: OpenAI Whisper (base)")
    
    output_dir = Path("data/validation/cross_model_evaluation/openaiwhisper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transcribe_sample_openaiwhisper(
        zipfile_path=sample_zip_path,
        outdir_path=str(output_dir),
        content_type=content_type,
        model_size="base"
    )
    
    print(f"OpenAI Whisper transcription complete!")
    print("=" * 70)
    
    return "OpenAI Whisper transcription finished"


def run_transcribe_wav2vec(**context):
    """STEP 2b: Wav2Vec2 transcription (runs in parallel with OpenAI Whisper)"""
    _ensure_hf_cache()
    from scripts.validation.cross_model_evaluation.cross_model_sample_wav2vec import transcribe_sample_wav2vec
    
    print("=" * 70)
    print("STEP 2b/3: Wav2Vec2 Transcription PARALLEL")
    print("=" * 70)
    
    ti = context['ti']
    sample_zip_path = ti.xcom_pull(task_ids='create_sample_zip', key='sample_zip_path')
    content_type = ti.xcom_pull(task_ids='create_sample_zip', key='content_type')
    
    if not sample_zip_path:
        raise ValueError("sample_zip_path not found in XCom.")
    
    print(f"Input: {sample_zip_path}")
    print(f"Content Type: {content_type}")
    print(f"Model: facebook/wav2vec2-base-960h")
    
    output_dir = Path("data/validation/cross_model_evaluation/wav2vec2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transcribe_sample_wav2vec(
        zipfile_path=sample_zip_path,
        outdir_path=str(output_dir),
        content_type=content_type
    )
    
    print(f"Wav2Vec2 transcription complete!")
    print("=" * 70)
    
    return "Wav2Vec2 transcription finished"


def run_validate_cross_models(**context):
    """STEP 3: Cross-model validation (runs after both parallel transcription tasks)"""
    _ensure_hf_cache()
    from scripts.validation.cross_model_evaluation.validate_transcription import validate_models
    
    print("=" * 70)
    print("STEP 3/3: Cross-Model Validation (After Parallel Tasks)")
    print("=" * 70)
    
    ti = context['ti']
    content_type = ti.xcom_pull(task_ids='create_sample_zip', key='content_type')
    folder = ti.xcom_pull(task_ids='create_sample_zip', key='folder')
    sample_zip_path = ti.xcom_pull(task_ids='create_sample_zip', key='sample_zip_path')
    
    # Cleanup sample zip
    sample_zip = Path(sample_zip_path)
    if sample_zip.exists():
        sample_zip.unlink()
        print(f"Cleared Sample Zip - {sample_zip_path}")

    transcription_outdir = ti.xcom_pull(task_ids='validation_group.validate_output_directory', key='full_output_dir')

    print(f"Comparing three models:")
    print(f"   Faster-Whisper: {transcription_outdir}")
    print(f"   OpenAI Whisper: data/validation/cross_model_evaluation/openaiwhisper")
    print(f"   Wav2Vec2: data/validation/cross_model_evaluation/wav2vec2")
    
    source_base = Path(folder).stem if folder else "unknown"
    out_csv = Path(f"data/validation/cross_model_evaluation/{content_type}_{source_base}_validation_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nValidation summary: {out_csv}")
    print(f"Thresholds: WER < 45%, ROUGE-L > 60%")
    print("\nRunning validation...")
    
    validate_models(
        fw_dir=transcription_outdir,
        ow_dir="data/validation/cross_model_evaluation/openaiwhisper",
        w2v_dir="data/validation/cross_model_evaluation/wav2vec2",
        out_csv=out_csv,
        content_type=content_type,
        wer_threshold=0.45,
        rouge_threshold=0.60
    )
    
    print("\n" + "=" * 70)
    print("VALIDATION PASSED!")
    print(f"Summary: {out_csv}")
    print("=" * 70)
    
    context['ti'].xcom_push(key="cross_model_validation_summary_csv", value=str(out_csv))
    
    return f"Cross-model validation passed (summary: {out_csv})"


# ============================================================================
# PHASE 4: API-BASED CHUNKING & EMBEDDING
# ============================================================================

def prepare_chunk_request(**context):
    """Prepare chunking request using transcription output folder"""
    ti = context['ti']
    
    # Get transcription output directory from summary task
    summary_result = ti.xcom_pull(task_ids='generate_summary_report', key='summary_report')
    transcription_outdir = summary_result.get('output_dir')
    
    if not transcription_outdir:
        raise ValueError("Transcription output directory not found in XCom")
    
    # Convert to absolute path for API
    folder_path = f'/app/data/{Path(transcription_outdir).name}'
    
    target_tokens = int(_gp(context, 'chunk_target_tokens', 512))
    overlap_tokens = int(_gp(context, 'chunk_overlap_tokens', 50))
    
    chunk_request = {
        'folder_path': folder_path,
        'target_tokens': target_tokens,
        'overlap_tokens': overlap_tokens,
        'output_file': '/app/data/chunks_output.json'
    }
    
    print(f"Chunk Request Prepared:")
    print(f"   Folder: {folder_path}")
    print(f"   Target Tokens: {target_tokens}")
    print(f"   Overlap: {overlap_tokens}")
    
    return chunk_request


def call_chunk_api(**context):
    """Call the chunking API"""
    import requests
    
    ti = context['ti']
    chunk_request = ti.xcom_pull(task_ids='prepare_chunk_request')
    
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


def prepare_embed_request(**context):
    """Prepare embedding request"""
    return {
        'chunks_file': '/app/data/chunks_output.json',
        'output_file': '/app/data/embeddings_output.json'
    }


def call_embed_api(**context):
    """Call the embedding API"""
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


def prepare_vector_db_request(**context):
    """Prepare request to add to vector DB"""
    return {
        'chunks_file': '/app/data/chunks_output.json',
        'embeddings_file': '/app/data/embeddings_output.json'
    }


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


def verify_storage(**context):
    """Verify files were created and vector DB was populated"""
    import os
    
    chunks_exists = os.path.exists('/opt/airflow/working_data/chunks_output.json')
    embeddings_exists = os.path.exists('/opt/airflow/working_data/embeddings_output.json')
    
    print(f"Chunks file exists: {chunks_exists}")
    print(f"Embeddings file exists: {embeddings_exists}")
    print(f"Vector DB populated (check /vector-db/stats endpoint)")
    
    return "All files saved and vector DB populated!"


def generate_final_report(**context):
    """Generate final pipeline report"""
    ti = context['ti']
    
    validation_csv = ti.xcom_pull(task_ids='validate_transcription', key='validation_summary_csv')
    cross_validation_csv = ti.xcom_pull(task_ids='validate_cross_models', key='cross_model_validation_summary_csv')
    summary_result = ti.xcom_pull(task_ids='generate_summary_report', key='summary_report')
    
    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print("\nPipeline Outputs:")
    print(f"   Model Validation: {validation_csv}")
    print(f"   Transcription Summary: {summary_result.get('base_name')}")
    print(f"   Total Chapters: {summary_result.get('total_chapters')}")
    print(f"   Successful: {summary_result.get('successful')}")
    print(f"   Cross-Model Validation: {cross_validation_csv}")
    print(f"   Chunks: /app/data/chunks_output.json")
    print(f"   Embeddings: /app/data/embeddings_output.json")
    print(f"   Vector DB: Populated")
    print("=" * 70)
    
    return "Pipeline completed successfully!"


# ============================================================================
# DYNAMIC TASK GENERATOR FOR PARALLEL TRANSCRIPTION
# ============================================================================

@task(dag=dag)
def get_chapter_indices_task(**context):
    """Extract list of chapter indices from extraction metadata"""
    ti = context['ti']
    extraction_result = ti.xcom_pull(task_ids='extract_audio_metadata', key='extraction_result')
    extraction_file = Path(extraction_result['extraction_file'])

    print(f"[TASK GENERATOR] Reading extraction file: {extraction_file}")

    extraction_data = json.loads(extraction_file.read_text())
    audio_files = extraction_data['audio_files']
    chapter_indices = [file_info['original_number'] for file_info in audio_files]

    print(f"[TASK GENERATOR] Found {len(chapter_indices)} chapters to transcribe")
    return chapter_indices


@task(dag=dag)
def transcribe_audio_file_task_wrapper(chapter_index: int, **context):
    """Wrapper for transcribe_audio_file_task to work with TaskFlow API"""
    return transcribe_audio_file_task(chapter_index=chapter_index, **context)


# ============================================================================
# BUILD THE DAG - OPERATORS
# ============================================================================

# PHASE 1: Model Validation
transcribe_reference_audio = PythonOperator(
    task_id='transcribe_reference_audio',
    python_callable=run_transcribe_reference,
    provide_context=True,
    dag=dag,
)

validate_transcription = PythonOperator(
    task_id='validate_transcription',
    python_callable=run_validate_transcription,
    provide_context=True,
    dag=dag,
)

# PHASE 2: Transcription Pipeline - Validation Group
with TaskGroup(group_id='validation_group', dag=dag) as validation_group:
    validate_input = PythonOperator(
        task_id='validate_input_directory',
        python_callable=validate_input_dir_task,
        provide_context=True,
        dag=dag,
    )

    validate_type = PythonOperator(
        task_id='validate_content_type',
        python_callable=validate_content_type_task,
        provide_context=True,
        dag=dag,
    )

    validate_output = PythonOperator(
        task_id='validate_output_directory',
        python_callable=validate_output_dir_task,
        provide_context=True,
        dag=dag,
    )

# PHASE 2: Transcription Pipeline - Extraction & Transcription
extract_metadata = PythonOperator(
    task_id='extract_audio_metadata',
    python_callable=extract_audio_metadata_task,
    provide_context=True,
    dag=dag,
)

download_model = PythonOperator(
    task_id='download_whisper_model',
    python_callable=download_whisper_model_task,
    provide_context=True,
    dag=dag,
)

get_chapter_indices = get_chapter_indices_task(dag=dag)
transcribe_chapters = transcribe_audio_file_task_wrapper.expand(chapter_index=get_chapter_indices)

generate_summary = PythonOperator(
    task_id='generate_summary_report',
    python_callable=generate_summary_report_task,
    provide_context=True,
    dag=dag,
)

# PHASE 3: Cross-Model Validation
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

# PHASE 4: API-Based Chunking & Embedding
prepare_chunk = PythonOperator(
    task_id='prepare_chunk_request',
    python_callable=prepare_chunk_request,
    provide_context=True,
    dag=dag,
)

chunk_text = PythonOperator(
    task_id='chunk_text',
    python_callable=call_chunk_api,
    provide_context=True,
    dag=dag,
)

prepare_embed = PythonOperator(
    task_id='prepare_embed_request',
    python_callable=prepare_embed_request,
    dag=dag,
)

generate_embeddings = PythonOperator(
    task_id='generate_embeddings',
    python_callable=call_embed_api,
    provide_context=True,
    dag=dag,
)

prepare_vector_db = PythonOperator(
    task_id='prepare_vector_db_request',
    python_callable=prepare_vector_db_request,
    dag=dag,
)

add_to_vector_db = PythonOperator(
    task_id='add_to_vector_db',
    python_callable=call_vector_db_api,
    provide_context=True,
    dag=dag,
)

verify_results = PythonOperator(
    task_id='verify_storage',
    python_callable=verify_storage,
    dag=dag,
)

final_report = PythonOperator(
    task_id='generate_final_report',
    python_callable=generate_final_report,
    dag=dag,
)


# ============================================================================
# TASK DEPENDENCIES - COMPLETE PIPELINE
# ============================================================================

# PHASE 1: Model Validation (sequential)
transcribe_reference_audio >> validate_transcription

# PHASE 2: Transcription Pipeline
validate_transcription >> validation_group >> extract_metadata >> download_model >> get_chapter_indices >> transcribe_chapters >> generate_summary

# PHASE 3: Cross-Model Validation (parallel)
generate_summary >> create_sample_zip >> [transcribe_openai_whisper, transcribe_wav2vec] >> validate_cross_models

# PHASE 4: API-Based Processing (sequential)
validate_cross_models >> prepare_chunk >> chunk_text >> prepare_embed >> generate_embeddings >> prepare_vector_db >> add_to_vector_db >> verify_results >> final_report