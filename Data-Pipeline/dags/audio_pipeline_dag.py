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
        'transcription_inputdir': 'data/raw/edison_lifeinventions',
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

def validate_input_dir_task(**context):
   """
    Task 1: Validate that input directory exists and is accessible.

    This task:
    - Checks if input directory exists
    - Verifies it's actually a directory (not a file)
    - Checks read permissions

    Returns: Validation result dictionary
    Raises: Exception if validation fails
    """

   from scripts.transcription.transcription_file_checks import validate_input_directory
   input_dir = _gp(context, 'transcription_inputdir')

   print(f" VALIDATION : Checking input directory : {input_dir}")

   result = validate_input_directory(input_dir)

   context['ti'].xcom_push(key = 'input_dir_validation', value=result)

   context['ti'].xcom_push(key = 'input_dir', value = input_dir)

   print(f"VALIDATION : Input Directory is valid")
   return result


def validate_output_dir_task(**context):
    """
     Task 2: Validate that output directory exists or can be created.

     This task:
     - Gets input_dir from XCom (saved by validate_input_dir_task)
     - Extracts base name from input directory (e.g., "edison_lifeinventions")
     - Constructs full output path: output_dir/base_name
     - Creates output directory if it doesn't exist
     - Verifies write permissions

     Returns: Validation result dictionary
     Raises: Exception if validation fails
     """

    from scripts.transcription.transcription_file_checks import validate_output_directory
    output_dir_base = 'data/transcription_results'

    ti = context['ti']
    input_dir = ti.xcom_pull(task_ids = 'validation_group.validate_input_directory', key = 'input_dir')

    base_name = Path(input_dir).stem.lower()

    full_output_dir = str(Path(output_dir_base) / base_name)

    print(f"VALIDATION : Input Directory (From XCom) : {input_dir}")
    print(f"VALIDATION : Base Name Extracted : {base_name}")
    print(f"VALIDATION : Full output directory : {full_output_dir}")

    result = validate_output_directory(full_output_dir)
    # Save both base output and full output to XCom for other tasks
    context['ti'].xcom_push(key='output_dir_validation', value=result)
    context['ti'].xcom_push(key='base_name', value=base_name)
    context['ti'].xcom_push(key='full_output_dir', value=full_output_dir)

    print(f"VALIDATION :  Output directory is valid")
    return result


def validate_content_type_task(**context):
    """
    Task 3: Validate that content type is 'audiobook' or 'podcast'.

    This task:
    - Checks content_type parameter
    - Determines split_type (chapter/episode)

    Returns: Validation result dictionary
    Raises: ValueError if content type is invalid
    """
    from scripts.transcription.transcription_file_checks import validate_content_type

    content_type = _gp(context, 'transcription_type', 'audiobook')

    print(f"VALIDATION : Checking content type : {content_type}")

    result = validate_content_type(content_type)

    context['ti'].xcom_push(key = 'content_type_validation', value = result)

    print(f"VALIDATION : Content type is valid : {result['split_type']}")

    return result


# ============================================================================
# TASK FUNCTIONS - EXTRACTION
# ============================================================================

def extract_audio_metadata_task(**context):
    """
    Task 4: Extract metadata from all audio files in input directory.

    This task:
    - Scans input directory for audio files
    - Extracts chapter/episode numbers from filenames
    - Saves metadata to JSON file (MAIN DATA STORAGE)
    - Returns minimal metadata to XCom (just paths and counts)

    Returns: Extraction result with file list
    Raises: Exception if no audio files found
    """
    from scripts.transcription.extraction_tasks import extract_and_list_audio_files

    # Get parameters
    input_dir = _gp(context, 'transcription_inputdir')
    content_type = _gp(context, 'transcription_type', 'audiobook')

    ti = context['ti']
    full_output_dir = ti.xcom_pull(task_ids='validate_output_directory', key='full_output_dir')

    print(f"[EXTRACTION] Input directory: {input_dir}")
    print(f"[EXTRACTION] Output directory: {full_output_dir}")
    print(f"[EXTRACTION] Scanning audio files...")

    result = extract_and_list_audio_files(input_dir = input_dir, output_dir= full_output_dir, audio_type= content_type)

    xcom_data = {
        'base_name': result['base_name'],
        'total_files': result['total_files'],
        'extraction_file': result['extraction_file'],  # ← Path to the real data
        'status': result['status']
    }
    context['ti'].xcom_push(key='extraction_result', value=xcom_data)

    print(f"[EXTRACTION] Found {result['total_files']} audio files ✓")
    print(f"[EXTRACTION] Metadata saved to: {result['extraction_file']}")
    print(f"[EXTRACTION] XCom size: ~{len(str(xcom_data))} bytes (minimal) ✓")

    return xcom_data

# ============================================================================
# TASK FUNCTIONS - TRANSCRIPTION (PARALLEL)
# ============================================================================

def download_whisper_model_task(**context):
    """
    Download Whisper model ONCE before parallel transcription.
    This prevents race conditions when multiple tasks try to download simultaneously.
    """
    from faster_whisper import WhisperModel

    model_size = _gp(context, 'model_size', 'base')
    compute_type = _gp(context, 'compute_type', 'float32')

    print(f"[MODEL DOWNLOAD] Pre-downloading Whisper model: {model_size}")
    print(f"[MODEL DOWNLOAD] This prevents parallel download conflicts...")

    # Download model (will be cached)
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    print(f"[MODEL DOWNLOAD] Model cached successfully ✓")
    print(f"[MODEL DOWNLOAD] Parallel transcription tasks can now use cached model")

    return "Model downloaded"




def transcribe_audio_file_task(chapter_index: int, **context):
    """
    Task 5: Transcribe a single audio file.

    This task is created DYNAMICALLY for each audio file.
    Multiple instances run IN PARALLEL.

     DATA FLOW:
    1. Pull extraction_file PATH from XCom (small)
    2. Read actual metadata from JSON file (large data)
    3. Transcribe audio
    4. Save result to JSON file (not XCom)
    5. Return minimal status to XCom

    Args:
        chapter_index: The chapter/episode number to transcribe

    Returns: Transcription result (minimal)
    Raises: Exception if transcription fails
    """

    _ensure_hf_cache()
    from scripts.transcription.trascription_tasks import transcribe_single_chapter


    # Get Parameters
    content_type = _gp(context, 'transcription_type', 'audiobook')
    model_size = _gp(context, 'transcription_model', 'base')
    beam_size = _gp(context, 'transcription_beam_size', 5)
    compute_type = _gp(context, 'transcription_compute_type', 'float32')

    ti = context['ti']
    extraction_result = ti.xcom_pull(task_ids = 'extract_audio_metadata', key = 'extraction_result')
    base_name = extraction_result['base_name']

    full_output_dir = ti.xcom_pull(task_ids='validate_output_directory', key='full_output_dir')

    print(f"[TRANSCRIPTION] Starting transcription for chapter {chapter_index}")
    print(f"[TRANSCRIPTION] Output directory: {full_output_dir}")


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
    print(f"[TRANSCRIPTION] Result saved to: {result.get('result_file', 'file')}")

    return result


# ============================================================================
# TASK FUNCTIONS - SUMMARY
# ============================================================================

def generate_summary_report_task(**context):
    """
    Task 6: Generate final summary report after all transcriptions complete.

    This task:
    - Reads all transcription result files from disk (not XCom)
    - Generates summary CSV
    - Cleans up metadata files (optional)

    DATA FLOW:
    1. Pull base_name from XCom (small)
    2. Read all result JSON files from disk (large data)
    3. Generate summary CSV
    4. Save minimal stats to XCom

    Returns: Summary statistics (minimal)
    Raises: Exception if no results found
    """
    from scripts.transcription.summary import generate_summary_report

    # Get parameters
    content_type = _gp(context, 'transcription_type', 'audiobook')


    ti = context['ti']
    extraction_result = ti.xcom_pull(task_ids='extract_audio_metadata', key='extraction_result')
    base_name = extraction_result['base_name']

    full_output_dir = ti.xcom_pull(task_ids='validate_output_directory', key='full_output_dir')

    print(f"[SUMMARY] Generating summary report for: {base_name}")
    print(f"[SUMMARY] Output directory: {full_output_dir}")


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
        'status': 'complete'
    }
    context['ti'].xcom_push(key='summary_report', value=xcom_data)

    print(f"[SUMMARY] Report generated ✓")
    print(f"[SUMMARY] Total chapters: {summary['total_chapters']}")
    print(f"[SUMMARY] Successful: {summary['successful']}")
    print(f"[SUMMARY] Failed: {summary['failed']}")
    print(f"[SUMMARY] XCom size: ~{len(str(xcom_data))} bytes (minimal) ✓")

    return xcom_data

# ============================================================================
# DYNAMIC TASK GENERATOR
# ============================================================================

def create_transcription_tasks(parent_group, extraction_result):
    """
    Dynamically create transcription tasks based on audio file count.

    This function creates one PythonOperator for each audio file.
    All tasks will run IN PARALLEL (as much as resources allow).

    Args:
        parent_group: The TaskGroup to add tasks to
        extraction_result: Result from extraction task (contains file list)

    Returns:
        List of created task operators
    """


    tasks = []

    # Read the extraction metadata file
    extraction_file = Path(extraction_result['extraction_file'])
    extraction_data = json.loads(extraction_file.read_text())
    audio_files = extraction_data['audio_files']

    print(f"TASK GENERATOR : Creatuing {len(audio_files)} transcription tasks")

    # Create one task for each audio file
    for file_info in audio_files:
        chapter_num = file_info['original_number']

        task = PythonOperator(
            task_id = f'transcribe_chapter_{chapter_num:02d}',
            python_callable = transcribe_audio_file_task,
            op_kwargs = {'chapter_index': chapter_num},
            provide_context = True,
            dag = dag,
        )

        tasks.append(task)
    print(f"TASK GENERATOR : Created {len(tasks)} parallel transcription tasks")
    return tasks


# ============================================================================
# BUILD THE DAG - TASK GROUPS & DEPENDENCIES
# ============================================================================
@task(dag = dag)
def get_chapter_indices_task(**context):
    """
    Helper task: Extract list of chapter indices from extraction metadata.

    This task runs after extraction and returns the list of chapter numbers
    to transcribe. This list is then used to dynamically create parallel
    transcription tasks.

    Returns: List of chapter indices [1, 2, 3, ...]
    """
    ti = context['ti']

    # Get extraction result from XCom
    extraction_result = ti.xcom_pull(task_ids='extract_audio_metadata', key='extraction_result')
    extraction_file = Path(extraction_result['extraction_file'])

    print(f"[TASK GENERATOR] Reading extraction file: {extraction_file}")

    # Read the full extraction metadata from file
    extraction_data = json.loads(extraction_file.read_text())
    audio_files = extraction_data['audio_files']

    # Extract chapter indices
    chapter_indices = [file_info['original_number'] for file_info in audio_files]

    print(f"[TASK GENERATOR] Found {len(chapter_indices)} chapters to transcribe")
    print(f"[TASK GENERATOR] Chapter indices: {chapter_indices}")

    return chapter_indices


@task(dag = dag)
def transcribe_audio_file_task_wrapper(chapter_index: int, **context):
    """
    Wrapper for transcribe_audio_file_task to work with TaskFlow API.

    Args:
        chapter_index: The chapter/episode number to transcribe

    Returns: Transcription result (minimal)
    """
    # Call the original function
    return transcribe_audio_file_task(chapter_index=chapter_index, **context)


# TASK Group 1 : VALIDATION (3 tasks run in parallel)
with TaskGroup(group_id='validation_group', dag=dag) as validation_group:
    """
    This group contains all validation tasks.
    All 3 tasks can run IN PARALLEL since they don't depend on each other.
    """

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
    # No dependencies inside validation group (all run in parallel)

validate_output = PythonOperator(
        task_id='validate_output_directory',
        python_callable=validate_output_dir_task,
        provide_context=True,
        dag=dag,
    )
# TASK 2: EXTRACTION (single task)
extract_metadata = PythonOperator(
    task_id='extract_audio_metadata',
    python_callable=extract_audio_metadata_task,
    provide_context=True,
    dag=dag,
)

# Create the task operator
download_model = PythonOperator(
    task_id='download_whisper_model',
    python_callable=download_whisper_model_task,
    provide_context=True,
    dag=dag,
)

# TASK GROUP 3: TRANSCRIPTION (dynamic parallel tasks)
# get_chapter_indices = PythonOperator(
#     task_id = 'get_chapter_indices',
#     python_callable = get_chapter_indices_task,
#     provide_context = True,
#     dag = dag,
# )
get_chapter_indices = get_chapter_indices_task(dag = dag)

# This uses .expand() to dynamically create one task per chapter
# transcribe_chapters = PythonOperator(
#     task_id = "transcribe_chapter",
#     python_callable = transcribe_audio_file_task,
#     provide_context = True,
#     dag = dag,
# ).expand(op_kwargs= [{'chapter_index': idx} for  idx in get_chapter_indices.output])

transcribe_chapters = transcribe_audio_file_task_wrapper.expand(chapter_index=get_chapter_indices)


# TASK 4: SUMMARY (single task)
generate_summary = PythonOperator(
    task_id='generate_summary_report',
    python_callable=generate_summary_report_task,
    provide_context=True,
    dag=dag,
)



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
    transcription_outdir = ti.xcom_pull(task_ids='validate_output_directory', key='full_output_dir')

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
(transcribe_reference_audio >> validate_transcription >> \


# Step 1: All validation tasks must complete before extraction
validation_group >> validate_output >> extract_metadata >> \

# Step 2: Extract metadata, then get chapter indices
# Step 3: Dynamically create transcription tasks (one per chapter, all parallel)
 download_model >> get_chapter_indices >> transcribe_chapters >>\

# Step 4: After ALL transcriptions complete, generate summary
 generate_summary >> create_sample_zip)

# PARALLEL BRANCHING: Both models transcribe simultaneously
create_sample_zip >> [transcribe_openai_whisper, transcribe_wav2vec]

# CONVERGENCE: Validation waits for BOTH parallel tasks
[transcribe_openai_whisper, transcribe_wav2vec] >> validate_cross_models

# Continue pipeline
validate_cross_models >> chunk_text >> generate_embeddings