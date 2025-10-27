"""
transcription_tasks.py
Core transcription functions using Faster-Whisper
"""
import sys
import argparse
import logging
import time
import json
from pathlib import Path
import psutil


from scripts.transcription.utils.audio_utils import save_lines

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for even more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output to console
    ]
)

logger = logging.getLogger(__name__)

def detect_device():
    """Detect GPU or CPU device automatically"""
    return "cpu"


def transcribe_single_chapter(
        chapter_index: int,
        base_name: str,
        content_type: str,
        output_dir: str,
        model_size: str = "base",
        beam_size: int = 5,
        compute_type: str = "float32",
):
    """
    Transcribe a single audio chapter.

    Args:
        chapter_index : Index of the chapter (1-based)
        base_name: Base name of the original input directory
        content_type: 'audiobook' or 'podcast'
        output_dir: Directory to save transcripts
        model_size: Faster-Whisper model size
        beam_size: Beam size for decoding
        compute_type: Compute type for model

    Returns:
        dict with transcription results
    """
    logger.info(f"Starting transcription for Chapter {chapter_index}: {base_name}")

    metadata_dir = Path(output_dir) / "transcription_metadata"
    extraction_file = metadata_dir / f"{base_name}_extraction.json"

    if not extraction_file.exists():
        logger.error(f"Extraction metadata not found : {extraction_file}")
        raise FileNotFoundError(f"Extraction metadata not found : {extraction_file}")

    # If file exists
    try:
        logger.info("File exists!")
        extraction_data = json.loads(extraction_file.read_text())
        audio_files = extraction_data['audio_files']
        logger.info(f"Audio Files : {audio_files}")

        audio_file_info = None
        logger.info(f"Searching for index : {chapter_index}")
        for file_info in audio_files:
            logger.info(f"Chapter : {file_info['original_number']}")
            if int(file_info['original_number']) == int(chapter_index):
                audio_file_info = file_info   # found a match with chapter
                break

        if not audio_file_info:
            logger.warning(f"No audio file with chapter index {chapter_index}")
            return {"status": "skipped", "chapter_index": chapter_index}

    except Exception as e:
        logger.error(f"Could not read extraction metadata: {str(e)}")
        raise

    audio_path = Path(audio_file_info['path'])
    original_filename = audio_file_info['filename']

    #Get original chapter/episode number
    original_number = audio_file_info.get('original_number')
    chapter_num = original_number if original_number is not None else chapter_index

    logger.info(f"Processing: {original_filename}")
    logger.info(f"Chapter/Episode number : {chapter_num}")
    logger.info(f"Audio file size: {audio_file_info['size_mb']: .2f} MB")

    # Transcribing the content
    try:
        # Importing here to avoid loading model during DAG parsing

        from faster_whisper import WhisperModel

        device = detect_device()
        logger.info(f"Using device : {device}")


        # Initialize model
        logger.info(f"Loading Faster-Whisper model : {model_size}")
        start_time = time.time()

        kwargs = {"device" : device}
        if compute_type:
            kwargs["compute_type"] = compute_type

        model = WhisperModel(model_size,  **kwargs)
        model_load_time = time.time() - start_time
        logger.info(f"Model loaded in {model_load_time:.2f} seconds")

        # Transcribe
        logger.info(f"Starting Transcription process....")
        transcribe_start = time.time()
        segments, info = model.transcribe(str(audio_path), beam_size = beam_size)

        # Collect segments
        lines = [f"[{s.start:.2f}-{s.end:.2f}] {s.text}" for s in segments]
        transcribe_time = time.time() - transcribe_start

        logger.info(f"Transcription completed in {transcribe_time:.2f} seconds")
        logger.info(f"Generated {len(lines)} segments")
        logger.info(f"Detected language : {getattr(info, 'language', 'unknown')}")

        # Save transcript
        if content_type.lower() == "audiobook":
            output_filename = f"audiobook_{base_name}_chapter_{chapter_num:02d}.txt"
        else:
            output_filename = f"podcast_{base_name}_episode_{chapter_num:02d}.txt"

        output_path = Path(output_dir) / output_filename
        save_lines(output_path, lines)
        logger.info(f"Transcript saved : {output_path}")

        # Collect metrics
        total_runtime = time.time() - start_time
        result = {
            "chapter_index": chapter_index,
            "chapter_num": chapter_num,
            "audio_file": original_filename,
            "transcript_file": output_filename,
            "model": f"Faster-Whisper({model_size})",
            "runtime_seconds": round(total_runtime, 2),
            "transcribe_time_seconds": round(transcribe_time, 2),
            "language": getattr(info, "language", "unknown"),
            "duration_seconds": getattr(info, "duration", None),
            "segments": len(lines),
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "status": "success"
        }

        if content_type.lower() == "audiobook":
            logger.info(f"Chapter {chapter_num} transcription successful")
        else:
            logger.info(f"Episode {chapter_num} transcription successful")

        # Save results to json
        results_dir = Path(output_dir) / "results_metadata"
        results_dir.mkdir(exist_ok = True, parents = True)

        if content_type.lower() == "audiobook":
            result_file = results_dir / f"{base_name}_chapter_{chapter_num:02d}_result.json"
        else:
            result_file = results_dir / f"{base_name}_episode_{chapter_num:02d}_result.json"

        result_file.write_text(json.dumps(result, indent = 2))
        logger.info(f"Result metadata saved to : {result_file}")

        # return minimal data (status)
        return {
            "chapter_index": chapter_index,
            "chapter_num": chapter_num,
            "status": "success",
            "result_file": str(result_file)
        }

    except Exception as e:
        logger.error(f"Error transcribing chapter {chapter_num}: {str(e)}", exc_info=True)
        # Save error result to file instead of XCom
        error_result = {
            "chapter_index": chapter_index,
            "chapter_num": chapter_num,
            "audio_file": original_filename,
            "status": "failed",
            "error": str(e)
        }

        # Save error to JSON file for later summary
        results_dir = Path(output_dir)  / "results_metadata"
        results_dir.mkdir(exist_ok=True, parents=True)

        if content_type.lower() == "audiobook":
            result_file = results_dir / f"{base_name}_chapter_{chapter_index:02d}_error.json"
        else:
            result_file = results_dir / f"{base_name}_episode_{chapter_index:02d}_error.json"

        result_file.write_text(json.dumps(error_result, indent=2))

        # Re-raise to trigger Airflow retry logic
        raise

if __name__ == "__main__":
    print("This is the script for zip extraction of raw audio files.")

    parser = argparse.ArgumentParser(description="Extraction Files and Folders")
    parser.add_argument("--chapter_index", required = True, default = "1",
                        help = "transcription_chapter_index")
    parser.add_argument("--input_dir", default="Data-Pipeline/data/raw/audios.zip",
                        help="Path to input directory containing audio files")
    parser.add_argument("--type", required=True,
                        help="Type of content: audiobook or podcast (used in file naming)")
    parser.add_argument("--outdir", default="Data-Pipeline/data/transcription_results",
                        help="Output directory for transcripts and summary CSV")

    args = parser.parse_args()

    base_name = str(Path(args.input_dir).stem.lower())

    transcribe_single_chapter(args.chapter_index, base_name, args.type, args.outdir)
