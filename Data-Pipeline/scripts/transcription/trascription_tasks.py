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
    """
    logger.info(f"Starting transcription for Chapter {chapter_index}: {base_name}")
    logger.info("Initializing file and metadata checks...")

    metadata_dir = Path(output_dir) / "transcription_metadata"
    extraction_file = metadata_dir / f"{base_name}_extraction.json"

    if not extraction_file.exists():
        logger.error(f"Extraction metadata not found : {extraction_file}")
        raise FileNotFoundError(f"Extraction metadata not found : {extraction_file}")

    try:
        logger.info(f"Reading extraction metadata → {extraction_file}")
        extraction_data = json.loads(extraction_file.read_text())
        audio_files = extraction_data['audio_files']
        logger.info(f"Loaded {len(audio_files)} audio entries from metadata.")
        logger.info(f"Searching for chapter index: {chapter_index}")

        audio_file_info = None
        for file_info in audio_files:
            if int(file_info['original_number']) == int(chapter_index):
                audio_file_info = file_info
                break

        if not audio_file_info:
            logger.warning(f"No audio file with chapter index {chapter_index}")
            return {"status": "skipped", "chapter_index": chapter_index}

    except Exception as e:
        logger.error(f"Could not read extraction metadata: {str(e)}")
        raise

    audio_path = Path(audio_file_info['path'])
    original_filename = audio_file_info['filename']
    original_number = audio_file_info.get('original_number')
    chapter_num = original_number if original_number is not None else chapter_index

    logger.info(f"Processing: {original_filename}")
    logger.info(f"Chapter/Episode number: {chapter_num}")
    logger.info(f"Audio file size: {audio_file_info['size_mb']:.2f} MB")

    try:
        from faster_whisper import WhisperModel

        device = detect_device()
        logger.info(f"Using device: {device}")
        logger.info(f"Loading Faster-Whisper model: {model_size}")

        start_time = time.time()
        kwargs = {"device": device}
        if compute_type:
            kwargs["compute_type"] = compute_type

        model = WhisperModel(model_size, **kwargs)
        model_load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {model_load_time:.2f} seconds")
        logger.info(f"Starting transcription for {original_filename}...")

        transcribe_start = time.time()
        segments, info = model.transcribe(str(audio_path), beam_size=beam_size)

        total_duration = getattr(info, "duration", None)
        if total_duration:
            logger.info(f"Detected duration: {total_duration:.2f}s")
        else:
            logger.info("Duration not available from model output.")

        lines = []
        last_pct_logged = -5
        for idx, s in enumerate(segments, start=1):
            lines.append(f"[{s.start:.2f}-{s.end:.2f}] {s.text}")
            # progress logging
            if total_duration and total_duration > 0:
                pct = int(min(100, (s.end / total_duration) * 100))
                if pct >= last_pct_logged + 5:
                    logger.info(
                        f"[PROGRESS] Chapter {chapter_num}: {pct}% "
                        f"({s.end:.1f}s / {total_duration:.1f}s) | "
                        f"'{s.text.strip()[:60]}...'"
                    )
                    last_pct_logged = pct
            elif idx % 10 == 0:
                logger.info(f"[PROGRESS] Processed {idx} segments so far...")

        transcribe_time = time.time() - transcribe_start
        logger.info(f"Transcription completed in {transcribe_time:.2f} seconds")
        logger.info(f"Generated {len(lines)} segments total")
        logger.info(f"Detected language: {getattr(info, 'language', 'unknown')}")

        # Save transcript
        logger.info("Saving transcript to output directory...")
        if content_type.lower() == "audiobook":
            output_filename = f"audiobook_{base_name}_chapter_{chapter_num:02d}.txt"
        else:
            output_filename = f"podcast_{base_name}_episode_{chapter_num:02d}.txt"

        output_path = Path(output_dir) / output_filename
        save_lines(output_path, lines)
        logger.info(f"[OK] Transcript saved → {output_path}")

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
            logger.info(f" Chapter {chapter_num} transcription successful")
        else:
            logger.info(f" Episode {chapter_num} transcription successful")

        # Save results metadata
        results_dir = Path(output_dir) / "results_metadata"
        results_dir.mkdir(exist_ok=True, parents=True)

        if content_type.lower() == "audiobook":
            result_file = results_dir / f"{base_name}_chapter_{chapter_num:02d}_result.json"
        else:
            result_file = results_dir / f"{base_name}_episode_{chapter_num:02d}_result.json"

        result_file.write_text(json.dumps(result, indent=2))
        logger.info(f"[OK] Result metadata saved → {result_file}")

        logger.info(f"Total runtime for Chapter {chapter_num}: {total_runtime:.2f} seconds")
        logger.info(f"CPU: {result['cpu_usage_percent']}% | Memory: {result['memory_usage_percent']}%")

        return {
            "chapter_index": chapter_index,
            "chapter_num": chapter_num,
            "status": "success",
            "result_file": str(result_file)
        }

    except Exception as e:
        logger.error(f" Error transcribing chapter {chapter_num}: {str(e)}", exc_info=True)

        error_result = {
            "chapter_index": chapter_index,
            "chapter_num": chapter_num,
            "audio_file": original_filename,
            "status": "failed",
            "error": str(e)
        }

        results_dir = Path(output_dir) / "results_metadata"
        results_dir.mkdir(exist_ok=True, parents=True)

        if content_type.lower() == "audiobook":
            result_file = results_dir / f"{base_name}_chapter_{chapter_index:02d}_error.json"
        else:
            result_file = results_dir / f"{base_name}_episode_{chapter_index:02d}_error.json"

        result_file.write_text(json.dumps(error_result, indent=2))
        logger.error(f"Error metadata saved to → {result_file}")
        raise


if __name__ == "__main__":
    print("This is the script for zip extraction of raw audio files.")

    parser = argparse.ArgumentParser(description="Extraction Files and Folders")
    parser.add_argument("--chapter_index", required=True, default="1",
                        help="transcription_chapter_index")
    parser.add_argument("--input_dir", default="Data-Pipeline/data/raw/audios.zip",
                        help="Path to input directory containing audio files")
    parser.add_argument("--type", required=True,
                        help="Type of content: audiobook or podcast (used in file naming)")
    parser.add_argument("--outdir", default="Data-Pipeline/data/transcription_results",
                        help="Output directory for transcripts and summary CSV")

    args = parser.parse_args()

    base_name = str(Path(args.input_dir).stem.lower())
    transcribe_single_chapter(args.chapter_index, base_name, args.type, args.outdir)
