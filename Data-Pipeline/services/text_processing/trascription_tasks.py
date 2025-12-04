import argparse
import json
import logging
import sys
import time
from pathlib import Path
import psutil


def save_lines(path: Path, lines):
    path.write_text("\n".join(lines), encoding="utf-8")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def detect_device():
    return "cpu"


def transcribe_single_chapter(
    chapter_index: int,
    base_name: str,
    audio_dir: str,
    transcript_dir: str,
    content_type: str,
    model_size: str = "base",
    beam_size: int = 5,
    compute_type: str = "float32",
):
    audio_dir = Path(audio_dir)
    transcript_dir = Path(transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # -------- FIND AUDIO FILES --------
    audio_files = sorted([
        p for p in audio_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in [".mp3", ".wav"]
        and not p.name.startswith(".")
        and not p.name.startswith("._")
        and p.stat().st_size > 0
    ])

    if not audio_files:
        raise FileNotFoundError(f"No valid audio files found in {audio_dir}")

    try:
        audio_path = audio_files[chapter_index - 1]
    except IndexError:
        raise FileNotFoundError(
            f"Requested chapter {chapter_index} but found only {len(audio_files)} audio files."
        )

    logger.info(f"Transcribing: {audio_path.name}")

    try:
        from faster_whisper import WhisperModel

        device = detect_device()
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        start_time = time.time()
        segments, info = model.transcribe(str(audio_path), beam_size=beam_size)

        lines = [f"[{s.start:.2f}-{s.end:.2f}] {s.text}" for s in segments]
        transcribe_time = time.time() - start_time

        # -------- SAVE TRANSCRIPT --------
        output_filename = (
            f"audiobook_{base_name}_chapter_{chapter_index:02d}.txt"
            if content_type.lower() == "audiobook"
            else f"podcast_{base_name}_episode_{chapter_index:02d}.txt"
        )

        output_path = transcript_dir / output_filename
        save_lines(output_path, lines)

        # -------- SAVE METADATA --------
        results_dir = transcript_dir / "results_metadata"
        results_dir.mkdir(exist_ok=True)

        result_file = results_dir / f"{base_name}_chapter_{chapter_index:02d}_result.json"

        result_data = {
            "chapter_index": chapter_index,
            "audio_file": audio_path.name,
            "transcript_file": output_filename,
            "segments": len(lines),
            "runtime_seconds": round(transcribe_time, 2),
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "status": "success",
        }

        result_file.write_text(json.dumps(result_data, indent=2))

        return {"chapter_index": chapter_index, "transcript_file": output_filename}

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise
