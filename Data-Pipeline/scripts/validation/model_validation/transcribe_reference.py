#!/usr/bin/env python3
"""
Transcribe Reference Audio with Faster-Whisper
-----------------------------------------------
- Takes a single audio file or folder containing audio files.
- Transcribes audio using Faster-Whisper model.
- Saves the transcription to a text file for later validation.
"""

import argparse
import sys
import time
from pathlib import Path

import psutil

# Add Data-Pipeline directory to path
script_dir = Path(__file__).resolve().parent
data_pipeline_dir = script_dir.parent.parent.parent
sys.path.insert(0, str(data_pipeline_dir))


# ----------------------------
# ASR
# ----------------------------
def transcribe_faster_whisper(audio_path: Path, model_size="base", device="cpu", beam_size=5, compute_type="float32"):
    """Transcribe a single audio file using Faster-Whisper with progress logs."""
    from faster_whisper import WhisperModel

    print(f"[INFO] Loading Faster-Whisper model: {model_size}", flush=True)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"[INFO] Transcribing {audio_path.name}...", flush=True)
    start = time.time()

    # Get streaming segments + info (contains duration, language)
    segments, info = model.transcribe(str(audio_path), beam_size=beam_size)

    total_duration = getattr(info, "duration", None)  # seconds
    text_pieces = []

    # Log every ~5% (tune this if you want more/less frequent logs)
    last_pct_logged = -5

    for seg in segments:
        # seg has .start, .end, .text, etc.
        text_pieces.append(seg.text)

        if total_duration and total_duration > 0:
            pct = int(min(100, (seg.end / total_duration) * 100))
            if pct >= last_pct_logged + 5:
                print(
                    f"[PROGRESS] {audio_path.name}: {pct}% "
                    f"({seg.end:.1f}s / {total_duration:.1f}s) | "
                    f"chunk='{seg.text.strip()[:60]}...'",
                    flush=True
                )
                last_pct_logged = pct
        else:
            # Fallback if duration isn't available
            print(
                f"[PROGRESS] {audio_path.name}: [{seg.start:.1f}s → {seg.end:.1f}s] "
                f"chunk='{seg.text.strip()[:60]}...'",
                flush=True
            )

    text = " ".join(text_pieces).strip()
    runtime = round(time.time() - start, 2)
    lang = getattr(info, "language", "unknown")

    return text, runtime, lang


def detect_device():
    """Detect available device for inference."""
    return "cpu"


# ----------------------------
# Transcribe from Folder or Single File
# ----------------------------
def transcribe_audio(
        audio_path: Path,
        out_transcript: Path,
        model_size="base",
        beam_size=5,
        compute_type="float32",
):
    """
    Transcribe audio file(s) and save combined transcript.
    Supports single audio files or folders containing audio files.
    """

    # Ensure output directory exists
    out_transcript.parent.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] CPU Usage: {psutil.cpu_percent()}%")
    print(f"[INFO] Memory Usage: {psutil.virtual_memory().percent}%")

    combined_transcript = []
    total_runtime = 0.0
    language = "unknown"

    # Check if input is a directory (folder)
    if audio_path.is_dir():
        print(f"[INFO] Processing audio files from folder: {audio_path}")

        # Get all audio files from the folder
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac'}
        audio_files = sorted([
            f for f in audio_path.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        ], key=lambda x: x.name)

        if not audio_files:
            sys.exit(f"[ERROR] No valid audio files found in folder: {audio_path}")

        print(f"[INFO] Found {len(audio_files)} valid audio files to process.")

        for audio_file in audio_files:
            print(f"\n▶ Transcribing {audio_file.name}...")
            try:
                hyp, runtime, lang = transcribe_faster_whisper(
                    audio_file,
                    model_size=model_size,
                    device=device,
                    beam_size=beam_size,
                    compute_type=compute_type,
                )
                total_runtime += runtime
                language = lang
                combined_transcript.append(hyp)
                print(f"[OK] Transcribed {audio_file.name} in {runtime}s")
            except Exception as e:
                print(f"[WARN] Skipping {audio_file.name}: {e}")
                continue

    elif audio_path.is_file():
        # Single audio file
        print(f"\n▶ Transcribing single audio file: {audio_path.name}...")
        try:
            hyp, runtime, lang = transcribe_faster_whisper(
                audio_path,
                model_size=model_size,
                device=device,
                beam_size=beam_size,
                compute_type=compute_type,
            )
            total_runtime = runtime
            language = lang
            combined_transcript.append(hyp)
            print(f"[OK] Transcribed {audio_path.name} in {runtime}s")
        except Exception as e:
            sys.exit(f"[ERROR] Failed to transcribe {audio_path.name}: {e}")

    else:
        sys.exit(f"[ERROR] Path does1 not exist or is not accessible: {audio_path}")

    # Combine all transcripts
    full_transcript = " ".join(combined_transcript)

    if not full_transcript:
        sys.exit("[ERROR] Generated transcript is empty.")

    # Save transcript to file
    out_transcript.parent.mkdir(parents=True, exist_ok=True)
    out_transcript.write_text(full_transcript, encoding="utf-8")

    print(f"\n[OK] Transcription saved → {out_transcript}")
    print(f"[INFO] Total Runtime: {round(total_runtime, 2)}s")
    print(f"[INFO] Detected Language: {language}")
    print(f"[INFO] Transcript Length: {len(full_transcript)} characters")
    print(f"[INFO] Word Count: {len(full_transcript.split())} words")


# ----------------------------
# CLI
# ----------------------------
def main():
    print("=== Faster-Whisper Reference Audio Transcription ===\n")

    parser = argparse.ArgumentParser(
        description="Transcribe reference audio using Faster-Whisper"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file or folder containing audio files"
    )
    parser.add_argument(
        "--out",
        default="Data-Pipeline/data/validation/model_validation/generated_transcript.txt",
        help="Output path for generated transcript (.txt)"
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Model size (tiny, base, small, medium, large-v3)"
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--compute-type", default="float32")

    args = parser.parse_args()

    transcribe_audio(
        Path(args.audio),
        Path(args.out),
        model_size=args.model,
        beam_size=args.beam_size,
        compute_type=args.compute_type,
    )


if __name__ == "__main__":
    main()
