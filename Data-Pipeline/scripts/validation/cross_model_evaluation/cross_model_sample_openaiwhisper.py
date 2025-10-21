#!/usr/bin/env python3
"""
OpenAI Whisper Sample Transcription
-----------------------------------
Transcribes sampled audio files from ZIP using OpenAI Whisper model
and saves standardized transcripts.
"""

import sys, time, zipfile, tempfile, argparse
from pathlib import Path
import whisper
from utils.audio_utils import ensure_paths, extract_zip_filtered, standardized_output_name


def transcribe_sample_openaiwhisper(zipfile_path: str, outdir_path: str, content_type: str, model_size="base"):
    zip_path = Path(zipfile_path)
    outdir = Path(outdir_path)
    ensure_paths(zip_path, outdir)

    print(f"[INFO] Loading OpenAI Whisper model: {model_size} (device=cpu)")
    model = whisper.load_model(model_size, device="cpu")

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[INFO] Extracting {zip_path.name} → {tmp}")
        audio_files = extract_zip_filtered(zip_path, Path(tmp))
        if not audio_files:
            sys.exit("[ERROR] No valid audio files found in ZIP.")

        for audio in audio_files:
            print(f"[OpenAI Whisper] Transcribing {audio.name} ...")
            start = time.time()
            result = model.transcribe(str(audio))
            lines = [f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}" for seg in result["segments"]]
            out_name = standardized_output_name(content_type, audio.name)
            out_file = outdir / out_name
            out_file.write_text("\n".join(lines), encoding="utf-8")
            print(f"[OK] Saved → {out_file} ({round(time.time() - start, 2)}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", required=True)
    parser.add_argument("--outdir", default="data/validation/cross_model_evaluation/openaiwhisper")
    parser.add_argument("--type", required=True, choices=["audiobook", "podcast"])
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    transcribe_sample_openaiwhisper(args.zipfile, args.outdir, args.type, args.model)
