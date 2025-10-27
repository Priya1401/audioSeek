#!/usr/bin/env python3
"""
OpenAI Whisper Sample Transcription
-----------------------------------
Transcribes sampled audio files from ZIP using OpenAI Whisper model
and saves standardized transcripts.
"""

import os
import sys, time, zipfile, tempfile, argparse
sys.path.append(os.path.abspath('../../../'))
from pathlib import Path
import whisper
from scripts.transcription.utils.audio_utils import ensure_paths, extract_zip_filtered, standardized_output_name


def transcribe_sample_openaiwhisper(zipfile_path: str, outdir_path: str, content_type: str, model_size="base"):
    zip_path = Path(zipfile_path)
    outdir = Path(outdir_path)
    ensure_paths(zip_path, outdir)

    print(f"[INFO] Loading OpenAI Whisper model: {model_size} (device=cpu)", flush=True)
    model = whisper.load_model(model_size, device="cpu")

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[INFO] Extracting {zip_path.name} → {tmp}", flush=True)
        audio_files = extract_zip_filtered(zip_path, Path(tmp))
        if not audio_files:
            sys.exit("[ERROR] No valid audio files found in ZIP.")

        print(f"[INFO] Found {len(audio_files)} valid audio files for transcription.", flush=True)
        total_files = len(audio_files)
        total_runtime = 0.0

        for idx, audio in enumerate(audio_files, start=1):
            print(f"\n[OpenAI Whisper] ({idx}/{total_files}) Transcribing {audio.name} ...", flush=True)
            start = time.time()

            # Start transcription
            result = model.transcribe(str(audio))

            # If the model supports segments, log progress
            segments = result.get("segments", [])
            num_segments = len(segments)
            if num_segments > 0:
                print(f"[INFO] {audio.name}: {num_segments} segments detected — logging progress every ~10%.", flush=True)
                last_pct_logged = -10
                for s_idx, seg in enumerate(segments, start=1):
                    pct = int((s_idx / num_segments) * 100)
                    if pct >= last_pct_logged + 10:
                        print(
                            f"[PROGRESS] {audio.name}: {pct}% ({s_idx}/{num_segments} segments) "
                            f"[{seg['start']:.1f}s → {seg['end']:.1f}s] text='{seg['text'].strip()[:60]}...'",
                            flush=True
                        )
                        last_pct_logged = pct

            # Format and save output
            lines = [f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}" for seg in segments]
            out_name = standardized_output_name(content_type, audio.name)
            out_file = outdir / out_name
            out_file.write_text("\n".join(lines), encoding="utf-8")

            elapsed = round(time.time() - start, 2)
            total_runtime += elapsed
            print(f"[OK] Saved → {out_file} ({elapsed}s)", flush=True)

        print(f"\n[OK] All files transcribed successfully.", flush=True)
        print(f"[INFO] Total processing time: {round(total_runtime, 2)}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", required=True)
    parser.add_argument("--outdir", default="data/validation/cross_model_evaluation/openaiwhisper")
    parser.add_argument("--type", required=True, choices=["audiobook", "podcast"])
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    transcribe_sample_openaiwhisper(args.zipfile, args.outdir, args.type, args.model)
