#!/usr/bin/env python3
"""
Validate Faster-Whisper Transcriptions Against a Combined Reference Text
-----------------------------------------------------------------------
- Takes a ZIP archive of multiple audio files.
- Uses helper functions from scripts.transcription.utils.audio_utils.
- Transcribes all valid audio files, concatenates transcripts in order.
- Compares combined text vs. a single reference transcript.
- Outputs WER, CER, and ROUGE-L for full combined validation.
"""

import argparse
import os
import sys
import time
import tempfile
from pathlib import Path
import psutil
import pandas as pd
from jiwer import wer, cer
from rouge_score import rouge_scorer

sys.path.append(os.path.abspath('../../../'))

from scripts.transcription.utils.audio_utils import extract_zip_filtered, ensure_paths


# ----------------------------
# ASR
# ----------------------------
def transcribe_faster_whisper(audio_path: Path, model_size="base", device="cpu", beam_size=5, compute_type="float32"):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    start = time.time()
    segments, info = model.transcribe(str(audio_path), beam_size=beam_size)
    text = " ".join(s.text for s in segments).strip()
    runtime = round(time.time() - start, 2)
    lang = getattr(info, "language", "unknown")
    return text, runtime, lang


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(pred: str, ref: str):
    pred = (pred or "").lower().strip()
    ref = (ref or "").lower().strip()
    if not pred or not ref:
        return None, None, None
    w = round(wer(ref, pred), 4)
    c = round(cer(ref, pred), 4)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    r = round(scorer.score(ref, pred)["rougeL"].fmeasure, 4)
    return w, c, r


def detect_device():
    return "cpu"


# ----------------------------
# Combined ZIP Validation
# ----------------------------
def validate_combined_zip(
    zip_path: Path,
    reference: Path,
    out_csv: Path,
    model_size="base",
    beam_size=5,
    compute_type="float32",
):
    """Extract ZIP, transcribe all audio files, concatenate, and validate."""
    ensure_paths(zip_path, out_csv.parent)

    if not reference.exists():
        sys.exit(f"[ERROR] Reference text not found: {reference}")

    device = detect_device()
    print(f"[INFO] Device: {device}")
    ref_text = reference.read_text(encoding="utf-8", errors="ignore")

    combined_transcript = []
    total_runtime = 0.0
    language = "unknown"

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[INFO] Extracting {zip_path.name} → {tmpdir}")
        audio_files = sorted(
            extract_zip_filtered(zip_path, Path(tmpdir)),
            key=lambda x: x.name
        )

        if not audio_files:
            sys.exit("[ERROR] No valid audio files found inside the ZIP.")

        print(f"[INFO] Found {len(audio_files)} valid audio files to process.")
        for audio_file in audio_files:
            print(f"\n▶ Transcribing {audio_file.name} ...")
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
                print(f"[OK] Transcribed {audio_file.name} ({runtime}s)")
            except Exception as e:
                print(f"[WARN] Skipping {audio_file.name}: {e}")
                continue

    full_hyp = " ".join(combined_transcript)
    w, c, r = compute_metrics(full_hyp, ref_text)
    if w is None:
        sys.exit("[ERROR] Could not compute metrics — empty hypothesis or reference.")

    summary = {
        "Zip File": zip_path.name,
        "Reference File": reference.name,
        "Model": f"Faster-Whisper({model_size})",
        "Language": language,
        "Total Runtime (s)": round(total_runtime, 2),
        "WER": w,
        "CER": c,
        "ROUGE-L": r,
        "CPU Usage (%)": psutil.cpu_percent(),
        "Memory Usage (%)": psutil.virtual_memory().percent,
    }

    df = pd.DataFrame([summary])
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Combined validation summary saved → {out_csv}")
    print("\n=== Combined Validation Summary ===")
    print(df.to_string(index=False))

    # --- Threshold-based Validation Gate ---
    WER_THRESHOLD = 0.30
    CER_THRESHOLD = 0.15
    ROUGE_THRESHOLD = 0.90

    if w > WER_THRESHOLD or c > CER_THRESHOLD or r < ROUGE_THRESHOLD:
        print("\nValidation FAILED:")
        print(f"WER ({w}) > {WER_THRESHOLD} or CER ({c}) > {CER_THRESHOLD} or ROUGE ({r}) < {ROUGE_THRESHOLD}")
        sys.exit(1)
    else:
        print("\nValidation PASSED:")
        print(f"Model performance meets thresholds — WER≤{WER_THRESHOLD}, CER≤{CER_THRESHOLD}, ROUGE≥{ROUGE_THRESHOLD}")
        sys.exit(0)


# ----------------------------
# CLI
# ----------------------------
def main():
    print("Running Faster-Whisper Model Validation")

    # parser = argparse.ArgumentParser(
    #     description="Validate Faster-Whisper transcription of ZIP audio set vs combined reference text"
    # )
    # parser.add_argument("--zipfile", required=True, help="ZIP file containing multiple audio files")
    # parser.add_argument("--reference", required=True, help="Single combined reference text file (.txt)")
    # parser.add_argument(
    #     "--out",
    #     default="Data-Pipeline/data/validation/model_validation/fasterwhisper_combined_validation_summary.csv",
    #     help="Output CSV for summary",
    # )
    # parser.add_argument("--model", default="base", help="Model size (tiny, base, small, medium, large-v3)")
    # parser.add_argument("--beam-size", type=int, default=5)
    # parser.add_argument("--compute-type", default="float32")
    # args = parser.parse_args()
    #
    # validate_combined_zip(
    #     Path(args.zipfile),
    #     Path(args.reference),
    #     Path(args.out),
    #     model_size=args.model,
    #     beam_size=args.beam_size,
    #     compute_type=args.compute_type,
    # )


if __name__ == "__main__":
    main()
