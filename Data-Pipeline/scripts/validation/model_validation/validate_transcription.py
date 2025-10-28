#!/usr/bin/env python3
"""
Validate Generated Transcription Against Official Reference
-----------------------------------------------------------
- Compares a generated transcript (from transcribe_reference.py) 
  against an official reference transcript.
- Computes WER, CER, and ROUGE-L metrics.
- Saves validation summary to CSV.
- Applies threshold-based pass/fail validation gate.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psutil
from jiwer import wer, cer
from rouge_score import rouge_scorer

# Add Data-Pipeline directory to path
script_dir = Path(__file__).resolve().parent
data_pipeline_dir = script_dir.parent.parent.parent
sys.path.insert(0, str(data_pipeline_dir))


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(pred: str, ref: str):
    """
    Calculate WER, CER, and ROUGE-L between prediction and reference.

    Args:
        pred: Predicted/generated transcript
        ref: Official reference transcript

    Returns:
        Tuple of (WER, CER, ROUGE-L F-measure)
    """
    pred = (pred or "").lower().strip()
    ref = (ref or "").lower().strip()

    if not pred or not ref:
        return None, None, None

    w = round(wer(ref, pred), 4)
    c = round(cer(ref, pred), 4)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    r = round(scorer.score(ref, pred)["rougeL"].fmeasure, 4)

    return w, c, r


# ----------------------------
# Validation
# ----------------------------
def validate_transcription(
        generated_transcript: Path,
        official_reference: Path,
        out_csv: Path,
        model_name="Faster-Whisper(base)",
):
    """
    Compare generated transcript against official reference and compute metrics.
    """
    print("[INFO] Checking file existence...", flush=True)

    # Ensure input files exist
    if not generated_transcript.exists():
        sys.exit(f"[ERROR] Generated transcript not found: {generated_transcript}")

    if not official_reference.exists():
        sys.exit(f"[ERROR] Official reference not found: {official_reference}")

    # Ensure output directory exists
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Reading generated transcript...", flush=True)
    generated_text = generated_transcript.read_text(encoding="utf-8", errors="ignore")
    print(f"[INFO] Finished reading generated transcript ({len(generated_text):,} chars)", flush=True)

    print("[INFO] Reading official reference transcript...", flush=True)
    reference_text = official_reference.read_text(encoding="utf-8", errors="ignore")
    print(f"[INFO] Finished reading reference transcript ({len(reference_text):,} chars)", flush=True)

    print(f"\n[INFO] Generated transcript length: {len(generated_text)} chars")
    print(f"[INFO] Reference transcript length: {len(reference_text)} chars")

    # Compute metrics
    print(f"\n[INFO] Computing WER, CER, and ROUGE-L metrics...", flush=True)
    w, c, r = compute_metrics(generated_text, reference_text)
    print("[INFO] Metric computation completed.", flush=True)

    if w is None:
        sys.exit("[ERROR] Could not compute metrics — empty hypothesis or reference.")

    print(f"[INFO] WER: {w}")
    print(f"[INFO] CER: {c}")
    print(f"[INFO] ROUGE-L: {r}")

    # Create summary
    print("[INFO] Creating validation summary dataframe...", flush=True)
    summary = {
        "Generated Transcript": generated_transcript.name,
        "Official Reference": official_reference.name,
        "Model": model_name,
        "WER": w,
        "CER": c,
        "ROUGE-L": r,
        "Generated Length (chars)": len(generated_text),
        "Reference Length (chars)": len(reference_text),
        "Generated Words": len(generated_text.split()),
        "Reference Words": len(reference_text.split()),
        "CPU Usage (%)": psutil.cpu_percent(),
        "Memory Usage (%)": psutil.virtual_memory().percent,
    }

    # Save to CSV
    print("[INFO] Writing validation summary to CSV...", flush=True)
    df = pd.DataFrame([summary])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved validation summary → {out_csv}", flush=True)

    # Clear existing transcript file if it exists
    if os.path.exists(generated_transcript):
        print(f"[INFO] Cleaning up generated transcript: {generated_transcript}", flush=True)
        os.remove(generated_transcript)

    print(f"\n[OK] Validation summary saved → {out_csv}")
    print("\n=== Validation Summary ===")
    print(df.to_string(index=False))

    # --- Threshold-based Validation Gate ---
    WER_THRESHOLD = 0.30
    CER_THRESHOLD = 0.15
    ROUGE_THRESHOLD = 0.90

    print("\n=== Threshold Validation ===")
    print(f"WER Threshold: ≤ {WER_THRESHOLD}")
    print(f"CER Threshold: ≤ {CER_THRESHOLD}")
    print(f"ROUGE-L Threshold: ≥ {ROUGE_THRESHOLD}")

    if w > WER_THRESHOLD or c > CER_THRESHOLD or r < ROUGE_THRESHOLD:
        print("\n Validation FAILED:")
        if w > WER_THRESHOLD:
            print(f"   • WER ({w}) exceeds threshold ({WER_THRESHOLD})")
        if c > CER_THRESHOLD:
            print(f"   • CER ({c}) exceeds threshold ({CER_THRESHOLD})")
        if r < ROUGE_THRESHOLD:
            print(f"   • ROUGE-L ({r}) below threshold ({ROUGE_THRESHOLD})")
        sys.exit(1)
    else:
        print("\n Validation PASSED:")
        print(f"   • WER ({w}) ≤ {WER_THRESHOLD}")
        print(f"   • CER ({c}) ≤ {CER_THRESHOLD}")
        print(f"   • ROUGE-L ({r}) ≥ {ROUGE_THRESHOLD}")
        print("\nModel performance meets all quality thresholds!")
        sys.exit(0)


# ----------------------------
# CLI
# ----------------------------
def main():
    print("=== Transcription Validation Against Official Reference ===\n")

    parser = argparse.ArgumentParser(
        description="Validate generated transcript against official reference"
    )
    parser.add_argument(
        "--generated",
        default="Data-Pipeline/data/validation/model_validation/generated_transcript.txt",
        help="Path to generated transcript file (.txt)"
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to official reference transcript file (.txt)"
    )
    parser.add_argument(
        "--out",
        default="Data-Pipeline/data/validation/model_validation/validation_summary.csv",
        help="Output CSV path for validation summary"
    )
    parser.add_argument(
        "--model-name",
        default="Faster-Whisper(base)",
        help="Model name for reporting"
    )

    args = parser.parse_args()

    validate_transcription(
        Path(args.generated),
        Path(args.reference),
        Path(args.out),
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
