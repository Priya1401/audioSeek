#!/usr/bin/env python3
"""
Cross-model validation with automatic failure detection
-------------------------------------------------------
Compares Faster-Whisper with OpenAI Whisper & Wav2Vec2.
Fails if BOTH models perform poorly on the same file.
"""

from pathlib import Path
import pandas as pd
from jiwer import wer
from rouge_score import rouge_scorer
import sys

# ------------------------
# Helpers
# ------------------------
def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").lower().strip()
    except Exception:
        return ""


def compute_scores(ref: str, hyp: str):
    if not ref or not hyp:
        return None, None
    w = wer(ref, hyp)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    r = scorer.score(ref, hyp)["rougeL"].fmeasure
    return round(w, 3), round(r, 3)


def get_all_txt(base: Path):
    return {p.name: p for p in base.rglob("*.txt")}


# ------------------------
# Main
# ------------------------
def validate_models(
    fw_dir="data/transcription_results",
    ow_dir="data/validation/cross_model_evaluation/openaiwhisper",
    w2v_dir="data/validation/cross_model_evaluation/wav2vec2",
    out_csv="data/validation/cross_model_evaluation/latest_validation_summary.csv",
    content_type="audiobook",
    wer_threshold=0.40,
    rouge_threshold=0.60,
):
    print("\n=== Running Validation vs Faster-Whisper ===")

    fw_dir, ow_dir, w2v_dir = Path(fw_dir), Path(ow_dir), Path(w2v_dir)
    fw_files, ow_files, w2v_files = get_all_txt(fw_dir), get_all_txt(ow_dir), get_all_txt(w2v_dir)
    rows = []
    alerts = []

    for fname, ref_path in fw_files.items():
        ref = load_text(ref_path)

        for model_name, model_files in [("OpenAI Whisper", ow_files), ("Wav2Vec2", w2v_files)]:
            if fname not in model_files:
                continue
            hyp = load_text(model_files[fname])
            w, r = compute_scores(ref, hyp)
            if w is None:
                continue
            rows.append({"File": fname, "Model": model_name, "WER": w, "ROUGE-L": r})

    if not rows:
        print("[WARN] No matching transcripts found.")
        return

    df = pd.DataFrame(rows)

    # --- Aggregate & flag concerning files ---
    df["Concerning"] = (df["WER"] > wer_threshold) | (df["ROUGE-L"] < rouge_threshold)
    df.to_csv(out_csv, index=False)

    # Detect files where BOTH models are concerning
    bad = (
        df[df["Concerning"]]
        .groupby("File")
        .filter(lambda g: len(g["Model"].unique()) > 1)
    )

    if not bad.empty:
        print("\nValidation FAILED for these files (both models concerning):")
        print(bad[["File", "Model", "WER", "ROUGE-L"]].to_string(index=False))
        print(f"\nSaved full summary → {out_csv}")
        sys.exit(1)  # fail the pipeline

    print(f"\nValidation passed. Summary saved → {out_csv}")
    print(df.to_string(index=False))
    sys.exit(0)


if __name__ == "__main__":
    validate_models()
