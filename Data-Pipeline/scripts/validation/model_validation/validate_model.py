#!/usr/bin/env python3
"""
Validate Faster-Whisper Transcriptions against Reference Text

Modes:
1) Single file:
   --audio data/raw/chapter_03.mp3 --reference data/reference/chapter_03.txt

2) Batch (match by stem):
   --audio-dir data/raw/edison --reference-dir data/reference/edison

Outputs a CSV summary with WER, CER, ROUGE-L and runtime.
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import psutil, pandas as pd
from jiwer import wer, cer
from rouge_score import rouge_scorer


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


# ----------------------------
# IO helpers
# ----------------------------
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ----------------------------
# Validation runners
# ----------------------------
def validate_single(audio: Path, reference: Path, out_csv: Path,
                    model_size="base", beam_size=5, compute_type="float32") -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = detect_device()
    print(f"[INFO] Device: {device}")

    if not audio.exists():
        print(f"[ERROR] Audio not found: {audio}"); return 1
    if not reference.exists():
        print(f"[ERROR] Reference not found: {reference}"); return 1

    print(f"[INFO] Validating {audio.name} against {reference.name}")
    hyp, runtime, lang = transcribe_faster_whisper(audio, model_size=model_size, device=device,
                                                   beam_size=beam_size, compute_type=compute_type)
    ref = read_text(reference)
    w, c, r = compute_metrics(hyp, ref)
    if w is None:
        print("[ERROR] Empty hypothesis or reference."); return 1

    row = {
        "Audio File": audio.name,
        "Reference File": reference.name,
        "Model": f"Faster-Whisper({model_size})",
        "Language": lang,
        "Runtime (s)": runtime,
        "WER": w,
        "CER": c,
        "ROUGE-L": r,
        "CPU Usage (%)": psutil.cpu_percent(),
        "Memory Usage (%)": psutil.virtual_memory().percent,
    }
    df = pd.DataFrame([row])
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved → {out_csv}")
    print("\n=== Validation Summary ===")
    print(df.to_string(index=False))
    return 0


def validate_batch(audio_dir: Path, reference_dir: Path, out_csv: Path,
                   model_size="base", beam_size=5, compute_type="float32") -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = detect_device()
    print(f"[INFO] Device: {device}")

    audio_files = {p.stem: p for p in audio_dir.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac")}
    ref_files = {p.stem: p for p in reference_dir.glob("*.txt")}
    shared = sorted(set(audio_files).intersection(ref_files))

    if not shared:
        print(f"[ERROR] No matching stems between {audio_dir} and {reference_dir}")
        return 1

    print(f"[INFO] Found {len(shared)} matching pairs.")
    rows = []
    for stem in shared:
        a, r = audio_files[stem], ref_files[stem]
        print(f"\n Validating {a.name} vs {r.name}")
        try:
            hyp, runtime, lang = transcribe_faster_whisper(a, model_size=model_size, device=device,
                                                           beam_size=beam_size, compute_type=compute_type)
            ref = read_text(r)
            w, c, rl = compute_metrics(hyp, ref)
            if w is None:
                print(f"[WARN] Skipping {stem}: empty hypothesis/reference")
                continue

            rows.append({
                "Audio File": a.name,
                "Reference File": r.name,
                "Model": f"Faster-Whisper({model_size})",
                "Language": lang,
                "Runtime (s)": runtime,
                "WER": w,
                "CER": c,
                "ROUGE-L": rl,
                "CPU Usage (%)": psutil.cpu_percent(),
                "Memory Usage (%)": psutil.virtual_memory().percent,
            })
        except Exception as e:
            print(f"[WARN] Skipping {stem}: {e}")

    if not rows:
        print("[ERROR] No successful validations."); return 1
    df = pd.DataFrame(rows).sort_values("WER").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Batch summary saved → {out_csv}")
    print("\n=== Batch Validation Summary ===")
    print(df.to_string(index=False))
    # Optional: non-zero exit if average WER too high
    return 0


# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Validate Faster-Whisper against reference text")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audio", help="Path to a single audio file (.mp3/.wav)")
    mode.add_argument("--audio-dir", help="Directory of audio files")
    p.add_argument("--reference", help="Path to reference .txt (single mode)")
    p.add_argument("--reference-dir", help="Directory with reference .txt files (batch mode)")
    p.add_argument("--out", default="Data-Pipeline/data/validation/fasterwhisper_vs_reference_summary.csv",
                   help="Output CSV path")
    p.add_argument("--model", default="base", help="Faster-Whisper model size")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--compute-type", default="float32")
    args = p.parse_args()

    out_csv = Path(args.out)

    if args.audio:
        if not args.reference:
            print("[ERROR] --reference is required with --audio")
            sys.exit(1)
        sys.exit(validate_single(Path(args.audio), Path(args.reference), out_csv,
                                 model_size=args.model, beam_size=args.beam_size, compute_type=args.compute_type))
    else:
        if not args.reference_dir:
            print("[ERROR] --reference-dir is required with --audio-dir")
            sys.exit(1)
        sys.exit(validate_batch(Path(args.audio_dir), Path(args.reference_dir), out_csv,
                                model_size=args.model, beam_size=args.beam_size, compute_type=args.compute_type))


if __name__ == "__main__":
    print("This is the script for validating model with audio and transcript")
    main()