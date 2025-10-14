#!/usr/bin/env python3
"""
AudioSEEK — Transcription Benchmark
Models: Faster-Whisper, WhisperX (optional: whisper.cpp)

Outputs:
  - results/transcription_results/faster_whisper.txt
  - results/transcription_results/whisperx.txt
  - results/transcription_benchmark_summary.csv
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import psutil
import pandas as pd

# ----------------------------
# Helpers: devices & io
# ----------------------------

def detect_devices():
    """
    Decide devices for each backend.
    - WhisperX uses PyTorch (CUDA/MPS/CPU)
    - Faster-Whisper uses CTranslate2 (CUDA/CPU). Metal (MPS) isn't used here, so default to CPU on mac.
    """
    dev = {"whisperx": "cpu", "faster_whisper": "cpu"}

    try:
        import torch
        if torch.cuda.is_available():
            dev["whisperx"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev["whisperx"] = "mps"
        else:
            dev["whisperx"] = "cpu"
    except Exception:
        dev["whisperx"] = "cpu"

    # For faster-whisper: use CUDA if available; otherwise CPU (MPS not used here)
    try:
        import torch
        if torch.cuda.is_available():
            dev["faster_whisper"] = "cuda"
        else:
            dev["faster_whisper"] = "cpu"
    except Exception:
        dev["faster_whisper"] = "cpu"

    return dev


def ensure_paths(audio_path: Path, results_dir: Path):
    if not audio_path.exists():
        sys.exit(f"[ERROR] Audio file not found: {audio_path}")
    results_dir.mkdir(parents=True, exist_ok=True)


def save_lines(path: Path, lines):
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Runners
# ----------------------------

def run_faster_whisper(audio_path: Path, device: str, model_size: str = "base", beam_size: int = 5, compute_type: str = None):
    """
    Transcribe with Faster-Whisper (CTranslate2).
    compute_type: e.g., "float16" on CUDA; None lets the lib choose.
    """
    from faster_whisper import WhisperModel
    start = time.time()

    kwargs = {"device": device}
    if compute_type:
        kwargs["compute_type"] = compute_type

    model = WhisperModel(model_size, **kwargs)
    segments, info = model.transcribe(str(audio_path), beam_size=beam_size)

    lines = []
    seg_count = 0
    for s in segments:
        seg_count += 1
        lines.append(f"[{s.start:.2f}-{s.end:.2f}] {s.text}")

    runtime = time.time() - start
    meta = {"language": getattr(info, "language", "unknown"), "duration": getattr(info, "duration", None)}
    return runtime, seg_count, lines, meta


def run_whisperx(audio_path: Path, device: str, model_size: str = "base"):
    """
    Transcribe with WhisperX, then align for word-precise timestamps.
    """
    import whisperx
    import torch

    # WhisperX prefers torch dtype control on MPS for stability
    if device == "mps":
        torch.set_default_dtype(torch.float32)

    start = time.time()
    asr_model = whisperx.load_model(model_size, device, compute_type="float32")
    result = asr_model.transcribe(str(audio_path))

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], align_model, metadata, str(audio_path), device)

    lines = [f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}" for seg in result_aligned["segments"]]
    runtime = time.time() - start
    seg_count = len(result_aligned["segments"])
    meta = {"language": result.get("language", "unknown")}
    return runtime, seg_count, lines, meta


def run_whisper_cpp(audio_path: Path, model_name: str = "base.en", results_dir: Path = Path("results/transcription_results")):
    """
    Optional CPU baseline using whisper.cpp binary on PATH.
    Change the command if your binary is named differently (e.g., 'main').
    """
    start = time.time()
    cmd = f'whisper-cpp --model {model_name} --output-txt --output-dir "{results_dir}" "{audio_path}"'
    code = os.system(cmd)
    runtime = time.time() - start
    if code != 0:
        raise RuntimeError("whisper.cpp command failed. Ensure the binary is installed and on PATH.")

    # Find the latest .txt produced
    txts = list(results_dir.glob("*.txt"))
    if not txts:
        raise FileNotFoundError("No transcript produced by whisper.cpp")
    # pick most recent
    txts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    text = txts[0].read_text(encoding="utf-8", errors="ignore").splitlines()
    seg_count = len(text)
    meta = {"language": "en"}
    return runtime, seg_count, text, meta


# ----------------------------
# Accuracy (optional)
# ----------------------------

def compute_wer(pred_text: str, ref_path: Path):
    try:
        from jiwer import wer
    except ImportError:
        return "N/A"
    if not ref_path.exists():
        return "N/A"
    ref = ref_path.read_text(encoding="utf-8", errors="ignore").lower()
    hyp = pred_text.lower()
    try:
        return round(wer(ref, hyp), 3)
    except Exception:
        return "N/A"


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="AudioSEEK Transcription Benchmark")
    parser.add_argument("--audio", default="data/raw/sample.mp3", help="Path to audio file")
    parser.add_argument("--outdir", default="results/transcription_results", help="Directory to store transcripts")
    parser.add_argument("--enable-whisperx", action="store_true", help="Run WhisperX benchmark")
    parser.add_argument("--enable-faster", action="store_true", help="Run Faster-Whisper benchmark")
    parser.add_argument("--enable-cpp", action="store_true", help="Run whisper.cpp benchmark (requires binary)")
    parser.add_argument("--faster-model", default="base", help="Faster-Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--whisperx-model", default="base", help="WhisperX model size (tiny, base, small, medium, large-v2)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for Faster-Whisper")
    parser.add_argument("--compute-type", default=None, help='Faster-Whisper compute type (e.g., "float16" on CUDA)')
    parser.add_argument("--ref", default="data/reference_transcript.txt", help="Optional reference transcript for WER")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    results_dir = Path(args.outdir)
    ensure_paths(audio_path, results_dir)

    devices = detect_devices()
    print(f"[INFO] Devices -> WhisperX: {devices['whisperx']} | Faster-Whisper: {devices['faster_whisper']}")

    rows = []

    # ------------------ Faster-Whisper ------------------
    if args.enable_faster:
        print("\n▶ Running Faster-Whisper ...")
        rt, segs, lines, meta = run_faster_whisper(
            audio_path=audio_path,
            device=devices["faster_whisper"],
            model_size=args.faster_model,
            beam_size=args.beam_size,
            compute_type=args.compute_type,
        )
        save_lines(results_dir / "faster_whisper.txt", lines)
        wer_val = compute_wer("\n".join(lines), Path(args.ref))
        rows.append({
            "Model": f"Faster-Whisper({args.faster_model})",
            "Runtime (s)": round(rt, 2),
            "Language": meta.get("language", "unknown"),
            "Segments": segs,
            "CPU Usage (%)": psutil.cpu_percent(),
            "Memory Usage (%)": psutil.virtual_memory().percent,
            "WER": wer_val,
            "Approx Cost/hr ($)": 0
        })

    # ------------------ WhisperX ------------------
    if args.enable_whisperx:
        print("\n▶ Running WhisperX ...")
        rt, segs, lines, meta = run_whisperx(
            audio_path=audio_path,
            device="cpu",
            model_size=args.whisperx_model,
        )
        save_lines(results_dir / "whisperx.txt", lines)
        wer_val = compute_wer("\n".join(lines), Path(args.ref))
        rows.append({
            "Model": f"WhisperX({args.whisperx_model})",
            "Runtime (s)": round(rt, 2),
            "Language": meta.get("language", "unknown"),
            "Segments": segs,
            "CPU Usage (%)": psutil.cpu_percent(),
            "Memory Usage (%)": psutil.virtual_memory().percent,
            "WER": wer_val,
            "Approx Cost/hr ($)": 0
        })

    # ------------------ whisper.cpp (optional) ------------------
    if args.enable_cpp:
        print("\n▶ Running whisper.cpp ...")
        try:
            rt, segs, lines, meta = run_whisper_cpp(
                audio_path=audio_path,
                model_name="base.en",
                results_dir=results_dir
            )
            # Save (note: whisper.cpp already writes its own txt; we also save a normalized one)
            save_lines(results_dir / "whisper_cpp.txt", lines)
            wer_val = compute_wer("\n".join(lines), Path(args.ref))
            rows.append({
                "Model": "whisper.cpp(base.en)",
                "Runtime (s)": round(rt, 2),
                "Language": meta.get("language", "unknown"),
                "Segments": segs,
                "CPU Usage (%)": psutil.cpu_percent(),
                "Memory Usage (%)": psutil.virtual_memory().percent,
                "WER": wer_val,
                "Approx Cost/hr ($)": 0
            })
        except Exception as e:
            print(f"[WARN] whisper.cpp run skipped: {e}")

    if not rows:
        print("[ERROR] No models selected. Use --enable-faster and/or --enable-whisperx (and/or --enable-cpp).")
        sys.exit(1)

    # ------------------ Summary ------------------
    df = pd.DataFrame(rows).sort_values("Runtime (s)").reset_index(drop=True)
    print("\n=== Benchmark Summary ===")
    print(df.to_string(index=False))

    summary_csv = Path("results/transcription_benchmark_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    print(f"\n[OK] Saved summary → {summary_csv}")


if __name__ == "__main__":
    main()
