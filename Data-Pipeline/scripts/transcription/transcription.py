#!/usr/bin/env python3
"""
AudioSEEK — Batch Faster-Whisper Transcription
Takes a ZIP archive of audio files (.mp3/.wav), transcribes each, and saves
type_zipfilename_chapter_{n}.txt in Data-Pipeline/data/transcription_results/
"""

import sys, time, argparse, zipfile, tempfile
from pathlib import Path
import psutil, pandas as pd


# ----------------------------
# Helpers
# ----------------------------

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma")


def detect_device():
    """Detect GPU or CPU device automatically."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_paths(zip_path: Path, results_dir: Path):
    """Check ZIP existence and ensure output directory exists."""
    if not zip_path.exists():
        sys.exit(f"[ERROR] Zip file not found: {zip_path}")
    results_dir.mkdir(parents=True, exist_ok=True)


def is_junk_macos(p: Path) -> bool:
    """
    Identify macOS junk or hidden files.
    - '._filename' resource forks
    - '__MACOSX' directories
    - hidden files starting with '.' (unless valid audio)
    """
    if p.name.startswith("._"):
        return True
    if any(part == "__MACOSX" for part in p.parts):
        return True
    if p.name.startswith(".") and p.suffix.lower() not in AUDIO_EXTS:
        return True
    return False


def is_valid_audio_path(p: Path) -> bool:
    """Check if path looks like a valid audio file we can transcribe."""
    if not p.is_file():
        return False
    if p.suffix.lower() not in AUDIO_EXTS:
        return False
    if is_junk_macos(p):
        return False
    try:
        if p.stat().st_size == 0:
            return False
    except Exception:
        return False
    return True


def collect_audio_files_from_zip(zip_path: Path, tmp_dir: Path):
    """Extract ZIP and return only valid audio file paths."""
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)
    all_files = [p for p in Path(tmp_dir).rglob("*")]
    valid_files = [p for p in all_files if is_valid_audio_path(p)]
    # Sort alphabetically for consistent numbering
    valid_files.sort(key=lambda x: x.name.lower())
    return valid_files


def run_faster_whisper(audio_path: Path, device: str, model_size: str,
                       beam_size: int, compute_type: str):
    """Transcribe a single audio file with Faster-Whisper."""
    from faster_whisper import WhisperModel
    start = time.time()

    kwargs = {"device": device}
    if compute_type:
        kwargs["compute_type"] = compute_type

    model = WhisperModel(model_size, **kwargs)
    segments, info = model.transcribe(str(audio_path), beam_size=beam_size)

    lines = [f"[{s.start:.2f}-{s.end:.2f}] {s.text}" for s in segments]
    runtime = time.time() - start
    meta = {
        "language": getattr(info, "language", "unknown"),
        "duration": getattr(info, "duration", None),
    }
    return runtime, len(lines), lines, meta


def save_lines(path: Path, lines):
    """Write transcript lines to text file."""
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch Faster-Whisper Transcription from ZIP")
    parser.add_argument("--zipfile", default="Data-Pipeline/data/raw/audios.zip",
                        help="Path to ZIP file containing audio files")
    parser.add_argument("--type", required=True,
                        help="Type of content: audiobook or podcast (used in file naming)")
    parser.add_argument("--outdir", default="Data-Pipeline/data/transcription_results",
                        help="Output directory for transcripts and summary CSV")
    parser.add_argument("--model", default="base",
                        help="Faster-Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--compute-type", default="float32",
                        help='Compute type (e.g., "float16" for CUDA, "float32" or "int8" for CPU)')
    args = parser.parse_args()

    zip_path = Path(args.zipfile)
    results_dir = Path(args.outdir)
    ensure_paths(zip_path, results_dir)

    # Sanitize and validate type
    type_name = args.type.strip().lower()
    if type_name not in {"audiobook", "podcast"}:
        sys.exit("[ERROR] --type must be either 'audiobook' or 'podcast'.")

    # Extract base name of the ZIP (without extension)
    zip_basename = zip_path.stem

    device = detect_device()
    print(f"[INFO] Device: {device}")

    rows = []

    # --- extract ZIP to a temp dir ---
    with tempfile.TemporaryDirectory() as tmp:
        print(f"[INFO] Extracting {zip_path.name} → {tmp}")
        audio_files = collect_audio_files_from_zip(zip_path, Path(tmp))

        if not audio_files:
            sys.exit("[ERROR] No valid audio files found in the ZIP.")

        print(f"[INFO] Found {len(audio_files)} valid audio files to process.")

        # --- loop through audio files ---
        for idx, audio_file in enumerate(audio_files, start=1):
            print(f"\n▶ Transcribing {audio_file.name} ...")
            try:
                rt, segs, lines, meta = run_faster_whisper(
                    audio_path=audio_file,
                    device=device,
                    model_size=args.model,
                    beam_size=args.beam_size,
                    compute_type=args.compute_type,
                )
            except Exception as e:
                # Skip any file that fails to decode or transcribe
                print(f"[WARN] Skipping '{audio_file.name}': {e}")
                continue

            # --- Construct standardized output name ---
            new_name = f"{type_name}_{zip_basename}_chapter_{idx:02d}.txt"
            out_txt = results_dir / new_name
            save_lines(out_txt, lines)
            print(f"[OK] Saved transcript → {out_txt}")

            rows.append({
                "Audio File": audio_file.name,
                "Transcript File": new_name,
                "Model": f"Faster-Whisper({args.model})",
                "Runtime (s)": round(rt, 2),
                "Language": meta.get("language", "unknown"),
                "Segments": segs,
                "CPU Usage (%)": psutil.cpu_percent(),
                "Memory Usage (%)": psutil.virtual_memory().percent,
            })

    # --- summary CSV ---
    if not rows:
        sys.exit("[ERROR] No successful transcriptions were produced.")
    df = pd.DataFrame(rows)
    summary_csv = results_dir / f"{type_name}_{zip_basename}_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"\n[OK] Summary saved → {summary_csv}")
    print("\n=== Batch Transcription Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    print("This is the script for transcription of raw audio files.")
    # main()
