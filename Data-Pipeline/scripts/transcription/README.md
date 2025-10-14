# Transcription (Faster-Whisper)

Batch-transcribe a ZIP of audio files with Faster-Whisper. The script ignores junk files (`.__*`, `__MACOSX`) and saves as standardized transcript filenames.

## How to Run

From your repo root:
```bash
python Data-Pipeline/scripts/transcription/transcription.py \
  --zipfile Data-Pipeline/data/raw/edison.zip \
  --type audiobook \
  --outdir Data-Pipeline/data/transcription_results/edison
```

**Arguments (most common):**
- `--zipfile` : Path to ZIP with `.mp3/.wav` files
- `--type`    : `audiobook` or `podcast` (used in output names)
- `--outdir`  : Output folder inside your project
- `--model`   : `tiny | base | small | medium | large-v3`
- `--compute-type` : CPU: `int8` (fast) or `float32` (quality); CUDA GPU: `float16`
- `--beam-size` : 1 (fastest) to ~5 (slightly better)

## File Naming Format

Each transcript is saved as:
```
{type}_{zipfilename}_chapter_{nn}.txt
```

**Example**
- Zip: `edison.zip`
- Type: `audiobook`
- Outputs:
```
audiobook_edison_chapter_01.txt
audiobook_edison_chapter_02.txt
...
audiobook_edison_chapter_53.txt
```

A summary CSV is also written:
```
{type}_{zipfilename}_summary.csv
```

## Outputs

- One transcript `.txt` per audio in `--outdir`
- A summary CSV with: `Audio File, Transcript File, Model, Runtime (s), Language, Segments, CPU Usage (%), Memory Usage (%)`

## Quick Setup

Python 3.10+ and FFmpeg are required.
```bash
# Python packages
pip install faster-whisper pandas psutil av

# FFmpeg (macOS / Ubuntu)
# macOS:  brew install ffmpeg
# Ubuntu: sudo apt-get install -y ffmpeg
```

You can also install the packages using requirements.txt

## Notes

- On macOS (CPU), `--compute-type int8` is a good default for speed.
- The script sorts input files alphabetically to assign chapter numbers consistently.
- Junk files like `._*` and folders like `__MACOSX` are skipped automatically.
