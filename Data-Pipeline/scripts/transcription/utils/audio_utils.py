from __future__ import annotations

import random
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

# --------- constants ---------
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma")


# --------- basic helpers ---------
def is_macos_junk(p: Path) -> bool:
    return p.name.startswith("._") or any(part == "__MACOSX" for part in p.parts)


def is_valid_audio(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() not in AUDIO_EXTS:
        return False
    if is_macos_junk(p):
        return False
    try:
        return p.stat().st_size > 0
    except Exception:
        return False


def ensure_paths(input_path: Path, outdir: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] Input not found: {input_path}")
    outdir.mkdir(parents=True, exist_ok=True)


# --------- zip extraction / sampling ---------
def extract_zip_filtered(zip_path: Path, dest_dir: Path) -> list[Path]:
    """
    Extract only valid audio files from ZIP, skipping macOS junk.
    Returns list of extracted audio file Paths.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        members = [
            name for name in z.namelist()
            if not name.startswith("__MACOSX/")
               and not name.split("/")[-1].startswith("._")
        ]
        z.extractall(dest_dir, members=members)

    return [p for p in Path(dest_dir).rglob("*") if is_valid_audio(p)]


def sample_zip_filtered(folder_path: Path, sample_size: int, out_zip_path: Path) -> Path:
    """
    Randomly sample valid audio files from a folder and write them to a sample ZIP.
    Skips macOS junk and non-audio files.
    """
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"[ERROR] Invalid folder path: {folder_path}")

    # Collect all valid audio files recursively
    candidates = [
        p for p in folder_path.rglob("*")
        if p.is_file()
           and not p.name.startswith("._")
           and p.suffix.lower() in AUDIO_EXTS
    ]

    if not candidates:
        raise ValueError("[ERROR] No valid audio files found in the folder.")

    # Randomly select sample_size files
    subset = random.sample(candidates, min(sample_size, len(candidates)))

    # Create a temporary directory (optional, but keeps code clean)
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        # Copy sampled files into temp dir (to flatten paths if needed)
        for src in subset:
            shutil.copy(src, tmp_dir / src.name)

        # Create zip
        with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z_out:
            for f in tmp_dir.iterdir():
                z_out.write(f, arcname=f.name)

        return out_zip_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def collect_audio_files_from_input_directory(input_dir: Path):
    """Extract all files from input directory"""
    all_files = [p for p in Path(input_dir).rglob("*")]
    valid_files = [p for p in all_files if is_valid_audio(p)]

    # Sort alphabetically for consistent numbering
    valid_files.sort(key=lambda x: x.name.lower())
    return valid_files


def save_lines(path: Path, lines):
    """Write Transcript lines to text files."""
    path.write_text("\n".join(lines), encoding='utf-8')


def extract_chapter_or_episode_number(filename: str) -> int | None:
    """
    Extract chapter or episode number from filename.
    Handles chapter 0, episode 0, and various naming patterns.

    Args:
        filename: Name of the audio file

    Returns:
        Chapter/episode number (can be 0), or None if not found

    Examples:
        'chapter_00_intro.mp3' -> 0
        'episode_0_pilot.mp3' -> 0
        'chapter_01.mp3' -> 1
        'ep05.mp3' -> 5
        'book_ch_03.mp3' -> 3
        '00_prologue.mp3' -> 0
        '01.mp3' -> 1
    """

    name = filename.lower()
    stem = Path(filename).stem.lower()  # Returns the stem of the file path instead of absolute path

    # Pattern 1: Explicit chapter/episode keywords
    # Matches: chapter_00, chapter00, chapter-00, chapter 00, ch_00, ch00, etc.
    patterns = [
        r'chapter[\s_-]*(\d+)',
        r'ch[\s_-]*(\d+)',
        r'episode[\s_-]*(\d+)',
        r'ep[\s_-]*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))

    # Pattern 2: Leading digits (common for sorted audio files)
    # Matches: 00_intro.mp3, 01.mp3, 001.mp3
    match = re.match(r'^(\d+)', stem)
    if match:
        return int(match.group(1))

    # Pattern 3: First sequence of digits found anywhere
    # Fallback for unusual naming
    match = re.search(r'(\d+)', stem)
    if match:
        return int(match.group(1))

    # No number found
    return None


def extract_chapter_number(filename: str) -> int | None:
    """
    Extract chapter number from audiobook filename.
    Wrapper for extract_chapter_or_episode_number.
    """
    return extract_chapter_or_episode_number(filename)


def extract_episode_number(filename: str) -> int | None:
    """
    Extract episode number from podcast filename.
    Wrapper for extract_chapter_or_episode_number.
    """
    return extract_chapter_or_episode_number(filename)


# --------- naming normalization / standardization ---------
def _derive_series_and_chapter(name: str):
    """
    Derive (series, chapter_num) from various incoming names, e.g.:
      - edison_lifeinventions_33_dyer_martin_64kb.mp3  -> ("edison_lifeinventions", 33)
      - audiobook_edison_lifeinventions_chapter_03.txt -> ("edison_lifeinventions", 3)
      - openaiwhisper_edison_lifeinventions_50_...     -> ("edison_lifeinventions", 50)
      - Romeo_and_Juliet_Act_1_64kb.mp3               -> ("romeo_and_juliet", 1)

    Rules:
      - Strip model prefixes (openaiwhisper_, wav2vec2_, audiobook_, podcast_, fasterwhisper_).
      - Treat "chapter_X", "act_X", "episode_X", "ep_X" as explicit markers:
          series = part before the keyword, number = X.
      - Otherwise, use first numeric token as chapter/episode index.
    """
    base = Path(name).stem.lower()

    # Strip common prefixes added by our own pipelines
    for p in ("openaiwhisper_", "wav2vec2_", "audiobook_", "podcast_", "fasterwhisper_"):
        if base.startswith(p):
            base = base[len(p):]

    # 1) Explicit chapter/act markers (audiobooks)
    # e.g. "romeo_and_juliet_act_1_64kb" -> series="romeo_and_juliet", num=1
    m = re.search(r"(chapter|act)[_\-\s]*(\d+)", base)
    if m:
        num = int(m.group(2))
        series = base[:m.start()].rstrip("_- ")
        return series, num

    # 2) Explicit episode/ep markers (podcasts)
    # e.g. "myshow_episode_10" -> series="myshow", num=10
    m = re.search(r"(episode|ep)[_\-\s]*(\d+)", base)
    if m:
        num = int(m.group(2))
        series = base[:m.start()].rstrip("_- ")
        return series, num

    # 3) First numeric token in underscore-split base
    #    e.g. "edison_lifeinventions_30_dyer_martin_64kb" -> ("edison_lifeinventions", 30)
    tokens = base.split("_")
    for i, tok in enumerate(tokens):
        if tok.isdigit():
            series = "_".join(tokens[:i]).rstrip("_- ")
            num = int(tok)
            return series, num

    # 4) Any number fallback
    m = re.search(r"(\d{1,3})", base)
    if m:
        num = int(m.group(1))
        series = base[:m.start()].rstrip("_- ")
        return series, num

    # 5) No number found
    return base, None


def standardized_output_name(content_type: str, audio_filename: str) -> str:
    """
    Build standardized output filename matching transcription_tasks.py:

    - Audiobooks:
        audiobook_{base_name}_chapter_{NN}.txt

    - Podcasts:
        podcast_{base_name}_episode_{NN}.txt

    where:
        base_name ~= 'series' derived from the original audio filename
        NN       = 2-digit chapter/episode number if found, else 'unknown'
    """
    series, num = _derive_series_and_chapter(audio_filename)
    series = (series or "").strip("_- ")

    num_str = f"{num:02d}" if isinstance(num, int) else "unknown"

    ctype = content_type.lower()
    if ctype == "podcast":
        suffix = "episode"
        prefix = "podcast"
    else:
        # default to audiobook naming
        suffix = "chapter"
        prefix = "audiobook"

    return f"{prefix}_{series}_{suffix}_{num_str}.txt"
