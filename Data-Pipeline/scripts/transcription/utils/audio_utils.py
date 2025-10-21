from __future__ import annotations
import re
import random
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

def sample_zip_filtered(zip_path: Path, sample_size: int, out_zip_path: Path) -> Path:
    """
    Randomly sample valid audio entries from a ZIP and write a smaller sample ZIP.
    Skips macOS junk and non-audio files.
    """
    tmp_extract = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            candidates = [
                name for name in z.namelist()
                if not name.startswith("__MACOSX/")
                and not name.split("/")[-1].startswith("._")
                and Path(name).suffix.lower() in AUDIO_EXTS
            ]
            if not candidates:
                raise ValueError("[ERROR] No valid audio files found in ZIP.")

            subset = random.sample(candidates, min(sample_size, len(candidates)))
            z.extractall(tmp_extract, members=subset)

        with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z_out:
            for name in subset:
                src = tmp_extract / name
                # flatten any subfolders in the original zip
                z_out.write(src, arcname=Path(name).name)

        return out_zip_path
    finally:
        shutil.rmtree(tmp_extract, ignore_errors=True)

# --------- naming normalization / standardization ---------
def _derive_series_and_chapter(name: str):
    """
    Derive (series, chapter_num) from various incoming names, e.g.:
      - edison_lifeinventions_33_dyer_martin_64kb.mp3  -> ("edison_lifeinventions", 33)
      - audiobook_edison_lifeinventions_chapter_03.txt -> ("edison_lifeinventions", 3)
      - openaiwhisper_edison_lifeinventions_50_...     -> ("edison_lifeinventions", 50)
    """
    base = Path(name).stem.lower()
    for p in ("openaiwhisper_", "wav2vec2_", "audiobook_", "podcast_", "fasterwhisper_"):
        if base.startswith(p):
            base = base[len(p):]

    # explicit 'chapter_XX'
    m = re.search(r"chapter[_\-\s]?(\d+)", base)
    if m:
        num = int(m.group(1))
        series = base[:m.start()].rstrip("_- ")
        return series, num

    # first numeric token
    tokens = base.split("_")
    for i, tok in enumerate(tokens):
        if tok.isdigit():
            series = "_".join(tokens[:i]).rstrip("_- ")
            num = int(tok)
            return series, num

    # any number fallback
    m = re.search(r"(\d{1,3})", base)
    if m:
        num = int(m.group(1))
        series = base[:m.start()].rstrip("_- ")
        return series, num

    # no number found
    return base, None

def standardized_output_name(content_type: str, audio_filename: str) -> str:
    """
    Build standardized output filename:
      '{type}_{series}_chapter_{NN}.txt'
    """
    series, num = _derive_series_and_chapter(audio_filename)
    num_str = f"{num:02d}" if isinstance(num, int) else "unknown"
    series = (series or "").strip("_- ")
    return f"{content_type.lower()}_{series}_chapter_{num_str}.txt"
