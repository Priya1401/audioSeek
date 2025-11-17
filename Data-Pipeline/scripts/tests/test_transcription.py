from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.transcription.transcription import detect_device, ensure_paths, is_valid_audio_path, \
    collect_audio_files_from_zip, run_faster_whisper, save_lines


def test_detect_device():
    device = detect_device()
     assert device == "gpu", "Forcing failure: expecting GPU but got CPU"


@patch('pathlib.Path.exists')
@patch('pathlib.Path.mkdir')
def test_ensure_paths(mock_mkdir, mock_exists):
    mock_exists.return_value = True
    zip_path = Path("test.zip")
    results_dir = Path("results")

    ensure_paths(zip_path, results_dir)
    mock_exists.assert_called()
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_is_valid_audio_path_valid():
    path = MagicMock(spec=Path)
    path.is_file.return_value = True
    path.suffix.lower.return_value = ".mp3"
    path.name.startswith.return_value = False
    path.stat.return_value.st_size = 1024

    assert is_valid_audio_path(path) == True


def test_is_valid_audio_path_invalid_extension():
    path = MagicMock(spec=Path)
    path.is_file.return_value = True
    path.suffix.lower.return_value = ".txt"

    assert is_valid_audio_path(path) == False


def test_is_valid_audio_path_junk_file():
    path = MagicMock(spec=Path)
    path.is_file.return_value = True
    path.suffix.lower.return_value = ".mp3"
    path.name.startswith.return_value = True  # Simulates "._file"

    assert is_valid_audio_path(path) == False


@patch('zipfile.ZipFile')
@patch('tempfile.TemporaryDirectory')
def test_collect_audio_files_from_zip(mock_temp_dir, mock_zip):
    mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
    mock_zip_instance = MagicMock()
    mock_zip.return_value.__enter__.return_value = mock_zip_instance

    # Mock extracted files
    mock_file = MagicMock(spec=Path)
    mock_file.is_file.return_value = True
    mock_file.suffix.lower.return_value = ".mp3"
    mock_file.name.startswith.return_value = False
    mock_file.stat.return_value.st_size = 1024

    with patch('pathlib.Path.rglob', return_value=[mock_file]):
        files = collect_audio_files_from_zip(Path("test.zip"), Path("/tmp"))
        assert len(files) == 1


# @patch('transcription.transcription.WhisperModel')
@patch('faster_whisper.WhisperModel')
def test_run_faster_whisper(mock_whisper_model):
    mock_instance = MagicMock()
    mock_instance.transcribe.return_value = (
        [MagicMock(start=0.0, end=5.0, text="Test")],
        MagicMock(language="en", duration=10.0)
    )
    mock_whisper_model.return_value = mock_instance

    runtime, segs, lines, meta = run_faster_whisper(
        Path("test.mp3"), "cpu", "base", 5, "float32"
    )

    assert segs == 1
    assert lines == ["[0.00-5.00] Test"]
    assert meta["language"] == "en"


@patch('pathlib.Path.write_text')
def test_save_lines(mock_write):
    path = Path("test.txt")
    lines = ["line1", "line2"]
    save_lines(path, lines)
    mock_write.assert_called_once_with("\n".join(lines), encoding="utf-8")
