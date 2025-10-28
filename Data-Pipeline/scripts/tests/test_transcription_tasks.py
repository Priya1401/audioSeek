import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.transcription.trascription_tasks import detect_device, \
    transcribe_single_chapter


def test_detect_device():
    assert detect_device() == "cpu"


def test_transcribe_single_chapter_success():
    # Create temp directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the metadata directory
        metadata_dir = Path(temp_dir) / "transcription_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Create the extraction JSON file
        extraction_file = metadata_dir / "test_base_extraction.json"
        audio_files_data = {
            "audio_files": [{
                "index": 1,
                "original_number": 1,
                "filename": "test.mp3",
                "path": "/tmp/test.mp3",
                "size_mb": 1.0
            }]
        }

        with open(extraction_file, 'w') as f:
            json.dump(audio_files_data, f)

        # Mock WhisperModel
        with patch('faster_whisper.WhisperModel') as mock_whisper:
            mock_instance = MagicMock()
            mock_segment = MagicMock()
            mock_segment.start = 0.0
            mock_segment.end = 5.0
            mock_segment.text = "Test"

            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.duration = 10.0

            mock_instance.transcribe.return_value = ([mock_segment], mock_info)
            mock_whisper.return_value = mock_instance

            # Mock save_lines to avoid file writing issues
            with patch('scripts.transcription.trascription_tasks.save_lines'):
                result = transcribe_single_chapter(1, "test_base", "audiobook",
                                                   temp_dir)

            assert result["status"] == "success"
            assert result["chapter_num"] == 1
