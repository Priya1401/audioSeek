import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from scripts.transcription.summary import generate_summary_report

@patch('pathlib.Path.exists')
@patch('pathlib.Path.glob')
@patch('pathlib.Path.read_text')
@patch('pandas.DataFrame.to_csv')
def test_generate_summary_report_success(mock_csv, mock_read, mock_glob, mock_exists):
    mock_exists.return_value = True
    mock_glob.return_value = [Path("result1.json"), Path("result2.json")]
    mock_read.side_effect = [
        '{"chapter_num": 1, "status": "success", "runtime_seconds": 10.0, "segments": 5}',
        '{"chapter_num": 2, "status": "success", "runtime_seconds": 15.0, "segments": 7}'
    ]

    result = generate_summary_report("test_base", "audiobook", "/tmp/output")

    assert result["total_chapters"] == 2
    assert result["successful"] == 2
    assert result["failed"] == 0
    assert result["total_runtime_seconds"] == 25.0
    mock_csv.assert_called_once()