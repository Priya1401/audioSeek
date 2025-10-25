#!/usr/bin/env python3
import argparse
from pathlib import Path

import os
import sys
sys.path.append(os.path.abspath('../../../'))
from scripts.transcription.utils.audio_utils import sample_zip_filtered
from scripts.validation.cross_model_evaluation.cross_model_sample_openaiwhisper import transcribe_sample_openaiwhisper
from scripts.validation.cross_model_evaluation.cross_model_sample_wav2vec import transcribe_sample_wav2vec
from scripts.validation.cross_model_evaluation.validate_transcription import validate_models

def run_cross_model_evaluation(original_zip: str, content_type: str, sample_size: int = 3):
    zip_path = Path(original_zip)
    sample_zip_path = Path("data/raw/sample_subset.zip")

    out_path = sample_zip_filtered(zip_path, sample_size, sample_zip_path)
    print(f"[OK] Created sample ZIP → {out_path}")

    print("\n=== Running OpenAI Whisper Sample Transcription ===")
    transcribe_sample_openaiwhisper(
        str(sample_zip_path),
        "data/validation/cross_model_evaluation/openaiwhisper",
        content_type=content_type,
    )

    print("\n=== Running Wav2Vec2 Sample Transcription ===")
    transcribe_sample_wav2vec(
        str(sample_zip_path),
        "data/validation/cross_model_evaluation/wav2vec2",
        content_type=content_type,
    )

    print("\n=== Running Validation vs Faster-Whisper ===")

    zip_base = Path(original_zip).stem
    out_csv = Path(f"data/validation/cross_model_evaluation/{content_type}_{zip_base}_validation_summary.csv")

    validate_models(
        fw_dir="data/transcription_results",
        ow_dir="data/validation/cross_model_evaluation/openaiwhisper",
        w2v_dir="data/validation/cross_model_evaluation/wav2vec2",
        out_csv=out_csv,
        content_type=content_type,
    )
    print("\nCross-model evaluation complete.")

def main():
    print("This is the script for transcription of raw audio files.")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--zipfile", required=True, help="Path to the main audio ZIP file")
    # parser.add_argument("--type", required=True, choices=["audiobook", "podcast"], help="Type of dataset")
    # parser.add_argument("--sample-size", type=int, default=3, help="Number of random files to sample")
    # args = parser.parse_args()
    # run_cross_model_evaluation(args.zipfile, args.type, args.sample_size)


if __name__ == "__main__":
    main()