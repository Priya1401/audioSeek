"""
summary_tasks.py
Functions for generating summary reports and cleanup
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for even more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output to console
    ]
)

logger = logging.getLogger(__name__)


def generate_summary_report(
        base_name: str,
        content_type: str,
        output_dir: str,
        cleanup_results: bool = False
):
    """
    Collect all transcription results from JSON files and generate a summary CSV.

    Args:
        base_name: Base name from input files
        content_type: 'audiobook' or 'podcast'
        output_dir: Directory where transcripts are saved
        cleanup_results : Clean up results (statistics) of transcribed content

    Returns:
        dict with summary statistics
    """
    logger.info(f"Generating summary report for: {base_name}")

    results_dir = Path(output_dir) / "results_metadata"

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        raise FileNotFoundError(f"Results metadata directory not found: {results_dir}")

    # Find all result JSON files for this book
    unit = "episode" if content_type.lower() in {"podcast"} else "chapter"
    result_files = list(results_dir.glob(f"{base_name}_{unit}_*_result.json"))
    error_files = list(results_dir.glob(f"{base_name}_{unit}_*_error.json"))

    logger.info(f"Found {len(result_files)} successful results")
    logger.info(f"Found {len(error_files)} error results")

    if not result_files and not error_files:
        logger.error("No transcription results found")
        raise ValueError("No transcription results found")

    # Load all results
    results = []

    for result_file in result_files:
        try:
            result_data = json.loads(result_file.read_text())
            results.append(result_data)
            logger.info(f"Loaded result: {result_file.name}")
        except Exception as e:
            logger.warning(f"Could not load result file {result_file}: {str(e)}")

    for error_file in error_files:
        try:
            error_data = json.loads(error_file.read_text())
            results.append(error_data)
            logger.info(f"Loaded error: {error_file.name}")
        except Exception as e:
            logger.warning(f"Could not load error file {error_file}: {str(e)}")

    if not results:
        logger.error("Could not load any results")
        raise ValueError("Could not load any results")

    logger.info(f"Collected {len(results)} transcription results")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by chapter index
    df = df.sort_values('chapter_num')

    # Generate summary statistics
    successful = df[df['status'] == 'success']
    failed = df[df['status'] == 'failed']

    summary_stats = {
        "base_name": base_name,
        "content_type": content_type,
        "total_chapters": len(df),
        "successful": len(successful),
        "failed": len(failed),
        "total_runtime_seconds": successful['runtime_seconds'].sum() if len(successful) > 0 else 0,
        "avg_runtime_seconds": successful['runtime_seconds'].mean() if len(successful) > 0 else 0,
        "total_segments": successful['segments'].sum() if len(successful) > 0 else 0
    }

    logger.info(f"Summary Statistics:")
    logger.info(f"  Total chapters: {summary_stats['total_chapters']}")
    logger.info(f"  Successful: {summary_stats['successful']}")
    logger.info(f"  Failed: {summary_stats['failed']}")
    logger.info(f"  Total runtime: {summary_stats['total_runtime_seconds']:.2f}s")
    logger.info(f"  Average runtime: {summary_stats['avg_runtime_seconds']:.2f}s")
    logger.info(f"  Total segments: {summary_stats['total_segments']}")

    # Save CSV
    output_path = Path(output_dir)
    csv_filename = f"{content_type}_{base_name}_summary.csv"
    csv_path = output_path / csv_filename
    df.to_csv(csv_path, index=False)
    logger.info(f"Summary CSV saved: {csv_path}")

    # Print DataFrame to logs
    logger.info("\n=== Transcription Summary ===")
    logger.info("\n" + df.to_string(index=False))

    # Clean up extraction metadata (no longer needed after summary)
    try:
        extraction_file = Path(output_dir) / base_name / "transcription_metadata" / f"{base_name}_extraction.json"
        if extraction_file.exists():
            extraction_file.unlink()
            logger.info(f"Cleaned up extraction metadata: {extraction_file}")

        # Remove extraction_metadata directory if empty
        extraction_dir = Path(output_dir) / base_name / "transcription_metadata"
        if extraction_dir.exists() and not list(extraction_dir.iterdir()):
            extraction_dir.rmdir()
            logger.info(f"Removed empty extraction_metadata directory")
    except Exception as e:
        logger.warning(f"Could not clean up extraction metadata: {str(e)}")

    # Clean up results statistics from transcription if set in function
    # False by default
    if cleanup_results:
        try:
            unit = "episode" if content_type in {"podcast"} else "chapter"
            result_files = list(results_dir.glob(f"{base_name}_{unit}_*_result.json"))
            error_files = list(results_dir.glob(f"{base_name}_{unit}_*_errors.json"))

            for file in result_files + error_files:
                file.unlink()
                logger.info(f"Cleaned up : {file}")

            result_dir = Path(output_dir) / base_name / "results_metadata"
            if result_dir.exists() and not list(result_dir.iterdir()):
                result_dir.rmdir()
                logger.info(f"Removed empty extraction_metadata directory")

        except Exception as e:
            logger.warning(f"Could not clean up results metadata: {str(e)}")

    return summary_stats


if __name__ == "__main__":
    print("This is the script for zip extraction of raw audio files.")

    parser = argparse.ArgumentParser(description="Extraction Files and Folders")
    parser.add_argument("--input_dir", default="Data-Pipeline/data/raw/audios.zip",
                        help="Path to input directory containing audio files")
    parser.add_argument("--type", required=True,
                        help="Type of content: audiobook or podcast (used in file naming)")
    parser.add_argument("--outdir", default="Data-Pipeline/data/transcription_results",
                        help="Output directory for transcripts and summary CSV")
    args = parser.parse_args()

    base = str(Path(args.input_dir).stem.lower())

    generate_summary_report(base, args.type, args.outdir, cleanup_results=True)
