"""
extraction_tasks.py
Functions for extracting and preparing audio files from ZIP archives
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from scripts.transcription.utils.audio_utils import collect_audio_files_from_input_directory
from scripts.transcription.utils.audio_utils import extract_chapter_number
from scripts.transcription.utils.audio_utils import extract_episode_number

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for even more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output to console
    ]
)

logger = logging.getLogger(__name__)


def extract_and_list_audio_files(input_dir: str, output_dir: str, audio_type: str):
    """
    Extract ZIP to temporary directory and return list of valid audio files.

    Args:
        input_dir: Path to the raw audio files
        output_dir: Path to store extraction metadata
        audio_type : The type of audiofile worked on

    Returns:
        dict containing temp directory path and list of audio file info
    """
    # logger.info(f"Extracting audio files from: {zip_path}")
    #
    # zip_file = Path(zip_path)
    # zip_basename = zip_file.stem
    #
    # # Create temporary directory to unzip audio files
    # tmp_dir = Path(tempfile.mkdtemp(prefix=f'{audio_type}_{zip_basename}_'))
    # logger.info(f"Created temporary directory : {tmp_dir}")

    logger.info(f"Extracting File Meta Data - Transcription")
    input_dir = Path(input_dir)

    try:

        # Extract and collect audio files
        audio_files = collect_audio_files_from_input_directory(input_dir)
        logger.info(f"Audio files at {str(input_dir)} : {audio_files}")

        if not audio_files:
            logger.error("No valid audio files found in input directory")
            raise ValueError("No valid audio files found in input directory")

        logger.info(f" Found {len(audio_files)} valid audio files")

        # Create list of file information
        file_info_list = []
        for idx, audio_path in enumerate(audio_files, start=1):

            # Extracting chapter/episode number from filename
            if audio_type.lower() == "audiobook":
                original_number = extract_chapter_number(audio_path.name)
            else:
                original_number = extract_episode_number(audio_path.name)

            file_info = {
                "index": idx,
                "original_number": original_number,
                "filename": audio_path.name,
                "path": str(audio_path),
                "size_mb": audio_path.stat().st_size / (1024 * 1024)
            }

            file_info_list.append(file_info)
            chapter_label = original_number if original_number is not None else idx
            if audio_type.lower() == "audiobook":
                logger.info(f" Chapter {chapter_label} : {audio_path.name} ({file_info['size_mb']:.2f} MB)")
            else:
                logger.info(f" Episode {chapter_label} : {audio_path.name} ({file_info['size_mb']:.2f} MB)")

        result = {
            "input_dir": str(input_dir),
            "audio_files": file_info_list,
            "total_files": len(file_info_list)
        }

        # store in Xcom to share small pieces of data between airflow tasks via airflow metadata database
        # context['task_instance'].xcom_push(key = 'extraction_result', value = result)

        # save to JSON file
        base_name = input_dir.stem.lower()
        metadata_dir = Path(output_dir) / 'transcription_metadata'
        metadata_dir.mkdir(exist_ok=True, parents=True)

        extraction_file = metadata_dir / f"{base_name}_extraction.json"
        extraction_file.write_text(json.dumps(result, indent=2))

        logger.info(f" Extraction Meta Data  complete : {len(audio_files)} files ready for transcription")

        return {
            "base_name": base_name,
            "total_files": len(file_info_list),
            "extraction_file": str(extraction_file),
            "status": "success"
        }

    except PermissionError as pe:
        error_msg = str(pe)
        if 'input' in error_msg.lower() or str(input_dir) in error_msg:
            logger.error(f"Permission denied reading input directory : {str(pe)}")
            raise PermissionError(f"Cannot read from input directory '{input_dir}' : {str(pe)}")

        else:
            logger.error(f"Permission denied writing to output directory: {str(pe)}")
            raise PermissionError(f"Cannot write to output directory '{output_dir}': {str(pe)}")

    except OSError as oe:
        logger.error(f"File System Error During Extracting Meta Data: {str(oe)}")
        raise OSError(f"Error Accessing File System : {str(oe)}")

    except ValueError as ve:
        logger.error(f"Validation Error : {str(ve)}")
        raise

    except Exception as e:
        logger.error(f"Unexpected Error During extraction : {str(e)}")
        raise


if __name__ == "__main__":
    print("This is the script for zip extraction of raw audio files.")

    parser = argparse.ArgumentParser(description="Extraction Files and Folders")
    parser.add_argument("--inputdir", default="Data-Pipeline/data/raw/audios.zip",
                        help="Path to input directory containing audio files")
    parser.add_argument("--type", required=True,
                        help="Type of content: audiobook or podcast (used in file naming)")
    parser.add_argument("--outdir", default="Data-Pipeline/data/transcription_results",
                        help="Output directory for transcripts and summary CSV")
    args = parser.parse_args()

    out_directory = args.outdir + "/" + str(Path(args.inputdir).stem.lower())

    extract_and_list_audio_files(args.inputdir, out_directory, args.type)
