
"""
sanity_checks.py
Sanity check functions for validating inputs before transcription
"""
import sys
import argparse
import logging
from pathlib import Path
import zipfile

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for even more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output to console
    ]
)

logger = logging.getLogger(__name__)



def validate_input_directory(input_dir:str):
    """
    Ensure input directory exists

    Args:
        input_dir: Path to input directory

    Returns:
        dict with directory info
    """
    logger.info(f"Validating input directory: {input_dir}")

    in_path = Path(input_dir)

    try:
        # Check if input folder exists
        if not in_path.is_dir():  # Check if path does not exist and is not folder
           logger.error(f"Directory Does Not Exist : {in_path}")
           raise ValueError(f"Directory does not exist or is not a directory: {in_path}")

    except PermissionError:
        logger.error(f"No Permission To Access Directory: {in_path}")
        raise

    except OSError as e:
        logger.error(f"Error Accessing Provided Input Directory : {in_path} - {str(e)}")
        raise

    except Exception as ex:
        logger.error(f"Error Validating Input Directory : {str(ex)}")
        raise


    return {
        "input_dir": input_dir,
        "exists": True,
        "writable": True,
        "status": "valid"
    }


def validate_output_directory(output_dir:str):
    """
    Ensure output directory exists or can be created.

    Args:
        output_dir: Path to output directory

    Returns:
        dict with directory info
    """
    logger.info(f"Validating output directory: {output_dir}")

    out_path = Path(output_dir)

    try:
        # Create directory if it does not exist
        out_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir}")

        #Check write permissions with sample file
        test_file = out_path / ".test_write"
        test_file.touch()   # creates the file, raises Permission Error if write access is missing
        test_file.unlink()  # Delete file after check
        logger.info("Output directory is writable")

    except PermissionError as e:
        logger.error(f"Missing Write Permission for output directory {output_dir}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error with output directory {output_dir} : {str(e)}")
        raise

    return {
        "output_dir": output_dir,
        "exists": True,
        "writable": True,
        "status": "valid"
    }


def validate_content_type(content_type: str):
    """
    Validate that content type is either 'audiobook' or 'podcast'.

    Args:
        content_type: Type of content

    Returns:
        dict with validated type

    Raises:
        ValueError: If content type is invalid
    """
    logger.info(f"Validating content type: {content_type}")

    type_name = content_type.strip().lower()

    if type_name not in {"audiobook", "podcast"}:
        logger.error(f"invalid content type : {content_type}")
        raise ValueError("Content type must be either 'audiobook' or 'podcast'")

    logger.info("Valid Content Type")

    if type_name.lower() == "audiobook":
        split_type = "chapter"
    else:
        split_type = "episode"

    return {
        "content_type" : type_name,
        "split_type" : split_type,
        "status" : "valid"
    }

if __name__ == "__main__":
    print("This is the script for transcription of raw audio files.")

    parser = argparse.ArgumentParser(description="Sanity_Checks for Files and Folders")
    parser.add_argument("--inputdir", default="Data-Pipeline/data/raw/audios.zip",
                        help="Path to directory containing audio files")
    parser.add_argument("--type", required=True,
                        help="Type of content: audiobook or podcast (used in file naming)")
    parser.add_argument("--outdir", default="Data-Pipeline/data/transcription_results",
                        help="Output directory for transcripts and summary CSV")
    args = parser.parse_args()

    validate_input_directory(args.inputdir)
    out_directory = args.outdir + "/"+ str(Path(args.inputdir).stem.lower())
    validate_output_directory(out_directory)
    validate_content_type(args.type)



