
import argparse
import json
import os
import sys
import logging

# Add the services path to sys.path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'text_processing'))

from utils import (
    parse_transcript,
    detect_chapters,
    chunk_text,
    collect_unique_entities
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Chunk transcript into semantic chunks with timestamps")
    parser.add_argument('--input', '-i', required=True, help='Path to transcript file (.txt)')
    parser.add_argument('--output', '-o', required=True, help='Path to output JSON file')
    parser.add_argument('--target-tokens', '-t', type=int, default=512, help='Target tokens per chunk (default: 512)')
    parser.add_argument('--overlap-tokens', '-l', type=int, default=50, help='Overlap tokens between chunks (default: 50)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not args.input.endswith('.txt'):
        logger.error("Input file must be a .txt file")
        sys.exit(1)

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            transcript = f.read()
    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        sys.exit(1)

    segments = parse_transcript(transcript)

    if not segments:
        logger.error("No valid transcript segments found in file")
        sys.exit(1)

    logger.info(f"Processing {len(segments)} segments")
    chapters = detect_chapters(segments)
    chunks = chunk_text(segments, args.target_tokens, args.overlap_tokens, chapters)
    logger.info(f"Generated {len(chunks)} chunks")

    all_entities = collect_unique_entities(chunks)

    response_data = {
        'chunks': chunks,
        'chapters': chapters,
        'entities': all_entities
    }

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2)
        logger.info(f"Chunks saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving output file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

