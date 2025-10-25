

import argparse
import json
import os
import sys
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text chunks")
    parser.add_argument('--input', '-i', required=True, help='Path to chunks JSON file')
    parser.add_argument('--output', '-o', required=True, help='Path to output JSON file')
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2', help='Sentence transformer model (default: all-MiniLM-L6-v2)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        sys.exit(1)

    texts = [chunk['text'] for chunk in data.get('chunks', [])]
    if not texts:
        logger.error("No texts found in chunks file")
        sys.exit(1)

    logger.info(f"Loading model: {args.model}")
    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        sys.exit(1)

    logger.info(f"Generating embeddings for {len(texts)} texts")
    try:
        embeddings = model.encode(texts).tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        sys.exit(1)

    logger.info("Embeddings generated successfully")

    response_data = {
        'embeddings': embeddings,
        'count': len(embeddings)
    }

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2)
        logger.info(f"Embeddings saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving output file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()