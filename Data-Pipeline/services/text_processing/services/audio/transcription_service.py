from pathlib import Path
from core.config import settings
from workers.tasks import transcribe_single_chapter


class TranscriptionService:

    @staticmethod
    def transcribe_directory(folder_path: str,
                             book_name: str,
                             model_size: str = "base",
                             beam_size: int = 5,
                             compute_type: str = "float32"):

        # Handle GCS paths
        if folder_path.startswith("gs://"):
            from google.cloud import storage
            import os
            
            # Parse bucket and prefix
            # gs://bucket_name/prefix/path
            parts = folder_path.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            # Create local temp dir
            local_temp_dir = Path(f"/app/temp_downloads/{book_name}")
            if local_temp_dir.exists():
                import shutil
                shutil.rmtree(local_temp_dir)
            local_temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Download files
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Downloading files from GCS: gs://{bucket_name}/{prefix} to {local_temp_dir}")
            
            downloaded_any = False
            for blob in blobs:
                if blob.name.lower().endswith((".mp3", ".wav")):
                    # Flatten structure: just download to local_temp_dir
                    file_name = Path(blob.name).name
                    logger.info(f"Downloading {blob.name} -> {local_temp_dir / file_name}")
                    blob.download_to_filename(str(local_temp_dir / file_name))
                    downloaded_any = True
            
            if not downloaded_any:
                logger.error(f"No audio files found in GCS path: {folder_path}")
                raise ValueError(f"No audio files found in GCS path: {folder_path}")
                
            audio_dir = local_temp_dir
        else:
            audio_dir = Path(folder_path)

        # Where transcript .txt files will be stored
        transcript_dir = Path(f"/app/raw_data/transcription_results/{book_name}")
        transcript_dir.mkdir(parents=True, exist_ok=True)

        # Detect audio files
        audio_files = sorted([
            p for p in audio_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in [".mp3", ".wav"]
            and not p.name.startswith(".")
            and not p.name.startswith("._")
            and p.stat().st_size > 0
        ])

        if not audio_files:
            raise ValueError(f"No valid audio files found in: {audio_dir}")

        transcripts = []

        for idx, _ in enumerate(audio_files, start=1):
            resp = transcribe_single_chapter(
                chapter_index=idx,
                base_name=book_name,
                audio_dir=str(audio_dir),           # <-- correct audio folder
                transcript_dir=str(transcript_dir), # <-- correct transcript folder
                content_type="audiobook",
                model_size=model_size,
                beam_size=beam_size,
                compute_type=compute_type
            )
            transcripts.append(resp)

        return {"files": transcripts}
