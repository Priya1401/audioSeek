from pathlib import Path
from trascription_tasks import transcribe_single_chapter


class TranscriptionService:

    @staticmethod
    def transcribe_directory(folder_path: str,
                             book_name: str,
                             model_size: str = "base",
                             beam_size: int = 5,
                             compute_type: str = "float32"):

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
