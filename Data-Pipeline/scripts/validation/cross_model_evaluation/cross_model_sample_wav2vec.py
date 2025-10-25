#!/usr/bin/env python3
"""
Wav2Vec2 Sample Transcription
-----------------------------
Transcribes sampled audio files from ZIP using facebook/wav2vec2-base-960h
and saves standardized transcripts.
"""

import os
import sys, time, zipfile, tempfile, argparse
from pathlib import Path
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
sys.path.append(os.path.abspath('../../../'))
from scripts.transcription.utils.audio_utils import ensure_paths, extract_zip_filtered, standardized_output_name


def transcribe_sample_wav2vec(zipfile_path: str, outdir_path: str, content_type: str):
    zip_path = Path(zipfile_path)
    outdir = Path(outdir_path)
    ensure_paths(zip_path, outdir)

    print("[INFO] Loading facebook/wav2vec2-base-960h ...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[INFO] Extracting {zip_path.name} → {tmp}")
        audio_files = extract_zip_filtered(zip_path, Path(tmp))
        if not audio_files:
            sys.exit("[ERROR] No valid audio files found in ZIP.")

        for audio in audio_files:
            print(f"[Wav2Vec2] Transcribing {audio.name} ...")
            start = time.time()

            # Load and preprocess audio
            waveform, sr = torchaudio.load(str(audio))
            if waveform.shape[0] > 1:  # stereo -> mono
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            waveform = waveform / waveform.abs().max()  # normalize

            # Inference
            input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            out_name = standardized_output_name(content_type, audio.name)
            out_file = outdir / out_name
            out_file.write_text(transcription.strip(), encoding="utf-8")
            print(f"[OK] Saved → {out_file} ({round(time.time() - start, 2)}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", required=True)
    parser.add_argument("--outdir", default="data/validation/cross_model_evaluation/wav2vec2")
    parser.add_argument("--type", required=True, choices=["audiobook", "podcast"])
    args = parser.parse_args()

    transcribe_sample_wav2vec(args.zipfile, args.outdir, args.type)
