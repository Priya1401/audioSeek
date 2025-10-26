#!/usr/bin/env python3
"""
Wav2Vec2 Sample Transcription (with memory cleanup)
---------------------------------------------------
Transcribes sampled audio files from ZIP using facebook/wav2vec2-base-960h
and saves standardized transcripts.
"""

import os
import sys, time, zipfile, tempfile, argparse, gc
from pathlib import Path
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

sys.path.append(os.path.abspath('../../../'))
from scripts.transcription.utils.audio_utils import ensure_paths, extract_zip_filtered, standardized_output_name


def cleanup_memory(tag=""):
    """Force Python and Torch to release as much memory as possible."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        print(f"[CLEANUP] Memory cleared {tag}")
    except Exception as e:
        print(f"[WARN] Memory cleanup failed: {e}")


def transcribe_sample_wav2vec(zipfile_path: str, outdir_path: str, content_type: str):
    zip_path = Path(zipfile_path)
    outdir = Path(outdir_path)
    ensure_paths(zip_path, outdir)

    cleanup_memory("(before model load)")

    print("[INFO] Loading facebook/wav2vec2-base-960h with memory optimization...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Load model with memory-efficient settings
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h",
        torch_dtype=torch.float32,
    )

    # Set to eval mode to save memory (disables dropout, etc.)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            print(f"[INFO] Extracting {zip_path.name} → {tmp}")
            audio_files = extract_zip_filtered(zip_path, Path(tmp))
            if not audio_files:
                sys.exit("[ERROR] No valid audio files found in ZIP.")

            for audio in audio_files:
                cleanup_memory(f"(before {audio.name})")

                print(f"[Wav2Vec2] Transcribing {audio.name} ...")
                start = time.time()

                # Initialize variables that will be deleted
                waveform = None
                input_values = None
                logits = None
                predicted_ids = None

                try:
                    # Load and preprocess audio
                    waveform, sr = torchaudio.load(str(audio))
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        waveform = resampler(waveform)
                        del resampler  # Clean up resampler
                    waveform = waveform / (waveform.abs().max() + 1e-9)

                    # Process in chunks if audio is long (> 30 seconds)
                    max_length = 16000 * 30  # 30 seconds
                    if waveform.shape[1] > max_length:
                        # Process long audio in chunks
                        transcriptions = []
                        for i in range(0, waveform.shape[1], max_length):
                            chunk = waveform[:, i:i + max_length]

                            input_values = processor(
                                chunk.squeeze(),
                                sampling_rate=16000,
                                return_tensors="pt"
                            ).input_values.to(device)

                            with torch.no_grad():
                                logits = model(input_values).logits
                                predicted_ids = torch.argmax(logits, dim=-1)
                                chunk_text = processor.batch_decode(predicted_ids)[0]
                                transcriptions.append(chunk_text)

                            # Clean up chunk tensors
                            if input_values is not None:
                                del input_values
                            if logits is not None:
                                del logits
                            if predicted_ids is not None:
                                del predicted_ids
                            cleanup_memory("(chunk processed)")

                        transcription = " ".join(transcriptions)
                    else:
                        # Process normally for short audio
                        input_values = processor(
                            waveform.squeeze(),
                            sampling_rate=16000,
                            return_tensors="pt"
                        ).input_values.to(device)

                        with torch.no_grad():
                            logits = model(input_values).logits

                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = processor.batch_decode(predicted_ids)[0]

                    out_name = standardized_output_name(content_type, audio.name)
                    out_file = outdir / out_name
                    out_file.write_text(transcription.strip(), encoding="utf-8")
                    print(f"[OK] Saved → {out_file} ({round(time.time() - start, 2)}s)")

                except Exception as e:
                    print(f"[ERROR] Failed on {audio.name}: {e}")
                    continue

                finally:
                    # Safe cleanup - check if variables exist before deleting
                    for var_name in ['waveform', 'input_values', 'logits', 'predicted_ids']:
                        if var_name in locals() and locals()[var_name] is not None:
                            del locals()[var_name]
                    cleanup_memory(f"(after {audio.name})")

    finally:
        # Clean up model and processor at the end
        print("[INFO] Cleaning up Wav2Vec2 model from memory...")
        del model
        del processor
        cleanup_memory("(final cleanup)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", required=True)
    parser.add_argument("--outdir", default="data/validation/cross_model_evaluation/wav2vec2")
    parser.add_argument("--type", required=True, choices=["audiobook", "podcast"])
    args = parser.parse_args()

    transcribe_sample_wav2vec(args.zipfile, args.outdir, args.type)