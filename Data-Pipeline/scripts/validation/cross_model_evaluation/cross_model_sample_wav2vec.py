#!/usr/bin/env python3
"""
Wav2Vec2 Sample Transcription (with memory cleanup)
---------------------------------------------------
Transcribes sampled audio files from ZIP using facebook/wav2vec2-base-960h
and saves standardized transcripts.
"""

import argparse
import gc
import os
import sys
import tempfile
import time
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
        print(f"[CLEANUP] Memory cleared {tag}", flush=True)
    except Exception as e:
        print(f"[WARN] Memory cleanup failed: {e}", flush=True)


def transcribe_sample_wav2vec(zipfile_path: str, outdir_path: str, content_type: str):
    zip_path = Path(zipfile_path)
    outdir = Path(outdir_path)
    ensure_paths(zip_path, outdir)

    cleanup_memory("(before model load)")

    print("[INFO] Loading facebook/wav2vec2-base-960h with memory optimization...", flush=True)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h",
        torch_dtype=torch.float32,
    )

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[INFO] Model loaded on device: {device}", flush=True)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            print(f"[INFO] Extracting {zip_path.name} → {tmp}", flush=True)
            audio_files = extract_zip_filtered(zip_path, Path(tmp))
            if not audio_files:
                sys.exit("[ERROR] No valid audio files found in ZIP.")

            print(f"[INFO] Found {len(audio_files)} valid audio files to process.", flush=True)
            total_files = len(audio_files)
            total_runtime = 0.0

            for idx, audio in enumerate(audio_files, start=1):
                cleanup_memory(f"(before {audio.name})")

                print(f"\n[Wav2Vec2] ({idx}/{total_files}) Transcribing {audio.name} ...", flush=True)
                start = time.time()

                waveform = None
                input_values = None
                logits = None
                predicted_ids = None

                try:
                    print(f"[INFO] Loading audio file {audio.name}...", flush=True)
                    waveform, sr = torchaudio.load(str(audio))
                    print(f"[INFO] Original sample rate: {sr}, Channels: {waveform.shape[0]}", flush=True)

                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                        print("[INFO] Converted to mono channel.", flush=True)
                    if sr != 16000:
                        print(f"[INFO] Resampling {audio.name} from {sr} Hz to 16000 Hz...", flush=True)
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        waveform = resampler(waveform)
                        del resampler
                    waveform = waveform / (waveform.abs().max() + 1e-9)

                    max_length = 16000 * 30  # 30 seconds
                    total_len = waveform.shape[1] / 16000
                    print(f"[INFO] Audio duration: {total_len:.1f}s", flush=True)

                    if waveform.shape[1] > max_length:
                        # Long audio - chunked processing
                        transcriptions = []
                        num_chunks = (waveform.shape[1] // max_length) + 1
                        print(f"[INFO] Splitting into {num_chunks} chunks (~30s each)", flush=True)

                        for i, start_idx in enumerate(range(0, waveform.shape[1], max_length), start=1):
                            end_idx = min(start_idx + max_length, waveform.shape[1])
                            print(f"[PROGRESS] {audio.name}: Processing chunk {i}/{num_chunks} ({start_idx}:{end_idx})",
                                  flush=True)
                            chunk = waveform[:, start_idx:end_idx]

                            input_values = processor(
                                chunk.squeeze(),
                                sampling_rate=16000,
                                return_tensors="pt"
                            ).input_values.to(device)

                            with torch.no_grad():
                                logits = model(input_values).logits
                                predicted_ids = torch.argmax(logits, dim=-1)
                                chunk_text = processor.batch_decode(predicted_ids)[0]
                                transcriptions.append(chunk_text.strip())

                            print(f"[PROGRESS] {audio.name}: Finished chunk {i}/{num_chunks}", flush=True)

                            # Clean up
                            if input_values is not None:
                                del input_values
                            if logits is not None:
                                del logits
                            if predicted_ids is not None:
                                del predicted_ids
                            cleanup_memory("(chunk processed)")

                        transcription = " ".join(transcriptions)
                        print(f"[INFO] Completed chunked transcription for {audio.name}", flush=True)

                    else:
                        # Short audio - direct inference
                        print(f"[INFO] Processing {audio.name} as short audio (no chunking)...", flush=True)
                        input_values = processor(
                            waveform.squeeze(),
                            sampling_rate=16000,
                            return_tensors="pt"
                        ).input_values.to(device)

                        with torch.no_grad():
                            logits = model(input_values).logits

                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = processor.batch_decode(predicted_ids)[0]
                        print(f"[INFO] Transcription complete for {audio.name}", flush=True)

                    out_name = standardized_output_name(content_type, audio.name)
                    out_file = outdir / out_name
                    out_file.write_text(transcription.strip(), encoding="utf-8")
                    elapsed = round(time.time() - start, 2)
                    total_runtime += elapsed
                    print(f"[OK] Saved → {out_file} ({elapsed}s)", flush=True)

                except Exception as e:
                    print(f"[ERROR] Failed on {audio.name}: {e}", flush=True)
                    continue

                finally:
                    # Safe cleanup
                    for var_name in ['waveform', 'input_values', 'logits', 'predicted_ids']:
                        if var_name in locals() and locals()[var_name] is not None:
                            del locals()[var_name]
                    cleanup_memory(f"(after {audio.name})")

            print(f"\n[OK] Completed transcription for all {total_files} files.", flush=True)
            print(f"[INFO] Total processing time: {round(total_runtime, 2)}s", flush=True)

    finally:
        print("[INFO] Cleaning up Wav2Vec2 model from memory...", flush=True)
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
