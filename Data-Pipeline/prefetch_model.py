import os
from faster_whisper import WhisperModel

model_name = os.environ.get("FW_MODEL", "base")
compute = os.environ.get("FW_COMPUTE", "int8")
cache_dir = os.environ.get("HF_HOME", "/opt/airflow/.cache/huggingface")

print(f"[MODEL PREFETCH] Downloading Faster-Whisper '{model_name}' to {cache_dir} ...")
WhisperModel(model_name, device="cpu", compute_type=compute, download_root=cache_dir)
print("[MODEL PREFETCH] Done.")