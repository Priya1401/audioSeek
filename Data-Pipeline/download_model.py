from sentence_transformers import SentenceTransformer
import os

print("Pre-downloading embedding model...")
model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name, trust_remote_code=True)
print(f"Successfully downloaded {model_name}")
