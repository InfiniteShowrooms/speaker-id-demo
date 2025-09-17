# test_env.py
from dotenv import load_dotenv
import os, torch
from speechbrain.inference import EncoderClassifier
from pyannote.audio import Pipeline

load_dotenv()
token = os.getenv("HF_TOKEN")
assert token and token.startswith("hf_") # "HF_TOKEN missing or invalid. Put it in .env as HF_TOKEN=hf_xxx"

print("CUDA available:", torch.cuda.is_available())

ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print("Embedding model ready.")

# Use explicit token AND the versioned pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token,
)
print("Diarization pipeline ready.")