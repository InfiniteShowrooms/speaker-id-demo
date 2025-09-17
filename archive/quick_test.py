# quick_test.py
from dotenv import load_dotenv
import os, torch
from pyannote.audio import Pipeline

load_dotenv()
token = os.getenv("HF_TOKEN")
assert token and token.startswith("hf_")

print("CUDA:", torch.cuda.is_available())

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token,
)
# Run on your short file
diar = pipeline("data/meetings/mini.wav")
print(diar)
# show first few segments
for i, ((seg, _), label) in enumerate(diar.itertracks(yield_label=True)):
    print(f"{i:02d}  {seg.start:.2f}–{seg.end:.2f}s  -> {label}")
    if i >= 5: break
