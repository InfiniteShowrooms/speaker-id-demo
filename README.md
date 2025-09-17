# Speaker Identification with ECAPA & Pyannote

## 📌 Problem
Meetings and client calls often contain multiple speakers. Standard diarization tools can segment “who spoke when,” but they don’t *name* the speakers. This project adds a speaker-ID layer so diarized meetings can be attributed to real people.

## 🌍 Impact
- Automatic speaker labeling in multi-speaker meetings  
- Enrollment support: add new people with 30–90s of audio  
- Useful for transcription, meeting analytics, and coaching tools  

## 🛠️ Solution
- **Embedding model:** [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) from SpeechBrain for speaker representations  
- **Diarization:** [pyannote.audio](https://huggingface.co/pyannote/speaker-diarization) for “who-spoke-when” segmentation  
- **Classifier:** Logistic Regression / SVM / kNN trained on ECAPA embeddings  
- **Enrollment:** New users can be enrolled with short audio samples and then matched in diarized meetings  
- **Evaluation:** Train/val/test splits with per-speaker accuracy reports  

## 🔄 Process
1. **Preprocess audio** → normalize all clips to 16 kHz mono WAV  
2. **Build manifests** → CSV files with `path,label,split` metadata  
3. **Extract embeddings** → ECAPA converts each clip into a vector (.npy)  
4. **Train classifier** → fit Logistic Regression / SVM / kNN on embeddings  
5. **Diarize meetings** → pyannote segments a conversation into speaker turns  
6. **Match speakers** → nearest-centroid cosine similarity + thresholds  
7. **Summarize** → Pandas DataFrames for readable reports, talk times, etc.  

## 📊 Results & Next Steps
- Achieved >90% accuracy on validation and test speakers  
- GPU acceleration reduced diarization of 4-minute audio from ~105s (CPU) to ~5s (GPU)  
- Short (<2s) segments remain challenging; rules and ASR could improve assignment  
- Next step: fuse ASR output for context-aware speaker continuity

---

## ⚠️ Data & Environment
- **Not included:** large Kaggle datasets (50h audio) are ignored via `.gitignore`  
- **Included:** enrollment clips and demo meeting audio (small test files)  
- Requires Python 3.10+ with:
  - `speechbrain`
  - `pyannote.audio`
  - `sklearn`
  - `pandas`, `matplotlib`
  - `torchaudio`, `librosa`
  - `tqdm`

Create and activate a virtual environment, then install:

```bash
pip install -r requirements.txt
