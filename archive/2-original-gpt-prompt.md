Speaker ID Pt02

Other thread was getting too slow so I'm going to start here fresh with summarized context.

=====
Project Summary: Speaker ID Engine with ML training for Data Science course
Data Training: I'm taking a few sets of audio clips with labeled speaker ID data, training them in my system for voice recognition. 
Final Product: I give 30-60 seconds of sample audio from specific speakers, then I'm pairing that with pyannote.audio's speaker segmentation features to pair each anonymous SPEAKER_00, etc with the actual person during a group audio recording. 
=====

Core VScode environment I've set up so far:
=====
Core environment

Python 3.11 (your venv is using this version, not 3.13 anymore because of NumPy issues).

pip / wheel / setuptools (for package management).

Data Science stack

numpy (pinned <2.0 for compatibility)

pandas (tabular work, manifests)

scikit-learn (classification, evaluation, confusion matrix, etc.)

matplotlib / seaborn (plots for analysis/presentation, if installed via DS bundle)

Deep Learning & audio

torch (PyTorch with CUDA 13 support via your NVIDIA driver)

torchaudio (audio I/O and transforms, tied to PyTorch version)

speechbrain (ECAPA embeddings + other pretrained speech models)

pyannote.audio (diarization pipeline)

Hugging Face ecosystem

huggingface_hub (model download/auth)

python-dotenv (load your HF_TOKEN from .env)

(optional) huggingface_hub[hf_xet] → improves large model download performance (you saw the warning about this).

Audio I/O utilities

librosa (resampling, feature extraction)

soundfile (read/write .wav with proper formats)

Utilities

tqdm (progress bars)

joblib (save/load sklearn models)

👉 In short:
You now have PyTorch + CUDA, Speechbrain/pyannote for embeddings & diarization, Hugging Face hub for gated models, and the standard DS stack for training & evaluation.=====

=====

Project plan (your original draft):
=====
Short answer: yes—you can build a small, legit DS/ML project that does “who spoke when” (diarization) and “who exactly is speaking” (speaker ID) for 2–3 voices. You don’t need to train a model from scratch: use solid pretrained embeddings + a lightweight classifier, and combine that with an off-the-shelf diarization pipeline.

Do current APIs already do this?

Most STT APIs now include speaker diarization (labeling turns as Speaker 1/2/3) but not true identity recognition out of the box (i.e., mapping a voice to “Justin vs. Marla” unless you build or enroll profiles around it). Examples: AWS Transcribe, Google STT v2, and Azure Speech all support diarization. 
AWS Documentation
Google Cloud
+1
Microsoft Learn

ElevenLabs’ new Scribe ASR also supports diarization with word-level timestamps; there are some practical limits noted by users (e.g., 8-minute clips when diarization is enabled). 
ElevenLabs
scribewave.com

Datasets you can use (besides the two you found)

AMI Meeting Corpus — gold standard for diarization (multi-speaker meetings with annotations). Great for evaluating diarization error rate (DER). 
groups.inf.ed.ac.uk
+1
openslr.org

VCTK — ~100+ English speakers, clean clips; good for training/finetuning speaker ID. (There are Kaggle mirrors.) 
Kaggle
+1

Mozilla Common Voice — many speakers with metadata; handy for quick experiments. 
Kaggle

A compact project blueprint (DS/ML-first, job-relevant)

Goal & deliverable
Input: a meeting audio where 2–3 known people talk.
Output: a transcript with named speakers (“Justin: …”, “Marla: …”), plus metrics (DER for diarization, accuracy/EER for ID), and a short report.

Pipeline design

Segmentation & diarization (who spoke when?): use pyannote.audio’s pretrained pipeline to cut the audio into speaker turns and cluster them. (You can finetune later, but not required to get results.) 
GitHub
Vast AI

Speaker embeddings (voice fingerprints): use SpeechBrain ECAPA-TDNN to extract embeddings per known speaker (30–60s enrollment each) and per diarized segment. ECAPA is a strong baseline trained on VoxCeleb. 
Hugging Face
speechbrain.readthedocs.io
GitHub

Assignment (ID): map each diarized cluster (or segment) to the closest enrolled speaker via cosine similarity. Add a threshold to mark “Unknown” when no enrolled voice is close enough.

Transcription overlay (optional): run your favorite ASR (e.g., Whisper, ElevenLabs Scribe, or cloud STT) and merge the timestamps with diarization to render “who said what.” (Scribe has word-timestamps + diarization if you want a quick baseline.) 
ElevenLabs

Training/ML component (what makes this a DS project)

Supervised option: Using your dataset (e.g., VCTK + your Kaggle set), compute embeddings and train a small classifier (logistic regression/SVM) to predict speaker IDs. Evaluate with accuracy/F1 per speaker.

Verification option: Frame it as speaker verification (same/different?), compute ROC and EER on held-out pairs—this is classic in speaker ID literature and showcases proper ML evaluation. (SpeechBrain docs show how to do this with ECAPA embeddings.) 
speechbrain.readthedocs.io

Diarization metric: Report DER against AMI references for one or two test files so you can quantify the “who-spoke-when” part. 
GitHub

Evaluation you can show in class/interviews

Confusion matrix for your 2–3-speaker classifier.

ROC/EER for verification.

DER on AMI sample(s); note effect of tuning min/max speakers. 
Stack Overflow

Ablations: with/without VAD clean-up, different similarity thresholds, or using cluster-centroids vs. segment-level votes.
=====

Kaggle projects I want to start with:
=====
Pretty sure these projects contain pre-labeled data. For my goals, the label needs to be which speaker is talking, right? I'm going to run training then test to make sure it identifies the right speaker.

These were the ones I found:
https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset
Found this other one with audio from 7 labeled famous people: https://www.kaggle.com/code/alkanerturan/speakerrecognition/input

Can these 2 work as a starting point?
=====

=====

Project plan (your most recent):
=====
What you’ll do (high level)

Download & organize data

Put them like this:

data/datasets/
  kaggle50/ <speaker_id>/file.wav …
  celebs7/  <speaker_id>/file.wav …


Make sure <speaker_id> folder names are consistent and unique across sets (e.g., prefix with dataset name to avoid collisions: k50_spk01, c7_barackobama, etc.).

Normalize audio

Convert everything to mono, 16 kHz, PCM_16 (small script below).

Build a manifest

A CSV with path,label for all files (script below).

Optionally filter to speakers with ≥ N seconds of audio (e.g., ≥ 60–120s).

Extract embeddings (ECAPA) → train classifier (scikit-learn)

Train/val/test split by clip but within the same set of speakers (since your goal is recognizing known voices).

Report accuracy/F1 + confusion matrix (great for your presentation).

Name speakers in diarized audio

Run pyannote diarization on a mixed meeting file.

Embed each segment and classify → assign real names (with an “Unknown” threshold if confidence is low).

(Optional) Robustness

Add simple data augmentation (noise, reverb, mp3 compression) to reduce mismatch between Kaggle audio and your mic/meeting audio.
=====