A-data/
  1-raw/                                   # original, unmodified sources (keep for provenance)
    1-1-datasets/                            # public/labeled corpora you train on (Kaggle, VCTK, AMI, ...)
      kaggle50/...                       # as downloaded (zip contents, any structure)
      celebs7/...                        # as downloaded
      ...                                 # other datasets
    1-2-speaker-enroll/                               # [ACTUAL PRODUCT] Raw 30-90sec enrollment clips you record for real people (AI coach could mute other mics if it's Zoom in separate rooms)
      justin/...                          # raw recordings (any format, any sr/channels)
      marla/...
      markus/...
      ...                                 # one folder per person
    1-3-client-meetings/                             # [ACTUAL PRODUCT] Raw meeting recordings to diarize & match existing /enroll/ names
      2025-09-02_team_sync/...            # meeting folder (date_title); can hold wav/mp3 and notes
      2025-09-09_client_call/...
      ...                                 # one folder per meeting

  2-processed/                              # standardized, model-ready assets
    2-1-wav16k/                               # everything converted → mono, 16 kHz, PCM16
      2-1-1-datasets/                           # normalized copies of training datasets (do NOT touch raw/)
        kaggle50/
          k50_spk01/...                   # speaker folders used for supervised training
          k50_spk02/...
          ...
        celebs7/
          c7_obama/...
          c7_beyonce/...
          ...
        ...                               
      2-1-2-speaker-enroll/                             # normalized enrollment audio for real people (kept separate)
        justin/*.wav                      # 30–90s total across 1–N clips is fine
        marla/*.wav
        markus/*.wav
        ...
      2-1-3-client-meetings/                           # normalized meetings (one wav per meeting, or chunks)
        2025-09-02_team_sync/meeting.wav  # single-file canonical source you’ll diarize
        2025-09-09_client_call/meeting.wav
        ...
    2-2-manifests/                            # CSV “tables” describing what exists and how to use it
      all.csv                             # path,label,duration_s for TRAINING DATASETS ONLY (datasets/*)
      split.csv                           # path,split (train|val|test) for the classifier training
      enroll_index.csv                    # OPTIONAL: path,person,duration_s for processed/enroll/*
      meetings_index.csv                  # OPTIONAL: meeting_id,path,duration_s for processed/meetings/*
      ...                                  # add any extra manifests you find useful (e.g., filtered lists)

B-work/                                     # caches & intermediate artifacts you can rebuild anytime
  0-
  1-dataset-ecapa-embeds/                           # clip-level embedding cache for training data (ECAPA stored as .npy files)
    npy/                                  # one .npy per clip from processed/wav16k/datasets/*
    index.csv                              # npy,label,split → drives training quickly without re-embedding
  2-speaker-enroll-ecapa/                                  # enrollment-side artifacts (kept separate from training)
    npy/                                   # OPTIONAL: per-clip enrollment embeddings (if you cache them)
      justin/*.npy
      marla/*.npy
      markus/*.npy
      ...
    ecapa_means.json                       # mean embedding per enrolled person (for naming diarized segments)
  3-client-meetings-diarization/                             # outputs from diarization + naming for meetings
    2025-09-02_team_sync/                  # one folder per meeting_id
      diarization.rttm                     # raw diarization output (who-spoke-when as RTTM)
      segments.tsv                         # start,end,cluster (SPEAKER_00, etc.)
      named_segments.tsv                   # start,end,cluster,who,cosine_sim (after enrollment matching)
      named_segments.json                  # same content as JSON
    2025-09-09_client_call/...
    ...
  transcripts/                             # OPTIONAL: ASR results you merge with diarization
    2025-09-02_team_sync/whisper.json      # word/segment timestamps for “who said what”
    ...                                    
  outputs/                                 # any presentation-friendly exports (charts, merged srt, etc.)
    2025-09-02_team_sync_named.srt         # “who: text” subtitles, for example
    metrics_summary.csv                    # DER, accuracy, etc. across meetings
    ...
  4-client-meetings-named-diary/      # where i put them after matching attempt

C-models/                                   # trained ML models + preprocessors for classifier
  logreg_ecapa.pkl                         # scikit-learn model trained on dataset embeddings
  scaler.joblib                            # StandardScaler for embeddings
  label_encoder.joblib                     # str speaker labels ↔ integer ids (classifier classes)
  ...

D-reports/                                  # evaluation artifacts to show in your write-up
  train_report.txt                         # metrics for train (sanity)
  val_report.txt                           # metrics for validation (model selection)
  test_report.txt                          # metrics for held-out test
  val_confusion_matrix.png                 # confusion matrices for slides
  test_confusion_matrix.png
  ...

X-scripts/                                  # Execution order of runnable pipeline steps (VS Code tasks point here)
  normalize_and_stage.py                   # raw → processed/wav16k for datasets/enroll/meetings
  build_manifest.py                        # build manifests/all.csv from processed/wav16k/datasets/*
  make_splits.py                           # create manifests/split.csv (no leakage)
  embed_from_manifest.py                   # make work/ecapa_embeds/npy + index.csv for training
  train_eval_classifier.py                 # train & report; save into models/ and reports/
  predict_folder.py                        # quick check: predict labels on a folder of wavs
  enroll_build_means.py                    # NEW: embed processed/enroll/* and write work/enroll/ecapa_means.json
  diarize_and_name.py                      # NEW: run pyannote on processed/meetings/*, then name via enroll means
  ...


=====

🔄 How it all fits together:
1. data/raw → download & park everything here. Never modify.
2. scripts/normalize_and_stage.py → converts audio → data/processed/wav16k/ with clean structure.
3. scripts/build_manifest.py → scans wav16k → builds all.csv.
4. scripts/make_splits.py → creates split.csv (train/val/test).
5. scripts/embed_from_manifest.py → reads manifests, extracts embeddings → work/ecapa_embeds/.
6. scripts/train_eval_classifier.py → trains model → saves to models/, reports to reports/.
7. scripts/predict_folder.py → quick sanity-check on new wavs.
8. scripts/name_diarized.py → uses pyannote diarization + trained classifier + enrollment → saves named meeting results → work/outputs/.

👉 This way:
* data/ = source of truth (raw + processed tables/audio).
* work/ = heavy caches (you can delete/rebuild if needed).
* models/ = trained ML deliverables.
* reports/ = things you show in class.
* scripts/ = reproducible pipeline.

