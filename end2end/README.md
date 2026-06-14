# Lip-to-Text Verification Model

This repository contains a PyTorch implementation for a Cross-Modal Lip-to-Text Verification system using the GRID dataset. The goal of this model is to take a sequence of lip movement frames and a text transcript, then compute an alignment-based consistency score (0 to 1). 


## Project Structure

```text
.
├── dataset/
│   ├── grid_loader.py          # GRID Dataset PyTorch Datasets
|.  |── digit_loader.py.        # DIGIT dataset loader
│   └── preprocessing.py        # Video frame extraction & lip cropping (96x96)
├── models/
│   ├── lip_encoder.py          # Spatial-temporal video feature extractor
│   ├── text_encoder.py         # Phoneme embedding and Transformer encoder
│   ├── cross_modal_alignment.py# Cross-attention and alignment scoring
│   └── verification_model.py   # Final consistency linear classifier
├── training/
│   └── train.py                # Training loop, contrastive logic, and BCE loss
├── evaluation/
│   ├── metrics.py              # Precision, ROC AUC, Accuracy, margin
│   └── evaluate.py             # Validation loops and inferences
├── pipeline/
│   └── data_preparation.py     # End-to-end dataset preprocessing
├── utils/
│   ├── phoneme_converter.py    # Maps English transcripts to integer phoneme vocabulary
│   └── negative_sampler.py     # Generates bad transcripts for contrastive training
├── main.py                     # Training orchestrator & evaluation suite
├── inference.py                # External inference execution script 
└── README.md                   # This file
```

---

## Model Architecture
```
    [Lip Video Frames]                   [Text Transcript]
     (T x 3 x 96 x 96)                     (Phoneme Seq)
           │                                    │
           ▼                                    ▼
   ┌───────────────────┐               ┌───────────────────┐
   │   LIP ENCODER     │               │   TEXT ENCODER    │
   │ ----------------- │               │ ----------------- │
   │ 1. ResNet18       │               │ 1. Embedding Layer│
   │ 2. Transformer    │               │ 2. Transformer    │
   │    Encoder        │               │    Encoder        │
   └───────────────────┘               └───────────────────┘
           │                                    │
           ▼                                    ▼
   (T x Feature_Dim)                    (L x Feature_Dim)
           │                                    │
           └───────────────┐  ┌─────────────────┘
                           ▼  ▼
                ┌─────────────────────────┐
                │ CROSS-MODAL ALIGNMENT   │
                │ ----------------------- │
                │ Multi-Head Cross        │
                │ Attention               │
                └─────────────────────────┘
                           │
                           ▼
                ┌─────────────────────────┐
                │    VERIFICATION HEAD    │
                │ ----------------------- │
                │ 1. Feature Pooling      │
                │ 2. Concatenation        │
                │ 3. MLP + Sigmoid        │
                └─────────────────────────┘
                           │
                           ▼
                   [Alignment Score]
                      (0.0 - 1.0)
```

### Component Breakdown
1. **Lip Encoder:** Extracts visual speech fragments (visemes) from raw frames. We use a **ResNet18** backbone to learn high-level spatial features for each individual frame. The continuous stream of features is then passed through a **Transformer Encoder** to grasp the sequential temporal context (e.g., mouth opening, transitioning, closing).
2. **Text Encoder:** Converts raw transcription words into mapped integer phoneme representations. These integers pass through a dense **Embedding Layer** followed by a **Transformer Encoder** to establish linguistic structure and grammatical bounds.
3. **Cross-Modal Alignment Engine:** Translates between the visual and audio-text modalities. Utilizing **Multi-Head Cross Attention**, the phonemic sequence queries the visual viseme sequence. It identifies *where* and *when* specific vocal sounds occur within the video.
4. **Verification Head:** Condenses the aligned structural matrix through pooling layers. The combined representations are routed through a Multi-Layer Perceptron (MLP) terminating in a strictly restricted Sigmoid activation, yielding the final alignment probability between `0` (Complete Desync) and `1` (Perfect Alignment).

---

## ⚙️ Pipeline Overview

### 1. Data Preprocessing
The `GRIDDataset` loads speaker videos and alignment files.
- **Cropping & Sizing:** Videos are actively tracked utilizing MediaPipe's FaceLandmarker, dynamically cropped tightly around the lips, and resized uniformly to `96x96`.
- **Temporal extraction:** Video lengths are capped/padded to standard frame limits (e.g., 75 bounds) converting into unified PyTorch tensors.
- **Transcript Parsing:** Transcript strings are mapped dynamically via `PhonemeConverter`.

### 2. Training & Negative Sampling
The loss relies on pushing mathematically correct sequences closer and forcing explicitly mismatched texts away.
- **Negative Sampler:** To teach the model what a "mismatch" looks like, we generate a negative sample for every true pair during training by dynamically augmenting the text. Instead of relying purely on completely distinct sentences, we randomly select from 3 strategic distortions:
  - **`wrong` / `random`**: Selects a completely random unrelated sentence from the dataset pool. 
    *(Example: "bin blue at f two now" ➔ "lay red with c nine please")*
  - **`shuffled`**: Randomizes the word order of the original transcript to force learning sequential temporal positioning. 
    *(Example: "bin blue at f two now" ➔ "now at blue f bin two")*
  - **`phoneme_similar`**: Selectively replaces a specific word with another valid GRID vocabulary word (such as swapping a color target). This ensures localized visual scrutiny prevents cheating via contextual sentence lengths. 
    *(Example: "bin blue at f two now" ➔ "bin red at f two now")*
- **Losses:** We combine **Binary Cross Entropy (BCE)** against a hard classification label, supplemented by a **Contrastive Margin Loss** maximizing explicit structural margins away from mismatch traps.

### 3. Evaluation
Our runtime evaluation tracks:
- **ROC AUC:** Area under the ROC curve tracking absolute classification separation.
- **Alignment Accuracy:** Total accuracy thresholding exact temporal metric alignment limits strictly passing 50% probabilities.
- **Score Margin:** The objective separation distance mapped between Valid Ground Truth and Synthesized Invalid metrics.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have all necessary data and vision pip packages installed:
```bash
pip install torch torchvision torchaudio scikit-learn numpy mediapipe opencv-python tqdm
```

### 1. Preparing the GRID Dataset
The pipeline natively processes real media from the GRID audiovisual dataset utilizing MediaPipe to track user bounds properly. **You must prepare the data before training the model.**

1. Download the [GRID audiovisual sentence corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/).
2. Extract the speaker folders into `data/raw/` (or strictly alter `raw_data_dir` via `config.json`).
3. Point your terminal at the pipeline directory and execute the setup loop dynamically ripping frames, scaling boundaries, and extracting audio parameters seamlessly:

```bash
python pipeline/data_preparation.py
```

This procedure actively constructs scaled `96x96` mp4 chunked archives, synchronized raw wav exports, and creates structured JSON manifest files (`train_split.json`, `val_split.json`, `test_split.json`).

### 2. Running the End-to-End Pipeline
Once the `processed` data directory is populated, you can run the full pipeline encompassing training splits, batch evaluations, and validation outputs simply by running:
```bash
`torchrun --nproc_per_node=2 main.py`
```


