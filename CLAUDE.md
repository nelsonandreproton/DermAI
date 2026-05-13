# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Binary skin lesion classifier: **psoriasis vs not_psoriasis**. EfficientNet-B0 backbone, fine-tuned via two-phase transfer learning. Served as a Gradio app with MCP support for Claude Desktop integration.

## Commands

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Build dataset (run in order)
python data/build_binary_dataset.py   # AICamp (C:/hf_cache) + DermNet (C:/kaggle_skin/dermnet)
python data/explore.py                # check class distribution

# Add more Kaggle data
python data/add_kaggle_images.py --src "C:/kaggle_skin/dermnet/train"

# Train locally (CPU, slow — prefer Colab)
python train/train.py

# Evaluate
python train/evaluate.py

# Run app
python app.py   # http://localhost:7860 — MCP at /gradio_api/mcp/sse
```

## Architecture

**Data pipeline:**
- `data/build_binary_dataset.py` — primary dataset builder. Reads from AICamp HF cache (`C:/hf_cache/aicamp-skin/`) and DermNet Kaggle (`C:/kaggle_skin/dermnet/`). Outputs to `data/prepared/train|val/{psoriasis,not_psoriasis}/`. Uses exact folder-name matching (DERMNET_MAP) to avoid the substring collision bugs that plagued earlier multi-class LABEL_MAP.
- `data/add_kaggle_images.py` — supplements existing prepared data. Has an EXCLUDE_FOLDERS set to block the mixed "Melanoma Skin Cancer Nevi and Moles" folder (contains both classes, cannot auto-label).
- `data/download_datasets.py` — downloads AICamp dataset from HF Hub via `snapshot_download` (not `datasets` library — hangs on Windows with long paths).

**Model (`train/train.py`, `app.py`, `train/evaluate.py`):**
- EfficientNet-B0 with custom head: `Dropout(0.3) → Linear(1280,256) → SiLU → Dropout(0.2) → Linear(256, num_classes)`
- Head accessed via `model.classifier[1].in_features` (index 1, not 0 — EfficientNet's classifier starts with a Dropout)
- Two-phase training: phase 1 frozen backbone (LR=1e-3, 15 epochs), phase 2 unfreeze last 3 feature blocks (LR=5e-5, 15 epochs)
- Class weights compensate for 6:1 imbalance (not_psoriasis >> psoriasis)
- GPU training: `train/colab_train.ipynb` — targets Google Colab T4, uses mixed precision, batch=64

**Inference (`app.py`):**
- `THRESHOLD` (default 0.55) — tune with the threshold sweep cell in the Colab notebook after each training run
- MCP tool exposed: `analyze_skin_lesion(image)` → returns prediction + probability + disclaimer
- Model loaded once at startup; `MODEL_READY` flag guards inference

**Model files (not committed, copy manually):**
- `model/best_model.pt` — weights, download from Google Drive after Colab training
- `model/class_names.json` — committed, contains `["not_psoriasis", "psoriasis"]`

## Key constraints

- **numpy<2** required — torchvision crashes with numpy 2.x
- **Windows 260-char path limit** — keep cache at `C:/hf_cache` (short path), not inside the project
- **DermNet "Melanoma Skin Cancer Nevi and Moles" folder** — always exclude, it contains mixed content
- **Domain shift** — dataset images are clinical/dermatoscopic; phone photos score lower confidence; tune THRESHOLD accordingly
- `datasets` library not used for download — hangs on Windows; use `huggingface_hub.snapshot_download` instead
