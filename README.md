# DermAI — Psoriasis Detector

Personal skin lesion classifier. Upload a photo → detects whether psoriasis is likely present.

**Model:** EfficientNet-B0, binary classifier (psoriasis / not_psoriasis)  
**App:** Gradio + MCP server (integrates with Claude Desktop)  
**Training:** Google Colab T4 GPU (free tier)  
**Deploy:** Hugging Face Spaces (free CPU tier)

> Disclaimer: personal/experimental use only. Does NOT replace medical diagnosis.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install "numpy<2"       # required — torchvision crashes with numpy 2.x
```

## Data

Two datasets combined:

| Dataset | Source | Psoriasis images |
|---------|--------|-----------------|
| AICamp Skin Conditions | Hugging Face (`notable12/AICamp-2023-Skin-Conditions-Dataset`) | ~160 |
| DermNet | Kaggle (`shubhamgoel27/dermnet`) | ~1400 |

```bash
# Download AICamp (requires HF_TOKEN in .env)
python data/download_datasets.py

# Download DermNet manually from Kaggle, extract to C:/kaggle_skin/dermnet/

# Build binary dataset
python data/build_binary_dataset.py

# Check distribution
python data/explore.py
```

## Training

**Recommended: Google Colab (free T4 GPU, ~45 min)**

1. Upload `dermAI_dataset.zip` to Google Drive root
2. Open `train/colab_train.ipynb` in Colab
3. Runtime → Change runtime type → T4 GPU
4. Run all cells
5. Download `dermAI_model/best_model.pt` + `class_names.json` from Drive
6. Copy to `model/`

**Local (CPU only, slow):**
```bash
python train/train.py
python train/evaluate.py
```

## Run

```bash
python app.py
# http://localhost:7860
# MCP endpoint: http://localhost:7860/gradio_api/mcp/sse
```

Tune `THRESHOLD` in `app.py` (default 0.30) based on the threshold sweep cell in the Colab notebook. Lower = more sensitive (fewer missed cases, more false alarms).

## Claude Desktop integration

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dermai": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:7860/gradio_api/mcp/sse"]
    }
  }
}
```

## Deploy to HF Spaces

1. Create Space at huggingface.co/new-space (Gradio, CPU free)
2. Upload: `app.py`, `requirements.txt`, `model/class_names.json`
3. Upload `model/best_model.pt` via HF web UI
4. Update MCP URL in Claude Desktop config to the Space URL
