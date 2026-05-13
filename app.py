"""
DermAI — Psoriasis Detector
Gradio app + MCP server for Claude Desktop.
Binary classifier: psoriasis vs not_psoriasis.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import gradio as gr

MODEL_DIR = Path(__file__).parent / "model"

# Calibrated from threshold sweep on val set — update after each training run
THRESHOLD = 0.30

DISCLAIMER = (
    "AVISO: Este resultado e apenas indicativo e NAO substitui "
    "diagnostico medico profissional. Consulte sempre um dermatologista."
)

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    with open(MODEL_DIR / "class_names.json") as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    model = efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location="cpu"))
    model.eval()
    return model, class_names


try:
    _model, _class_names = load_model()
    MODEL_READY = True
except FileNotFoundError:
    _model, _class_names = None, []
    MODEL_READY = False


def analyze_skin_lesion(image: Image.Image) -> dict:
    """
    Analyze a skin photo and return whether psoriasis is likely present.

    Args:
        image: PIL Image of the skin area (photo taken by user)
    Returns:
        dict with prediction (psoriasis/not_psoriasis), confidence, and disclaimer
    """
    if not MODEL_READY:
        return {"error": "Model not loaded. Train and copy model/best_model.pt first."}
    if image is None:
        return {"error": "No image provided."}

    tensor = IMG_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]

    pso_idx = _class_names.index("psoriasis")
    pso_prob = probs[pso_idx].item()

    return {
        "prediction": "psoriasis" if pso_prob >= THRESHOLD else "not_psoriasis",
        "psoriasis_probability": f"{pso_prob:.1%}",
        "disclaimer": DISCLAIMER,
    }


def _format_result(result: dict) -> str:
    if "error" in result:
        return f"Erro: {result['error']}"

    prob = result["psoriasis_probability"]
    if result["prediction"] == "psoriasis":
        return (
            f"## Possivelmente PSORIASE ({prob})\n\n"
            "Foram detectados padroes consistentes com psoriase.\n\n"
            "**Recomendacao:** Consulte um dermatologista para confirmacao.\n\n"
            f"---\n_{result['disclaimer']}_"
        )
    else:
        return (
            f"## Nao parece psoriase ({prob} de probabilidade)\n\n"
            "Nao foram detectados padroes tipicos de psoriase.\n\n"
            "Se tiver duvidas ou os sintomas persistirem, consulte um medico.\n\n"
            f"---\n_{result['disclaimer']}_"
        )


def gradio_analyze(image):
    return _format_result(analyze_skin_lesion(image))


with gr.Blocks(title="DermAI — Detector de Psoriase", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# DermAI -- Detector de Psoriase")
    gr.Markdown(
        "> Uso pessoal. Nao substitui diagnostico medico. "
        "Consulte sempre um dermatologista."
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                type="pil",
                label="Foto da lesao",
                sources=["upload", "webcam"],
            )
            btn = gr.Button("Analisar", variant="primary", size="lg")
        with gr.Column(scale=1):
            output = gr.Markdown(label="Resultado")

    btn.click(gradio_analyze, inputs=img_input, outputs=output)

    gr.Markdown("---")
    gr.Markdown(
        "**Modelo:** EfficientNet-B0 — classifica psoriase vs outras lesoes de pele"
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True)
