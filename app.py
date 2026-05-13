"""
DermAI — Psoriasis Detector
Gradio app + MCP server for Claude Desktop.
Binary classifier: psoriasis vs not_psoriasis.
"""

import json
from pathlib import Path

import threading
import numpy as np
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


_heatmap_lock = threading.Lock()


def _generate_heatmap(image: Image.Image) -> Image.Image | None:
    """Grad-CAM heatmap targeting the psoriasis class on features[-1]."""
    if not MODEL_READY:
        return None
    rgb = image.convert("RGB")
    tensor = IMG_TRANSFORM(rgb).unsqueeze(0)  # (1,3,224,224)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    target_layer = _model.features[-1]

    def _save_activation(_, __, output):
        activations.append(output.detach())

    def _save_gradient(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    # Lock prevents concurrent calls from corrupting each other's hooks/gradients
    with _heatmap_lock:
        fwd_hook = target_layer.register_forward_hook(_save_activation)
        try:
            bwd_hook = target_layer.register_full_backward_hook(_save_gradient)
            try:
                _model.eval()
                logits = _model(tensor)
                pso_idx = _class_names.index("psoriasis")
                _model.zero_grad()
                logits[0, pso_idx].backward()
            finally:
                bwd_hook.remove()
        finally:
            fwd_hook.remove()

    # Global-average-pool the gradients to get per-channel weights
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
    cam = (weights * activations[0]).sum(dim=1).squeeze(0)  # (H,W)
    cam = torch.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize CAM to original image size and apply JET colormap (pure PIL/numpy)
    w, h = rgb.size
    cam_pil = Image.fromarray(np.uint8(255 * cam)).resize((w, h), Image.BILINEAR)
    cam_arr = np.array(cam_pil, dtype=np.float32) / 255.0  # [0,1]

    # JET colormap: blue→cyan→green→yellow→red
    r = np.clip(1.5 - np.abs(4.0 * cam_arr - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * cam_arr - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * cam_arr - 1.0), 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)  # (H,W,3) float [0,1]

    orig_arr = np.array(rgb, dtype=np.float32) / 255.0
    blend = 0.55 * orig_arr + 0.45 * heatmap
    return Image.fromarray(np.uint8(255 * blend.clip(0, 1)))


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
    result_md = _format_result(analyze_skin_lesion(image))
    heatmap = _generate_heatmap(image) if image is not None else None
    return result_md, heatmap


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
            heatmap_output = gr.Image(
                type="pil",
                label="Zona suspeita (Grad-CAM)",
                visible=True,
            )

    btn.click(gradio_analyze, inputs=img_input, outputs=[output, heatmap_output])

    gr.Markdown("---")
    gr.Markdown(
        "**Modelo:** EfficientNet-B0 — classifica psoriase vs outras lesoes de pele"
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True)
