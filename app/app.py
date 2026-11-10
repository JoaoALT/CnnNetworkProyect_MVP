"""
Flower Recognition MVP – Oxford 102 Category Flower Dataset
Supports both PyTorch (.pt) and ONNX (.onnx) model formats, with automatic fallback.
"""

import os
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
IMG_SIZE = 224
RESIZE_SHORT = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TOP_K = 5

# Calibración y umbral: los genera train_modelo_b.py en model/config.json
_cfg_path = os.path.join(MODEL_DIR, "config.json")
CONFIG = {}
if os.path.exists(_cfg_path):
    with open(_cfg_path, encoding="utf-8") as f:
        CONFIG = json.load(f)
TEMPERATURE = float(CONFIG.get("temperature", 1.0))      # T < 1 corrige la subconfianza del label smoothing
CONF_THRESHOLD = float(CONFIG.get("conf_threshold", 0.5))  # por debajo: "no estoy seguro"
TTA_FLIP = bool(CONFIG.get("tta_flip", True))             # promediar con la imagen reflejada

CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris",
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_numpy(image: Image.Image) -> np.ndarray:
    """Preprocess image to match notebook's val_tf: resize short side -> center crop -> normalize."""
    image = image.convert("RGB")
    w, h = image.size
    scale = RESIZE_SHORT / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)

    w, h = image.size
    left = (w - IMG_SIZE) // 2
    top = (h - IMG_SIZE) // 2
    image = image.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC to CHW
    return arr[np.newaxis]

# ---------------------------------------------------------------------------
# Backend Engine (ONNX or PyTorch)
# ---------------------------------------------------------------------------

class ModelRunner:
    def __init__(self):
        self.engine = None
        self.backend = None
        self._init_model()

    def _init_model(self):
        onnx_file = os.path.join(MODEL_DIR, "flower_model_b.onnx")
        pt_file = os.path.join(MODEL_DIR, "model_b_best.pt")

        # 1. Try ONNX Runtime
        if os.path.exists(onnx_file):
            try:
                import onnxruntime as ort
                self.engine = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
                self.backend = "onnx"
                print(f"[INFO] Loaded ONNX model from {onnx_file}")
                return
            except Exception as e:
                print(f"[WARN] Failed to load ONNX: {e}")

        # 2. Try PyTorch
        try:
            import torch
            import torchvision.models as models
            import torch.nn as nn

            model = models.resnet50(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            )

            # Sin pesos entrenados el modelo respondería al azar: mejor fallar que dar predicciones falsas
            state_dict = torch.load(pt_file, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"[INFO] Loaded PyTorch checkpoint from {pt_file}")

            model.eval()
            self.engine = model
            self.backend = "pytorch"
            return
        except Exception as e:
            print(f"[WARN] Failed to load PyTorch: {e}")

        raise RuntimeError("No model backend available! Please supply flower_model_b.onnx or model_b_best.pt")

    def _logits(self, batch: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            input_name = self.engine.get_inputs()[0].name
            return self.engine.run(None, {input_name: batch})[0]
        import torch
        with torch.no_grad():
            return self.engine(torch.from_numpy(batch)).numpy()

    def predict(self, tensor_np: np.ndarray) -> np.ndarray:
        """Probabilidades calibradas: softmax(logits / T), promediadas con la imagen reflejada (TTA)."""
        batch = tensor_np
        if TTA_FLIP:
            batch = np.concatenate([tensor_np, np.ascontiguousarray(tensor_np[..., ::-1])])
        logits = self._logits(batch) / TEMPERATURE
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.mean(axis=0)

runner = None

def get_runner():
    global runner
    if runner is None:
        runner = ModelRunner()
    return runner

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Invalid image file format"}), 400

    try:
        tensor = preprocess_numpy(image)
        probs = get_runner().predict(tensor)

        top_idx = probs.argsort()[::-1][:TOP_K]
        results = [
            {"class": CLASS_NAMES[i], "probability": f"{probs[i] * 100:.1f}%"}
            for i in top_idx
        ]
        return jsonify({
            "predictions": results,
            "confident": bool(probs[top_idx[0]] >= CONF_THRESHOLD),
            "threshold": CONF_THRESHOLD,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "backend": get_runner().backend,
                    "temperature": TEMPERATURE, "tta_flip": TTA_FLIP, "conf_threshold": CONF_THRESHOLD})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
