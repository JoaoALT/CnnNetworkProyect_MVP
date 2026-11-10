"""
Export the trained Model B (ResNet-50 fine-tuned on Oxford 102) to ONNX format.
Reads app/model/model_b_best.pt (created by train_modelo_b.py) and saves app/model/flower_model_b.onnx.

Requires: pip install onnx onnxscript onnxruntime
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torchvision.models as models

OUTPUT_ONNX = Path("app/model/flower_model_b.onnx")
INPUT_PT = Path("app/model/model_b_best.pt")


def build_model(num_classes=102, dropout=0.3):
    # weights=None: every weight must come from the trained checkpoint
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    return m


def main():
    # torch.onnx prints emojis; without this the Windows console (cp1252) crashes the export
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not INPUT_PT.exists():
        raise SystemExit(f"Missing {INPUT_PT}. Train the model first: python train_modelo_b.py")

    model = build_model()
    # strict=True: stops here if the checkpoint does not match, instead of exporting an untrained model
    model.load_state_dict(torch.load(INPUT_PT, map_location="cpu"), strict=True)
    model.eval()
    print(f"Loaded trained weights from {INPUT_PT}")

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting ONNX model to {OUTPUT_ONNX}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_ONNX),
        export_params=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
    )

    # Keep everything in a single file (the exporter may write weights to a separate .data file)
    model_proto = onnx.load(str(OUTPUT_ONNX), load_external_data=True)
    onnx.save(model_proto, str(OUTPUT_ONNX), save_as_external_data=False)
    data_file = Path(str(OUTPUT_ONNX) + ".data")
    if data_file.exists():
        data_file.unlink()

    # Check that ONNX gives the same output as PyTorch (batch of 2 = image + mirrored image, as the app uses)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUTPUT_ONNX), providers=["CPUExecutionProvider"])
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = model(x).numpy()
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.abs(expected - got).max())
    if diff > 1e-3:
        raise SystemExit(f"ONNX output differs from PyTorch (max diff {diff:.2e})")

    size_mb = OUTPUT_ONNX.stat().st_size / 1e6
    print(f"Export complete and verified: {OUTPUT_ONNX} ({size_mb:.2f} MB, max diff vs PyTorch {diff:.1e})")


if __name__ == "__main__":
    main()
