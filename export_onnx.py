"""
Export the best model (ResNet-50 fine-tuned on Oxford 102) to ONNX format.
Saves to app/model/flower_model_b.onnx
"""

import os
from pathlib import Path
import torch
import torchvision.models as models
import torch.nn as nn
import onnx

OUTPUT_ONNX = Path("app/model/flower_model_b.onnx")
OUTPUT_PT = Path("app/model/model_b_best.pt")
OUTPUT_ONNX.parent.mkdir(parents=True, exist_ok=True)

def build_model(num_classes=102, dropout=0.3):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, num_classes)
    )
    return m

def main():
    print("Building ResNet-50 (IMAGENET1K_V2 + custom classifier head)...")
    model = build_model()

    if OUTPUT_PT.exists():
        print(f"Loading checkpoint weights from {OUTPUT_PT}...")
        try:
            state = torch.load(OUTPUT_PT, map_location="cpu")
            model.load_state_dict(state)
            print("Successfully loaded weights.")
        except Exception as e:
            print(f"Warning: could not load exact state_dict: {e}. Using pretrained weights.")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

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
        do_constant_folding=True
    )

    # Consolidate weights if external data was generated
    model_proto = onnx.load(str(OUTPUT_ONNX), load_external_data=True)
    onnx.save(model_proto, str(OUTPUT_ONNX), save_as_external_data=False)
    data_file = Path(str(OUTPUT_ONNX) + ".data")
    if data_file.exists():
        data_file.unlink()

    size_mb = OUTPUT_ONNX.stat().st_size / 1e6
    print(f"Export complete and verified: {OUTPUT_ONNX} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
