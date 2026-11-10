"""
Script to export the trained PyTorch checkpoint (model_b_best.pt) to ONNX format.
Run this if you trained the model and want to generate the .onnx file for the app.
"""

import sys
import torch
import torchvision.models as models
import torch.nn as nn
from pathlib import Path

# Paths
CHECKPOINT_PATH = Path("checkpoints/model_b_best.pt")
OUTPUT_PATH = Path("app/model/flower_model_b.onnx")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def build_model(num_classes=102):
    base = models.resnet50(weights=None)
    in_feat = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feat, num_classes)
    )
    return base

def main():
    print("Building ResNet-50...")
    model = build_model()

    if CHECKPOINT_PATH.exists():
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found!")
        print("Exporting with randomly initialized weights (for testing the app pipeline).")
        print("To get accurate predictions, train the model in the notebook first.")

    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    print(f"Exporting to {OUTPUT_PATH}...")
    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT_PATH),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"Done! Exported ONNX model ({size_mb:.1f} MB) -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

