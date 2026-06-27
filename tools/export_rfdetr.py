import sys
import os
from pathlib import Path

import torch


if __name__ == "__main__":
    # reference: https://github.com/roboflow/rf-detr/releases/tag/1.8.2
    repo_dir = Path(__file__).resolve().parents[2] / "work" / "rf-detr"
    sys.path.insert(0, str(repo_dir / "src"))

    from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

    variant = os.environ.get("RFDETR_VARIANT", "nano").strip().lower()
    variant_map = {
        "base": RFDETRBase,
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    if variant not in variant_map:
        raise ValueError(f"Unsupported RFDETR_VARIANT: {variant}. Choose from {sorted(variant_map)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Variant:", variant)

    model = variant_map[variant](device=device)
    weight_path = Path(model.model_config.pretrain_weights).resolve()
    dtype = torch.float16 if device == "cuda" else torch.float32

    model.optimize_for_inference(
        compile=True,
        batch_size=1,
        dtype=dtype,
    )

    ts_model = model.model.inference_model
    save_path = weight_path.with_suffix(".torchscript")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(ts_model, str(save_path))

    loaded = torch.jit.load(str(save_path), map_location=device)
    loaded.eval()
    dummy_input = torch.randn(1, 3, model.model.resolution, model.model.resolution, device=device)
    if device == "cuda":
        dummy_input = dummy_input.half()
    outputs = loaded(dummy_input)

    print(f"Weights: {weight_path}")
    print(f"Saved: {save_path}")
    print("Output type:", type(outputs).__name__)
    if isinstance(outputs, tuple):
        print("Output shapes:", [tuple(x.shape) for x in outputs])
