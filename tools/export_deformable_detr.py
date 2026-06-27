import torch
from pathlib import Path

from transformers import DeformableDetrForObjectDetection


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        # Deformable DETR expects an explicit pixel_mask input.
        # For plain single-image inference without padding, a full-ones mask is enough.
        pixel_mask = torch.ones(
            pixel_values.shape[0],
            pixel_values.shape[2],
            pixel_values.shape[3],
            device=pixel_values.device,
            dtype=torch.long,
        )
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        # Export only the tensors we need for cvmd inference, which keeps the TorchScript output simple.
        return outputs.logits, outputs.pred_boxes


if __name__ == "__main__":
    # reference: https://huggingface.co/SenseTime/deformable-detr
    # Use the Transformers implementation instead of the original repository path,
    # because disable_custom_kernels=True avoids the custom deformable attention op,
    # making TorchScript export much more reliable.
    model = DeformableDetrForObjectDetection.from_pretrained(
        "SenseTime/deformable-detr",
        disable_custom_kernels=True,
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model.to(device)

    # Wrap the model so the exported graph takes only pixel_values as input.
    wrapper = ExportWrapper(model).to(device).eval()
    # Use the official DETR-style inference size as a stable export shape.
    dummy_input = torch.randn(1, 3, 800, 800, device=device)

    if device == "cuda":
        # Keep export dtype aligned with the main GPU inference path in cvmd.
        wrapper = wrapper.half()
        dummy_input = dummy_input.half()

    # strict=False is more tolerant for tracing large transformer models.
    ts_model = torch.jit.trace(wrapper, dummy_input, strict=False)

    save_path = Path(__file__).resolve().parents[1] / "temp" / "model_weights" / "deformable_detr.torchscript"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ts_model.save(str(save_path))
    print(f"Saved: {save_path}")
