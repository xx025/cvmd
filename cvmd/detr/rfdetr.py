import torch

from cvmd.registry import register_model
from cvmd.detr.detr import TorchscriptDETRBase


@register_model("rfdetr", "rfdetrdetect", "rf-detr")
class RFDETR(TorchscriptDETRBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("imgsz", 384)
        kwargs.setdefault("nc", 90)
        super().__init__(*args, **kwargs)

    def _decode_predictions(self, preds):
        boxes = preds[0][0]
        logits = preds[1][0, :, :-1]
        probs = logits.sigmoid()
        scores, labels = probs.max(-1)
        return boxes, scores, labels
