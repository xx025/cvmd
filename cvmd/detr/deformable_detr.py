import torch

from cvmd.detr.detr import TorchscriptDETRBase
from cvmd.registry import register_model


@register_model("deformabledetr", "deformable_detr", "deformable-detr")
class DeformableDETR(TorchscriptDETRBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("imgsz", 800)
        kwargs.setdefault("nc", 91)
        super().__init__(*args, **kwargs)

    def _decode_predictions(self, preds):
        logits = preds[0][0]
        boxes = preds[1][0]

        probs = logits.sigmoid().flatten()
        topk = min(100, probs.numel())
        scores, indices = torch.topk(probs, k=topk)
        labels = indices % logits.shape[-1]
        box_indices = torch.div(indices, logits.shape[-1], rounding_mode="floor")
        boxes = boxes[box_indices]
        return boxes, scores, labels

    def _filter_detections(self, boxes, scores, labels):
        keep = (scores > self.conf) & (labels > 0)

        if self.classes is not None:
            classes_t = torch.as_tensor(self.classes, device=labels.device)
            keep = keep & torch.isin(labels, classes_t)

        return boxes[keep], scores[keep], labels[keep]
