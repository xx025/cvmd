import numpy as np
import torch
import torchvision.transforms as T

from cvmd.registry import register_model
from cvmd.yolo.ops import xywhn2xyxy
from cvmd.yolo.yolov8 import Yolov8Detect


class TorchscriptDETRBase(Yolov8Detect):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("weights", None)
        super().__init__(*args, **kwargs)
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(self.imgsz),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_model(self, *args, **kwds):
        weights = kwds.get("weights", self.weights)
        if weights is None:
            raise ValueError(
                f"{self.__class__.__name__}.load_model() requires an explicit TorchScript weights path."
            )
        super().load_model(*args, **kwds)

    def __pre_process__(self, *args, **kwds):
        im0 = args[0]
        x = self.transform(im0).unsqueeze(0).to(self.device)  # 1CHW
        x = x.to(self._model_dtype)
        return x

    def __post_process__(self, preds, *args, **kwds):
        im0 = args[0]
        boxes, scores, labels = self._decode_predictions(preds)
        boxes, scores, labels = self._filter_detections(boxes, scores, labels)
        return self._finalize_boxes(boxes, scores, labels, im0)

    def _decode_predictions(self, preds):
        raise NotImplementedError

    def _filter_detections(self, boxes, scores, labels):
        keep = scores > self.conf

        if self.classes is not None:
            classes_t = torch.as_tensor(self.classes, device=labels.device)
            keep = keep & torch.isin(labels, classes_t)

        return boxes[keep], scores[keep], labels[keep]

    def _finalize_boxes(self, boxes, scores, labels, im0):
        if boxes.numel() == 0:
            return np.zeros((0, 6), dtype=np.float32)

        h, w = im0.shape[:2]
        xyxy = xywhn2xyxy(boxes, w, h)
        xyxy[:, 0::2].clamp_(min=0, max=w)
        xyxy[:, 1::2].clamp_(min=0, max=h)

        results = torch.cat(
            [xyxy, scores.unsqueeze(-1), labels.unsqueeze(-1).float()],
            dim=-1,
        )
        return results.cpu().numpy().astype(np.float32)


@register_model("detr", "detrdet", "detrdetect")
class DETR(TorchscriptDETRBase):
    def _decode_predictions(self, preds):
        logits = preds["pred_logits"][0]
        boxes = preds["pred_boxes"][0]
        probs = logits.softmax(-1)
        scores, labels = probs[..., :-1].max(-1)
        return boxes, scores, labels
