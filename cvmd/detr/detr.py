import torch
from cvmd.registry import register_model
from cvmd.yolo.yolov8 import Yolov8Detect
import numpy as np
import torchvision.transforms as T
from cvmd.yolo.ops import xywhn2xyxy


@register_model("detr", "detrdetect", "DETR")
class DETR(Yolov8Detect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(self.imgsz),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __pre_process__(self, *args, **kwds):
        im0 = args[0]
        x = self.transform(im0).unsqueeze(0).to(self.device)  # 1CHW
        x = x.to(self._model_dtype)
        return x

    def __post_process__(self, preds, *args, **kwds):
        im0 = args[0]
        logits = preds["pred_logits"][0]
        boxes = preds["pred_boxes"][0]
        probs = logits.softmax(-1)
        scores, labels = probs[..., :-1].max(-1)

        keep = scores > self.conf
        
        if self.classes is not None:
            classes_t = torch.as_tensor(self.classes, device=labels.device)
            keep = keep & torch.isin(labels, classes_t)
                
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        if boxes.numel() == 0:
            return np.zeros((0, 6), dtype=np.float32)

        h, w = im0.shape[:2]
        xyxy = xywhn2xyxy(boxes, w, h)  # alternative way

        xyxy[:, 0::2].clamp_(min=0, max=w)
        xyxy[:, 1::2].clamp_(min=0, max=h)

        # Concatenate to (x1, y1, x2, y2, conf, cls)
        results = torch.cat(
            [xyxy, scores.unsqueeze(-1), labels.unsqueeze(-1).float()], dim=-1
        )

        return results.cpu().numpy().astype(np.float32)
