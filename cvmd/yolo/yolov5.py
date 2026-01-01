from cvmd.registry import register_model
import numpy as np
from .v5ops import process_mask_native, non_max_suppression
from .yolov8 import Yolov8Detect, Yolov8Segment
from .ops import scale_boxes


@register_model("yolov5", "yolov5det", "yolov5detect")
class Yolov5Detect(Yolov8Detect):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_process__(self, pred, *args, **kwds):
        pred = non_max_suppression(
            pred, self.conf, self.iou, classes=self.classes
        )[0]
        pred[:, :4] = scale_boxes(self.imgsz, pred[:, :4], args[0].shape).round()
        return pred.cpu().numpy()


@register_model("yolov5seg", "yolov5segment")
class Yolov5Segment(Yolov8Segment):
    def __init__(self, *args, **kwargs):
        self.mask_thr = kwargs.pop("mask_thr", 0.6)
        self.nm = kwargs.get("nm", None)
        super().__init__(*args, **kwargs)

    def __post_process__(self, pred, *args, **kwds):
        pred, proto = pred

        proto = proto[0]

        if self.nm is None:
            self.nm = proto.shape[0]
        pred = non_max_suppression(
            pred,
            self.conf,
            self.iou,
            classes=self.classes,
            nm=self.nm,
        )[0]

        if len(pred) == 0:
            return np.zeros((0, 6), dtype=np.float32), np.zeros(
                (0, *args[0].shape[:2]), dtype=bool
            )

        pred[:, :4] = scale_boxes(self.imgsz, pred[:, :4], args[0].shape).round()
        masks = process_mask_native(proto, pred[:, 6:], pred[:, :4], args[0].shape[:2])
        masks = masks > self.mask_thr
        return pred[:, :6].cpu().numpy(), masks.cpu().numpy()
