import torch

from cvmd.registry import register_model

from .v5ops import non_max_suppression_v5

from .yolov8 import Yolov8Detect

from .ops import scale_boxes


@register_model("yolov5", "yolov5det", "yolov5detect")
class Yolov5Detect(Yolov8Detect):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_process__(self, pred, *args, **kwds):
        pred, _ = pred
        pred = non_max_suppression_v5(
            pred,
            self.conf,
            self.iou,
            classes=self.classes,
            nc=self.nc,
        )[0]
        pred[:, :4] = scale_boxes(self.imgsz, pred[:, :4], args[0].shape).round()
        return pred.cpu().numpy()


@register_model("yolov5seg", "yolov5segment")
class Yolov5Segment:

    def __init__(self, *args, **kwargs):

        raise NotImplementedError("Yolov5Segment is not yet implemented.")
