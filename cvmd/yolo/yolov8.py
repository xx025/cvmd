from typing import Tuple, TypedDict, Optional, Sequence, Union

import torch

from cvmd.registry import register_model
from cvmd.utils.torchutils import normalize_device

from .ops import (
    letterbox,
    non_max_suppression,
    process_mask_native,
    scale_boxes,
)


class YoloInitKwargs(TypedDict, total=False):
    model_name: Optional[str]
    weights: Union[str, bytes, bytearray]
    device: Union[str, torch.device]
    conf: float
    iou: float
    classes: Optional[Sequence[int]]
    imgsz: Union[int, Tuple[int, int]]
    half: bool
    nc: Optional[int]
    load_warm_up: bool
    mask_thr: float

@register_model("yolov8", "yolov8det", "yolov8detect")
class Yolov8Detect:

    def __init__(self, *args, **kwargs: YoloInitKwargs):
        self.weights = kwargs.get("weights", "yolov8s.torchscript")
        self.device = normalize_device(kwargs.get("device", "cpu"))
        self.conf = kwargs.get("conf", 0.25)
        self.iou = kwargs.get("iou", 0.45)
        self.classes = kwargs.get("classes", None)
        self.imgsz = kwargs.get("imgsz", 640)
        self.half = kwargs.get("half", False)
        self.nc = kwargs.get("nc", None)
        self._load_warmup = kwargs.get("load_warm_up", True)
        self._model_dtype = None

        if isinstance(self.imgsz, int):
            self.imgsz = (self.imgsz, self.imgsz)

    def load_model(self, *args, **kwds):
        self.weights = kwds.get("weights", self.weights)
        self.device = normalize_device(kwds.get("device", self.device))
        if isinstance(self.weights, (bytes, bytearray)):
            import io

            self.weights = io.BytesIO(self.weights)
        model = torch.jit.load(self.weights, map_location=self.device)
        model.eval()
        use_half = self.half and self.device.type != "cpu"
        self.model = model.half() if use_half else model.float()
        self._model_dtype = next(self.model.parameters()).dtype
        self._warmup() if self._load_warmup else None
    

    def _warmup(self, *args, **kwds):
        import numpy as np

        empty_im = np.zeros((*self.imgsz, 3), dtype=np.uint8)  # HWC
        self.__call__(empty_im)

    def __call__(self, *args, **kwds):
        """
        arguments:
            images: numpy array of shape (h,w,3) in RGB format

        returns: numpy array of shape (n,6) where each row is
            x1, y1, x2, y2, confidence, class
        """
        if not hasattr(self, "model"):
            raise RuntimeError("Model is not loaded. Call `load_model` first.")

        with torch.inference_mode():
            x = self.__pre_process__(*args, **kwds)
            pred = self.model(x)
            pred = self.__post_process__(pred, *args, **kwds)
        return pred

    def __post_process__(self, pred, *args, **kwds):
        pred = non_max_suppression(
            pred,
            self.conf,
            self.iou,
            classes=self.classes,
            nc=self.nc,
        )[0]
        pred[:, :4] = scale_boxes(self.imgsz, pred[:, :4], args[0].shape[:2]).round()
        return pred.cpu().numpy()

    def __pre_process__(self, *args, **kwds):
        im0 = args[0]
        im1 = letterbox(im0, new_shape=self.imgsz)
        x = torch.from_numpy(im1).permute(2, 0, 1).to(self.device)
        x = x.to(self._model_dtype)
        x = x.unsqueeze(0) / 255.0
        return x

@register_model("yolov8seg", "yolov8segment")
class Yolov8Segment(Yolov8Detect):

    def __init__(self, *args, **kwargs: YoloInitKwargs):
        self.mask_thr = kwargs.pop("mask_thr", 0.6)
        self.nc = kwargs.get("nc", None)
        super().__init__(*args, **kwargs)

    def __post_process__(self, pred, *args, **kwds):
        im0 = args[0]
        pred0, proto = pred

        if self.nc is None:
            self.nc = int(pred0.shape[1] - 4 - proto.shape[1])

        proto = proto[0]
        det = non_max_suppression(
            pred0, self.conf, self.iou, classes=self.classes, nc=self.nc
        )[0]
        if det is None or len(det) == 0:
            import numpy as np
            return np.zeros((0, 6), dtype=np.float32), np.zeros((0, *im0.shape[:2]), dtype=bool)

        det[:, :4] = scale_boxes(self.imgsz, det[:, :4], im0.shape[:2]).round()
        masks = process_mask_native(proto, det[:, 6:], det[:, :4], im0.shape[:2])
        masks = masks > self.mask_thr

        return det[:, :6].cpu().numpy(), masks.cpu().numpy()
