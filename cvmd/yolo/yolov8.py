import torch

from .ops import (
    letterbox,
    non_max_suppression,
    process_mask_native,
    scale_boxes,
)



class Yolov8Detect:

    def __init__(self, *args, **kwargs):
        self.weights = kwargs.get("weights", "yolov8s.torchscript")
        self.device = kwargs.get("device", "cpu")
        self.conf = kwargs.get("conf", 0.25)
        self.iou = kwargs.get("iou", 0.45)
        self.classes = kwargs.get("classes", None)
        self.imgsz = kwargs.get("imgsz", 640)
        self.half = kwargs.get("half", False)
        self.nc = kwargs.get("nc", None)
        self.model = self.load_model()

        if isinstance(self.imgsz, int):
            self.imgsz = (self.imgsz, self.imgsz)

    def load_model(self):
        self.model = torch.jit.load(self.weights, map_location=self.device)

    def __call__(self, *args, **kwds):
        """
        arguments:
            images: numpy array of shape (h,w,3) in BGR format

        returns: numpy array of shape (n,6) where each row is
            x1, y1, x2, y2, confidence, class
        """
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
        pred[:, :4] = scale_boxes(self.imgsz, pred[:, :4], args[0].shape).round()
        return pred.cpu().numpy()

    def __pre_process__(self, *args, **kwds):
        im0 = args[0]
        im1 = letterbox(im0, new_shape=self.imgsz)
        x = torch.from_numpy(im1).permute(2, 0, 1).to(self.device)
        x = x.half() if self.half else x.float()
        x = x.unsqueeze(0) / 255.0
        return x


class Yolov8Segment(Yolov8Detect):

    def __init__(self, *args, **kwargs):
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
            return det.cpu().numpy(), im0[:0, :, 0]

        det[:, :4] = scale_boxes(self.imgsz, det[:, :4], im0.shape).round()
        masks = process_mask_native(proto, det[:, 6:], det[:, :4], im0.shape[:2])
        masks = masks > self.mask_thr

        return det[:, :6].cpu().numpy(), masks.cpu().numpy()
