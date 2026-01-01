from cvmd.registry import register_model
from .yolov8 import Yolov8Detect, Yolov8Segment

@register_model("yolov11", "yolov11det", "yolov11detect")
class Yolov11Detect(Yolov8Detect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register_model("yolov11seg", "yolov11segment")
class Yolov11Segment(Yolov8Segment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
