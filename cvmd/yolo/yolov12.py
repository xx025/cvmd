from cvmd.registry import register_model
from .yolov8 import Yolov8Detect, Yolov8Segment

@register_model("yolov12", "yolov12det", "yolov12detect")
class Yolov12Detect(Yolov8Detect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register_model("yolov12seg", "yolov12segment")
class Yolov12Segment(Yolov8Segment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
