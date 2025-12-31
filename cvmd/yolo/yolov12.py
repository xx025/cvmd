from .yolov8 import Yolov8Detect, Yolov8Segment


class Yolov12Detect(Yolov8Detect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Yolov12Segment(Yolov8Segment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
