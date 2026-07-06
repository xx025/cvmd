from typing import Optional, Sequence, Tuple, TypedDict, Union

import torch


class ModelInitKwargs(TypedDict, total=False):
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


YoloInitKwargs = ModelInitKwargs
