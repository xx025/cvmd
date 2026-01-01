import pytest
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from pycvt import draw_bounding_boxes, overlay_masks
from cvmd import Yolov5Detect, Yolov5Segment, Yolov8Segment, Yolov11Detect, Yolov11Segment, DETR

def run_detection_test(model_class, weights, sample_image, blank_image, save_dir, name, imgsz=640):
    if not Path(weights).exists():
        pytest.skip(f"Weights not found: {weights}")
    
    model = model_class(
        weights=weights,
        device="cuda",
        conf=0.25,
        iou=0.45,
        imgsz=imgsz,
        half=True,
    )
    model.load_model()
    
    # Test real image
    results = model(sample_image)
    bbox = results[..., :4]
    conf = results[..., 4]
    cls = results[..., 5].astype(int)
    plot_im = draw_bounding_boxes(
        image=sample_image,
        boxes=bbox,
        labels=[f"{c}-{s:.2f}" for c, s in zip(cls, conf)],
        line_width=2,
    )
    iio.imwrite(save_dir / f"{name}_result.jpg", plot_im)
    
    # Test blank image
    blank_results = model(blank_image)
    assert len(blank_results) >= 0

def run_segmentation_test(model_class, weights, sample_image, blank_image, save_dir, name):
    if not Path(weights).exists():
        pytest.skip(f"Weights not found: {weights}")
        
    model = model_class(
        weights=weights,
        device="cuda",
        conf=0.25,
        iou=0.45,
        imgsz=640,
        half=True,
    )
    model.load_model()
    
    # Test real image
    results, masks = model(sample_image)
    plot_im = overlay_masks(image=sample_image, masks=masks)
    
    bbox = results[..., :4]
    conf = results[..., 4]
    cls = results[..., 5].astype(int)
    plot_im = draw_bounding_boxes(
        image=plot_im,
        boxes=bbox,
        labels=[f"{c}-{s:.2f}" for c, s in zip(cls, conf)],
        line_width=2,
    )
    iio.imwrite(save_dir / f"{name}_seg_result.jpg", plot_im)
    
    # Test blank image
    blank_results, blank_masks = model(blank_image)
    assert len(blank_results) >= 0

@pytest.mark.parametrize("model_info", [
    (Yolov5Detect, "temp/model_weights/yolov5l.torchscript", "yolov5l_detect", 640),
    (Yolov11Detect, "temp/model_weights/yolo11l.torchscript", "yolov11l_detect", 640),
    (DETR, "temp/model_weights/detr_resnet50.torchscript", "detr_resnet50", 800),
])
def test_detection(model_info, sample_image, blank_image, save_dir):
    model_class, weights, name, imgsz = model_info
    run_detection_test(model_class, weights, sample_image, blank_image, save_dir, name, imgsz=imgsz)

@pytest.mark.parametrize("model_info", [
    (Yolov5Segment, "temp/model_weights/yolov5l-seg.torchscript", "yolov5l_seg"),
    (Yolov8Segment, "temp/model_weights/yolov8l-seg.torchscript", "yolov8l_seg"),
    (Yolov11Segment, "temp/model_weights/yolo11l-seg.torchscript", "yolov11l_seg"),
])
def test_segmentation(model_info, sample_image, blank_image, save_dir):
    model_class, weights, name = model_info
    run_segmentation_test(model_class, weights, sample_image, blank_image, save_dir, name)
