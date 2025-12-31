from pathlib import Path
import numpy as np
from cvmd import Yolov11Detect
from pycvt import draw_bounding_boxes

import imageio.v3 as iio


def main():

    Path("temp/runs").mkdir(parents=True, exist_ok=True)
    
    model = Yolov11Detect(
        weights="temp/yolo11l.torchscript",
        device="cuda",
        conf=0.25,
        iou=0.45,
        imgsz=640,
        half=True,
    )
    model.load_model()
    im = iio.imread(
        "temp/train2017/000000000025.jpg", mode="RGB"
    )
    results = model(im)
    print("Detection results:", results)
    bbox = results[..., :4]
    conf = results[..., 4]
    cls = results[..., 5].astype(int)
    plot_im = draw_bounding_boxes(
        image=im,
        boxes=bbox,
        labels=[f"{c}-{s:.2f}" for c, s in zip(cls, conf)],
        line_width=3,
    )
    iio.imwrite("temp/runs/yolov11_result.jpg", plot_im)

    # Test blank image
    blank_im = 255 * np.ones((1024, 800, 3), dtype=np.uint8)
    blank_results = model(blank_im)
    print("Blank image results:", blank_results)

    plot_im_blank = draw_bounding_boxes(
        image=blank_im,
        boxes=blank_results[..., :4],
        labels=[
            f"{c}-{s:.2f}"
            for c, s in zip(blank_results[..., 5].astype(int), blank_results[..., 4])
        ],
        line_width=3,
    )
    iio.imwrite("temp/runs/yolov11_blank_result.jpg", plot_im_blank)


if __name__ == "__main__":
    main()
