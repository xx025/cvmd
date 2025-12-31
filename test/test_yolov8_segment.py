from pathlib import Path
import numpy as np
from cvmd import Yolov8Segment
from pycvt import draw_bounding_boxes, overlay_masks, get_color

import imageio.v3 as iio


def main():

    Path("temp/runs").mkdir(parents=True, exist_ok=True)

    model = Yolov8Segment(
        weights="/home/user/worksapce/cvmd/temp/yolov8l-seg.torchscript",
        device="cuda",
        conf=0.4,
        iou=0.45,
        imgsz=640,
        half=True,
    )
    model.load_model()
    im = iio.imread("/home/user/worksapce/cvmd/temp/coco128/images/train2017/000000000074.jpg", mode="RGB")
    results, masks = model(im)
    print("Detection results:", results)
    bbox = results[..., :4]
    conf = results[..., 4]
    cls = results[..., 5].astype(int)

    # 将 masks

    print("Masks shape:", masks.shape)
    plot_im = overlay_masks(im, masks, alpha=0.4, color=get_color("masks"))

    plot_im = draw_bounding_boxes(
        image=plot_im,
        boxes=bbox,
        labels=[f"{c}-{s:.2f}" for c, s in zip(cls, conf)],
        line_width=3,
    )
    iio.imwrite("temp/runs/yolov8l_result.jpg", plot_im)

    # Test blank image
    blank_im = 255 * np.ones((1024, 800, 3), dtype=np.uint8)
    blank_results, blank_masks = model(blank_im)
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
