from pathlib import Path
import numpy as np
from pycvt import draw_bounding_boxes,overlay_masks

import imageio.v3 as iio

from cvmd import Yolov5Segment


def main():

    save_dir = Path("/home/user/worksapce/cvmd/runs/yolov5_test")

    save_dir.mkdir(parents=True, exist_ok=True)

    model = Yolov5Segment(
        weights="/home/user/worksapce/cvmd/temp/yolov5l-seg.torchscript",
        device="cuda",
        conf=0.4,
        iou=0.45,
        imgsz=640,
        half=True,
    )
    model.load_model()
    im = iio.imread(
        "/home/user/worksapce/cvmd/temp/coco128/images/train2017/000000000074.jpg",
        mode="RGB",
    )
    results ,masks= model(im)
    
    plot_im = overlay_masks(
        image=im,
        masks=masks
    )
    
    print("Detection results:", results)
    bbox = results[..., :4]
    conf = results[..., 4]
    cls = results[..., 5].astype(int)
    plot_im = draw_bounding_boxes(
        image=plot_im,
        boxes=bbox,
        labels=[f"{c}-{s:.2f}" for c, s in zip(cls, conf)],
        line_width=2,
    )
    iio.imwrite(save_dir / "yolov5lsegment_result.png", plot_im)

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
        line_width=1,
    )
    iio.imwrite(save_dir / "yolov5lsegment_blank_result.jpg", plot_im_blank)


if __name__ == "__main__":
    main()
