from pathlib import Path
import math
import random

import imageio.v3 as iio
import numpy as np
from PIL import Image
from pycvt import draw_bounding_boxes, generate_sliding_windows

from cvmd import detect_with_windows
from cvmd import Yolov11Detect
from cvmd import Yolov11Segment


def make_mosaic(
    img_dir: str,
    num_images: int = 64,          # 32 或 64
    thumb_size: tuple = (480, 480),# 每张缩略图大小
    seed: int = 0,                 # 固定随机种子，方便复现
    out_path: str = "./mosaic.jpg"
):
    img_dir = Path(img_dir)

    # 收集所有图片路径
    all_images = sorted(img_dir.rglob("*.jpg"))
    total = len(all_images)
    if total == 0:
        raise RuntimeError(f"No jpg found in {img_dir}")

    # 如果图片不足 num_images，就全部用
    n = min(num_images, total)

    # 随机采样
    random.seed(seed)
    selected = random.sample(all_images, n)

    # 计算最接近正方形的网格：cols≈sqrt(n)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    W, H = thumb_size
    mosaic_w = cols * W
    mosaic_h = rows * H

    print(f"Total images: {total}")
    print(f"Selected: {n}")
    print(f"Grid: {rows} rows x {cols} cols")
    print(f"Mosaic size: {mosaic_w} x {mosaic_h}")

    # 创建大画布
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), (0, 0, 0))

    # 逐张读入并贴到大图上
    for idx, img_path in enumerate(selected):
        print(f"[{idx+1:02d}/{n:02d}] Processing {img_path}")

        # 用 imageio 读
        arr = iio.imread(img_path, mode="RGB")  # (h, w, 3)
        im = Image.fromarray(arr)

        # 统一缩放
        im = im.resize((W, H), Image.BILINEAR)

        # 计算贴图位置
        r = idx // cols
        c = idx % cols
        x = c * W
        y = r * H

        mosaic.paste(im, (x, y))

    # 保存
    mosaic.save(out_path, quality=95)
    print(f"Saved mosaic to: {Path(out_path).resolve()}")


def main():
    # img_dir = "/home/user/worksapce/cvmd/temp/coco128/images/train2017"

    # # 你可以改成 32 或 64
    # make_mosaic(
    #     img_dir=img_dir,
    #     num_images=64,          # 32 或 64
    #     thumb_size=(256, 256),
    #     seed=0,
    #     out_path="./mosaic_64.jpg"
    # )
    pass

    im=Path("/home/user/worksapce/cvmd/temp/mosaic_64.jpg")
    
    image= iio.imread(im, mode="RGB")
    windows= generate_sliding_windows(
        im_shape= image.shape[:2],
        ws=(800, 800),
        s=(200, 200),
    )
    
    model = Yolov11Segment(
        weights="/home/user/worksapce/cvmd/temp/yolo11l-seg.torchscript",
        device="cuda",
        conf=0.25,
        iou=0.45,
        imgsz=640,
        half=True,
    )
    model.load_model()
    
    detected= detect_with_windows(
        image=image,
        windows=windows,
        model= model,
        merge=True,
        merge_iou=0.001
    )
    
    print(f"Detected {detected.shape[0]} objects in total.")
    
    plot_im = draw_bounding_boxes(
        image=image,
        boxes=detected[..., :4],
        labels=[
            f"{c}-{s:.2f}"
            for c, s in zip(detected[..., 5].astype(int), detected[..., 4])
        ],
        line_width=3,
    )
    iio.imwrite("temp/runs/detect_with_windows_result.jpg", plot_im)


if __name__ == "__main__":
    main()
