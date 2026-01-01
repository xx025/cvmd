from pathlib import Path
import imageio.v3 as iio
from pycvt import draw_bounding_boxes, generate_sliding_windows

from cvmd import detect_with_windows
from cvmd import Yolov11Segment
from .utils import make_mosaic


def main():
    img_dir = "/home/user/worksapce/cvmd/temp/coco128/images/train2017"
    mosaic_path = "/home/user/worksapce/cvmd/temp/mosaic_64.jpg"
    
    if not Path(mosaic_path).exists():
        print("Generating mosaic...")
        make_mosaic(
            img_dir=img_dir,
            num_images=64,
            thumb_size=(256, 256),
            seed=0,
            out_path=mosaic_path
        )

    image = iio.imread(mosaic_path, mode="RGB")
    windows = generate_sliding_windows(
        im_shape=image.shape[:2],
        ws=(800, 800),
        s=(200, 200),
    )
    
    weights = "/home/user/worksapce/cvmd/temp/yolo11l-seg.torchscript"
    if not Path(weights).exists():
        print(f"Weights not found: {weights}")
        return

    model = Yolov11Segment(
        weights=weights,
        device="cuda",
        conf=0.25,
        iou=0.45,
        imgsz=640,
        half=True,
    )
    model.load_model()
    
    detected = detect_with_windows(
        image=image,
        windows=windows,
        model=model,
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
    save_dir = Path("runs/windows_test")
    save_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(save_dir / "detect_with_windows_result.jpg", plot_im)


if __name__ == "__main__":
    main()
