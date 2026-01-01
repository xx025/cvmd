from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm
from cvmd import YoloInitKwargs
from cvmd import IMAGE_EXTS
import imageio.v3 as iio
import pandas as df

from cvmd.utils.ray_infer import InferActor, ray_infer_iter
from pycvt import get_color, save_yolo_annotations, draw_bounding_boxes, xyxy2xywhn


# must can be pickled
def handler(
    task,
    model_config: Dict[str, Any],
    runs_config: Dict[str, Any],
    *args,
    **kwds,
) -> Dict[str, Any]:

    # print(f"Handling task: {task}")
    # print(f"Model config inside handler: {model_config}")
    # print(f"Runs config inside handler: {runs_config}")
    # print(f"Additional args: {args}, kwds: {kwds}")

    model = model_config["model"]
    out_dir = runs_config.get("save_dir", "runs/ray_runs_default")
    img_exts = IMAGE_EXTS
    folder_path = Path(task)
    save_dir = Path(runs_config.get("save_dir", out_dir)) / folder_path.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取当前进程ID
    import os

    pid = os.getpid()

    save_dir = save_dir / f"pid_{pid}"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_plot = runs_config.get("save_plot", False)
    save_plot_dir = save_dir / "plots"
    if save_plot:
        save_plot_dir.mkdir(parents=True, exist_ok=True)

    save_txt = runs_config.get("save_txt", False)
    save_txt_dir = save_dir / "labels"
    if save_txt:
        save_txt_dir.mkdir(parents=True, exist_ok=True)

    save_source = runs_config.get("save_source", False)
    save_source_dir = save_dir / "source"
    if save_source:
        save_source_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in folder_path.iterdir():
        if img_path.suffix.lower() not in img_exts:
            continue
        img = iio.imread(img_path.as_posix(), mode="RGB")
        pred = model(img)
        if isinstance(pred, tuple):
            pred = pred[0]  # for yolov11seg returning (pred, masks)
        cls = pred[:, 5]
        boxes = pred[:, :4]
        scores = pred[:, 4]
        
        if len(boxes) == 0:
            results.append((img_path.name, cls, boxes, scores))
            continue

        if save_txt:
            xywhn_boxes = xyxy2xywhn(boxes, img.shape[1], img.shape[0])
            save_yolo_annotations(
                file_path=save_txt_dir / f"{img_path.stem}.txt",
                cls=cls,
                bboxes=xywhn_boxes,
            )
        if save_plot:
            plot_img = draw_bounding_boxes(
                image=img,
                boxes=boxes,
                labels=[f"{int(c)}_{s:.2f}" for c, s in zip(cls, scores)],
                line_width=2
            )
            save_path = save_plot_dir / img_path.name
            save_path = save_path.with_suffix(".jpg")
            iio.imwrite(save_path.as_posix(), plot_img, quality=95)
        if save_source:
            source_path = save_source_dir / img_path.name
            if not source_path.exists():
                os.link(img_path.as_posix(), source_path.as_posix())

        results.append((img_path.name, cls, boxes, scores))
    return {"folder": folder_path.name, "results": results}


def main():

    data_root = "/home/user/worksapce/cvmd/temp/coco128/images/train2017"

    mode_config = YoloInitKwargs(
        model_name="yolov11seg",
        weights="/home/user/worksapce/cvmd/temp/yolo11l-seg.torchscript",
        device="cuda",
        conf=0.25,
        iou=0.9,
        imgsz=640,
        half=True,
        classes=list(range(40)),
    )

    runs_config = {
        "save_dir": "/home/user/worksapce/cvmd/runs/ray_runs3",
        "save_txt": True,
        "save_plot": True,
        "save_source": True,
    }

    tasks = [data_root] * 100  # 这里 task 可以是 folder path，也可以是 dict

    # ============================================
    # Start Ray inference
    # ============================================

    it = ray_infer_iter(
        InferActor,
        tasks,
        actor_kwargs={
            "runs_config": runs_config,
            "model_config": mode_config,
            "handler": handler,
        },
        remote_method="infer",
        # remote_args=(out_dir,), # if needed, may be use runs_config or model_config
    )

    # ============================================
    # Collect results （not ordered）
    # ============================================

    with tqdm(total=len(tasks), desc="infer folders", unit="folder") as pbar:
        for r in it:
            pbar.update(1)


if __name__ == "__main__":
    main()
