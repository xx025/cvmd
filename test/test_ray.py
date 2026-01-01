from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm
from cvmd import YoloInitKwargs
from cvmd import IMAGE_EXTS
import imageio.v3 as iio
import pandas as df

from cvmd.utils.ray_infer import InferActor, ray_infer_iter


# must can be pickled
def handler(
    task,
    model_config: Dict[str, Any],
    runs_config: Dict[str, Any],
    *args,
    **kwds,
) -> Dict[str, Any]:

    print(f"Handling task: {task}")
    print(f"Model config inside handler: {model_config}")
    print(f"Runs config inside handler: {runs_config}")
    print(f"Additional args: {args}, kwds: {kwds}")
    model = model_config["model"]
    out_dir = runs_config.get("save_dir", "runs/ray_runs")
    img_exts = IMAGE_EXTS
    folder_path = Path(task)
    save_dir = Path(runs_config.get("save_dir", out_dir)) / folder_path.name
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in folder_path.iterdir():
        if img_path.suffix.lower() not in img_exts:
            continue
        img = iio.imread(img_path.as_posix(),mode="RGB")
        pred = model(img)

        if isinstance(pred, tuple):
            pred = pred[0]  # for yolov11seg returning (pred, masks)

        cls = pred[:, 5]
        boxes = pred[:, :4]
        scores = pred[:, 4]
        results.append((img_path.name, cls, boxes, scores))
    return df.DataFrame(results, columns=["image_name", "class", "boxes", "scores"])


def main():

    data_root = "/home/user/worksapce/cvmd/temp/coco128/images/train2017"
    out_dir = "/home/user/worksapce/cvmd/runs/ray_runs"
    weights = "/home/user/worksapce/cvmd/temp/yolo11l-seg.torchscript"

    mode_config = YoloInitKwargs(
        model_name="yolov11seg",
        weights=weights,
        device="cuda",
        conf=0.25,
        iou=0.45,
        imgsz=640,
        half=True,
    )

    runs_config = {
        "save_dir": out_dir,
        "save_txt": True,
        "save_conf": True,
        "save_crop": False,
        "save_mask": True,
        "save_plot": True,
    }

    tasks = [data_root] * 100  # 这里 task 可以是 folder path，也可以是 dict

    it = ray_infer_iter(
        InferActor,
        tasks,
        actor_kwargs={
            "runs_config": runs_config,
            "model_config": mode_config,
            "handler": handler,
        },
        remote_method="infer",
        remote_args=(out_dir,),
    )

    with tqdm(total=len(tasks), desc="infer folders", unit="folder") as pbar:
        for r in it:
            pbar.update(1)


if __name__ == "__main__":
    main()
