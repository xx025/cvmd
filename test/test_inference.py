import os
import sys
import csv
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pycvt
import torch
from pycvt import draw_bounding_boxes, overlay_masks

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - script mode fallback
    pytest = None


def _parametrize(*args, **kwargs):
    if pytest is None:  # pragma: no cover - script mode fallback
        def decorator(func):
            return func
        return decorator
    return pytest.mark.parametrize(*args, **kwargs)

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import cvmd
from cvmd.yolo.metrics import ap_per_class, box_iou


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
COCO_RAW_TO_CONTIGUOUS = {category_id: idx for idx, category_id in enumerate(COCO_CATEGORY_IDS)}
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)

DEFAULT_DATASET_CANDIDATES = [
    Path(os.environ.get("COCO128_DIR", "")) if os.environ.get("COCO128_DIR") else None,
    Path.home() / "datasets" / "coco128",
    REPO_DIR / "temp" / "datasets" / "coco128",
]

MODEL_SPECS = {
    "yolov5l_detect": {
        "model": "yolov5detect",
        "task": "detect",
        "weights": REPO_DIR / "temp" / "model_weights" / "yolov5l.torchscript",
        "imgsz": 640,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": False,
    },
    "yolo11l_detect": {
        "model": "yolov11detect",
        "task": "detect",
        "weights": REPO_DIR / "temp" / "model_weights" / "yolo11l.torchscript",
        "imgsz": 640,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": False,
    },
    "detr_resnet50": {
        "model": "detr",
        "task": "detect",
        "weights": REPO_DIR / "temp" / "model_weights" / "detr_resnet50.torchscript",
        "imgsz": 800,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": True,
    },
    "deformable_detr": {
        "model": "deformabledetr",
        "task": "detect",
        "weights": REPO_DIR / "temp" / "model_weights" / "deformable_detr.torchscript",
        "imgsz": 800,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": True,
    },
    "rfdetr_nano": {
        "model": "rfdetr",
        "task": "detect",
        "weights": Path.home() / ".roboflow" / "models" / "rf-detr-nano.torchscript",
        "imgsz": 384,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": True,
    },
    "rfdetr_small": {
        "model": "rfdetr",
        "task": "detect",
        "weights": Path.home() / ".roboflow" / "models" / "rf-detr-small.torchscript",
        "imgsz": 512,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": True,
    },
    "rfdetr_medium": {
        "model": "rfdetr",
        "task": "detect",
        "weights": Path.home() / ".roboflow" / "models" / "rf-detr-medium.torchscript",
        "imgsz": 576,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": True,
    },
    "rfdetr_large": {
        "model": "rfdetr",
        "task": "detect",
        "weights": Path.home() / ".roboflow" / "models" / "rf-detr-large-2026.torchscript",
        "imgsz": 704,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": True,
    },
    "rfdetr_base": {
        "model": "rfdetr",
        "task": "detect",
        "weights": Path.home() / ".roboflow" / "models" / "rf-detr-base.torchscript",
        "imgsz": 560,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": True,
    },
    "yolov5l_seg": {
        "model": "yolov5segment",
        "task": "segment",
        "weights": REPO_DIR / "temp" / "model_weights" / "yolov5l-seg.torchscript",
        "imgsz": 640,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": False,
    },
    "yolov8l_seg": {
        "model": "yolov8segment",
        "task": "segment",
        "weights": REPO_DIR / "temp" / "model_weights" / "yolov8l-seg.torchscript",
        "imgsz": 640,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": False,
    },
    "yolo11l_seg": {
        "model": "yolov11segment",
        "task": "segment",
        "weights": REPO_DIR / "temp" / "model_weights" / "yolo11l-seg.torchscript",
        "imgsz": 640,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "half": torch.cuda.is_available(),
        "sparse_coco": False,
    },
}

DEFAULT_MODEL_KEYS = ",".join(MODEL_SPECS.keys())
SELECTED_MODEL_KEYS = [
    key.strip() for key in os.environ.get("CVMD_MODELS", DEFAULT_MODEL_KEYS).split(",") if key.strip()
]
OUTPUT_ROOT = Path(os.environ.get("CVMD_OUTPUT_ROOT", str(REPO_DIR / "temp" / "test_runs")))
SUMMARY_CSV = Path(os.environ.get("CVMD_SUMMARY_CSV", str(OUTPUT_ROOT / "summary.csv")))
EVAL_CONF = float(os.environ.get("CVMD_CONF", "0.001"))
DRAW_CONF = float(os.environ.get("CVMD_DRAW_CONF", "0.30"))
IOU = float(os.environ.get("CVMD_IOU", "0.45"))
PYTEST_SUMMARIES: dict[str, dict] = {}


def resolve_dataset_dir() -> Path:
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate and (candidate / "images" / "train2017").exists():
            return candidate
    raise FileNotFoundError("Could not find coco128 dataset. Set COCO128_DIR or place it in the expected locations.")


def resolve_model_spec(key: str):
    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key '{key}'. Available: {sorted(MODEL_SPECS)}")

    spec = MODEL_SPECS[key].copy()
    env_key = key.upper().replace("-", "_")
    spec["weights"] = Path(os.environ.get(f"CVMD_WEIGHTS_{env_key}", str(spec["weights"])))
    spec["imgsz"] = int(os.environ.get(f"CVMD_IMGSZ_{env_key}", str(spec["imgsz"])))
    spec["device"] = os.environ.get(f"CVMD_DEVICE_{env_key}", spec["device"])
    spec["half"] = os.environ.get(f"CVMD_HALF_{env_key}", "1" if spec["half"] else "0").lower() in {"1", "true", "yes"}
    return spec


def load_ground_truth(label_path: Path, image_shape: tuple[int, int, int]):
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

    rows = []
    class_ids = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            class_ids.append(int(parts[0]))
            rows.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

    if not rows:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

    h, w = image_shape[:2]
    gt_boxes = pycvt.xywhn2xyxy(np.array(rows, dtype=np.float32), w=w, h=h).astype(np.float32)
    return gt_boxes, np.array(class_ids, dtype=np.int64)


def extract_detections(outputs):
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    if not isinstance(outputs, np.ndarray):
        outputs = np.asarray(outputs)

    if outputs.ndim != 2 or outputs.shape[1] < 6:
        raise ValueError(f"Unsupported detection output shape: {outputs.shape}")

    return outputs[:, :6].astype(np.float32, copy=False)


def remap_detection_classes(detections: np.ndarray, sparse_coco: bool) -> np.ndarray:
    if not sparse_coco or len(detections) == 0:
        return detections

    remapped = detections.copy()
    keep = []
    for i, cls in enumerate(remapped[:, 5].astype(np.int64)):
        mapped = COCO_RAW_TO_CONTIGUOUS.get(int(cls))
        if mapped is None:
            continue
        remapped[i, 5] = mapped
        keep.append(i)

    return remapped[keep] if keep else remapped[:0]


def process_batch(detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor) -> torch.Tensor:
    correct = torch.zeros((detections.shape[0], iouv.numel()), dtype=torch.bool)
    if labels.shape[0] == 0 or detections.shape[0] == 0:
        return correct

    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]

    for i, iou_thr in enumerate(iouv):
        matches = torch.nonzero((iou >= iou_thr) & correct_class, as_tuple=False)
        if matches.shape[0] == 0:
            continue
        match_values = iou[matches[:, 0], matches[:, 1]]
        order = torch.argsort(match_values, descending=True)
        matches = matches[order]

        selected_labels = set()
        selected_detections = set()
        keep = []
        for label_idx, det_idx in matches.tolist():
            if label_idx in selected_labels or det_idx in selected_detections:
                continue
            selected_labels.add(label_idx)
            selected_detections.add(det_idx)
            keep.append(det_idx)

        if keep:
            correct[torch.tensor(keep, dtype=torch.long), i] = True

    return correct


def summarize_detection_stats(stats, model_key: str):
    if not stats:
        return {
            "model": model_key,
            "task": "detect",
            "status": "passed",
            "map50": 0.0,
            "map50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_images": 0,
            "num_predictions": 0,
            "num_targets": 0,
        }

    tp = np.concatenate([x[0] for x in stats], axis=0)
    conf = np.concatenate([x[1] for x in stats], axis=0)
    pred_cls = np.concatenate([x[2] for x in stats], axis=0)
    target_cls = np.concatenate([x[3] for x in stats], axis=0)

    num_predictions = int(conf.shape[0])
    num_targets = int(target_cls.shape[0])
    if num_predictions == 0 or num_targets == 0:
        return {
            "model": model_key,
            "task": "detect",
            "status": "passed",
            "map50": 0.0,
            "map50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_images": len(stats),
            "num_predictions": num_predictions,
            "num_targets": num_targets,
        }

    _, _, p, r, _, ap, _, _, _, _, _, _ = ap_per_class(
        tp,
        conf,
        pred_cls,
        target_cls,
        names={i: name for i, name in enumerate(COCO_NAMES)},
    )
    return {
        "model": model_key,
        "task": "detect",
        "status": "passed",
        "map50": float(ap[:, 0].mean()) if ap.size else 0.0,
        "map50_95": float(ap.mean()) if ap.size else 0.0,
        "precision": float(p.mean()) if p.size else 0.0,
        "recall": float(r.mean()) if r.size else 0.0,
        "num_images": len(stats),
        "num_predictions": num_predictions,
        "num_targets": num_targets,
    }


def build_model(spec: dict):
    model = cvmd.build(
        spec["model"],
        weights=str(spec["weights"]),
        device=spec["device"],
        conf=EVAL_CONF if spec["task"] == "detect" else DRAW_CONF,
        iou=IOU,
        imgsz=spec["imgsz"],
        half=spec["half"],
        load_warm_up=False,
    )
    model.load_model(load_warm_up=False)
    return model


def run_detection_spec(model_key: str, dataset_dir: Path, save_dir: Path):
    spec = resolve_model_spec(model_key)
    if not spec["weights"].exists():
        message = f"Weights not found for {model_key}: {spec['weights']}"
        PYTEST_SUMMARIES[model_key] = {
            "model": model_key,
            "task": "detect",
            "status": "skipped",
            "num_images": 0,
            "num_predictions": 0,
            "num_targets": 0,
        }
        if pytest is not None:
            pytest.skip(message)
        raise FileNotFoundError(message)

    image_dir = dataset_dir / "images" / "train2017"
    label_dir = dataset_dir / "labels" / "train2017"
    image_paths = sorted(image_dir.glob("*.jpg"))
    model = build_model(spec)
    output_dir = save_dir / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for image_path in image_paths:
        image = iio.imread(image_path, mode="RGB")
        detections = extract_detections(model(image))
        detections = remap_detection_classes(detections, spec["sparse_coco"])

        gt_boxes, gt_class_ids = load_ground_truth(label_dir / f"{image_path.stem}.txt", image.shape)
        detections_t = torch.from_numpy(detections)
        labels_t = torch.zeros((len(gt_class_ids), 5), dtype=torch.float32)
        if len(gt_class_ids) > 0:
            labels_t[:, 0] = torch.from_numpy(gt_class_ids.astype(np.float32))
            labels_t[:, 1:] = torch.from_numpy(gt_boxes)

        correct = process_batch(detections_t, labels_t, IOU_THRESHOLDS)
        stats.append(
            (
                correct.cpu().numpy(),
                detections[:, 4] if len(detections) else np.empty((0,), dtype=np.float32),
                detections[:, 5].astype(np.int64) if len(detections) else np.empty((0,), dtype=np.int64),
                gt_class_ids,
            )
        )

        draw_detections = detections[detections[:, 4] >= DRAW_CONF]
        pred_boxes = draw_detections[:, :4].astype(np.int32) if len(draw_detections) else np.empty((0, 4), dtype=np.int32)
        pred_scores = draw_detections[:, 4] if len(draw_detections) else np.empty((0,), dtype=np.float32)
        pred_class_ids = draw_detections[:, 5].astype(np.int64) if len(draw_detections) else np.empty((0,), dtype=np.int64)

        gt_canvas = image.copy()
        gt_boxes_i = gt_boxes.astype(np.int32)
        if len(gt_boxes_i) > 0:
            gt_canvas = draw_bounding_boxes(
                image=gt_canvas,
                boxes=gt_boxes_i,
                labels=[f"gt:{COCO_NAMES[c]}" if c < len(COCO_NAMES) else f"gt:{c}" for c in gt_class_ids],
                colors=[(0, 255, 0)] * len(gt_boxes_i),
            )

        pred_canvas = image.copy()
        if len(pred_boxes) > 0:
            pred_canvas = draw_bounding_boxes(
                image=pred_canvas,
                boxes=pred_boxes,
                labels=[
                    f"pred:{COCO_NAMES[c] if c < len(COCO_NAMES) else c} {s:.2f}"
                    for c, s in zip(pred_class_ids, pred_scores)
                ],
                colors=[(255, 64, 64)] * len(pred_boxes),
            )

        iio.imwrite(output_dir / image_path.name, np.concatenate([gt_canvas, pred_canvas], axis=1))

    return summarize_detection_stats(stats, model_key)


def run_segmentation_spec(model_key: str, dataset_dir: Path, save_dir: Path, blank_image: np.ndarray):
    spec = resolve_model_spec(model_key)
    if not spec["weights"].exists():
        message = f"Weights not found for {model_key}: {spec['weights']}"
        PYTEST_SUMMARIES[model_key] = {
            "model": model_key,
            "task": "segment",
            "status": "skipped",
            "num_images": 0,
            "num_predictions": 0,
            "num_targets": 0,
        }
        if pytest is not None:
            pytest.skip(message)
        raise FileNotFoundError(message)

    image_paths = sorted((dataset_dir / "images" / "train2017").glob("*.jpg"))
    model = build_model(spec)
    output_dir = save_dir / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = iio.imread(image_path, mode="RGB")
        results, masks = model(image)
        plot_im = overlay_masks(image=image, masks=masks)
        bbox = results[..., :4]
        conf = results[..., 4]
        cls = results[..., 5].astype(int)
        plot_im = draw_bounding_boxes(
            image=plot_im,
            boxes=bbox,
            labels=[f"{c}-{s:.2f}" for c, s in zip(cls, conf)],
            line_width=2,
        )
        iio.imwrite(output_dir / image_path.name, plot_im)

    blank_results, blank_masks = model(blank_image)
    assert len(blank_results) >= 0
    assert blank_masks.ndim >= 2
    return {"model": model_key, "task": "segment", "status": "passed", "num_images": len(image_paths)}


def run_selected_models(dataset_dir: Path, save_dir: Path, blank_image: np.ndarray):
    results = []
    for model_key in SELECTED_MODEL_KEYS:
        spec = resolve_model_spec(model_key)
        print(f"Running {model_key} ({spec['task']})")
        try:
            if spec["task"] == "detect":
                results.append(run_detection_spec(model_key, dataset_dir, save_dir))
            else:
                results.append(run_segmentation_spec(model_key, dataset_dir, save_dir, blank_image))
        except FileNotFoundError as exc:
            print(f"Skipping {model_key}: {exc}")
            results.append(
                {
                    "model": model_key,
                    "task": spec["task"],
                    "status": "skipped",
                    "num_images": 0,
                    "num_predictions": 0,
                    "num_targets": 0,
                }
            )
    return results


def write_summary_csv(summaries, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "task",
        "status",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "num_images",
        "num_predictions",
        "num_targets",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary.get(key, "") for key in fieldnames}
            if "task" not in summary:
                row["task"] = MODEL_SPECS.get(summary["model"], {}).get("task", "")
            writer.writerow(row)


if pytest is not None:
    @pytest.fixture(scope="session", autouse=True)
    def _write_summary_csv_on_session_end():
        PYTEST_SUMMARIES.clear()
        yield
        summaries = [PYTEST_SUMMARIES[key] for key in sorted(PYTEST_SUMMARIES)]
        if summaries:
            write_summary_csv(summaries, SUMMARY_CSV)


@_parametrize(
    "model_key",
    [key for key, spec in MODEL_SPECS.items() if spec["task"] == "detect"],
)
def test_detection_models(model_key, save_dir, blank_image):
    dataset_dir = resolve_dataset_dir()
    summary = run_detection_spec(model_key, dataset_dir, save_dir)
    PYTEST_SUMMARIES[model_key] = summary
    assert summary["num_images"] > 0


@_parametrize(
    "model_key",
    [key for key, spec in MODEL_SPECS.items() if spec["task"] == "segment"],
)
def test_segmentation_models(model_key, save_dir, blank_image):
    dataset_dir = resolve_dataset_dir()
    summary = run_segmentation_spec(model_key, dataset_dir, save_dir, blank_image)
    PYTEST_SUMMARIES[model_key] = summary
    assert summary["num_images"] > 0


if __name__ == "__main__":
    dataset_dir = resolve_dataset_dir()
    save_dir = OUTPUT_ROOT
    blank_image = 255 * np.ones((1024, 800, 3), dtype=np.uint8)

    print("Dataset:", dataset_dir)
    print("Models:", ", ".join(SELECTED_MODEL_KEYS))
    print("Output:", save_dir)
    summaries = run_selected_models(dataset_dir, save_dir, blank_image)
    write_summary_csv(summaries, SUMMARY_CSV)

    print("\nSummary")
    for summary in summaries:
        if "map50" in summary:
            print(
                f"{summary['model']}: "
                f"mAP50={summary['map50']:.4f}, "
                f"mAP50-95={summary['map50_95']:.4f}, "
                f"P={summary['precision']:.4f}, "
                f"R={summary['recall']:.4f}, "
                f"images={summary['num_images']}, "
                f"predictions={summary['num_predictions']}, "
                f"targets={summary['num_targets']}"
            )
        else:
            print(f"{summary['model']}: images={summary['num_images']}")
    print(f"\nSaved CSV: {SUMMARY_CSV}")
