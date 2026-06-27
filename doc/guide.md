# CVMD Guide

## Design Philosophy

`cvmd` is intentionally designed around `batch=1` inference.

- A direct `model(image)` call keeps the API small and predictable.
- Variable-resolution images can be processed without manual batch padding.
- When more throughput is needed, the preferred path is horizontal scaling with Ray rather than increasing batch complexity inside the model wrapper.

## Sliding-Window Inference

For large images, use `detect_with_windows`:

```python
from cvmd.utils.windows import detect_with_windows

windows = [[0, 0, 640, 640], [320, 320, 960, 960]]

results = detect_with_windows(
    image,
    windows,
    model,
    merge=True,
    merge_iou=0.2,
)
```

## Distributed Inference With Ray

```python
from cvmd.utils.ray_infer import ray_infer_iter, InferActor

def my_handler(task, model_config, runs_config):
    model = model_config["model"]
    image = task["image"]
    return model(image)

tasks = [{"image": img} for img in my_images]
results = ray_infer_iter(
    InferActor,
    tasks,
    num_actors=4,
    actor_kwargs={
        "model_config": {"model_name": "yolov11det", "weights": "yolo11l.torchscript"},
        "handler": my_handler,
    },
)

for r in results:
    print(r)
```

## Testing

Main test entry:

- [test_inference.py](../test/test_inference.py)

Other examples:

- [test_detect_with_windows.py](../test/test_detect_with_windows.py)
- [test_ray.py](../test/test_ray.py)
