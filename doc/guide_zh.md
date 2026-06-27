# CVMD 使用指南

## 设计理念

`cvmd` 有意围绕 `batch=1` 推理来设计。

- 直接调用 `model(image)`，接口更小、更稳定。
- 可以自然处理不同分辨率图像，不需要手动做 batch padding。
- 当需要更高吞吐时，更推荐通过 Ray 做横向扩展，而不是在模型封装层里增加 batch 复杂度。

## 滑窗推理

对于大图，可以使用 `detect_with_windows`：

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

## 基于 Ray 的分布式推理

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

## 测试

主测试入口：

- [test_inference.py](../test/test_inference.py)

其他示例：

- [test_detect_with_windows.py](../test/test_detect_with_windows.py)
- [test_ray.py](../test/test_ray.py)
