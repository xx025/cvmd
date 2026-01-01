# CVMD

> A Computer Vision Model Development toolkit.
> cvmd uses NumPy arrays as both input and output, aiming to provide a unified and concise model inference interface.

## Key Features

- **Unified API**: "NumPy in, NumPy out" design. All models share a consistent interface, making it easy to switch between different YOLO versions.
- **Flexible Registry**: Easily extend the library with custom models using the `@register_model` decorator.
- **Production Ready**: Optimized for inference using TorchScript, removing dependencies on training codebases.
- **Scalable Inference**: Built-in support for [Ray](https://www.ray.io/) to enable multi-GPU distributed inference for large datasets.
- **Advanced Utilities**: Includes sliding window inference for high-resolution images and Weighted Boxes Fusion (WBF) for result merging.
- **Clean Architecture**: Modular design with minimal redundancy, making it lightweight and easy to maintain.

## Design Philosophy: Why Batch=1?

`cvmd` is intentionally designed to process one image at a time (`batch=1`). This choice prioritizes:

- **API Simplicity**: A direct `model(image)` call is intuitive and returns a clean NumPy array, avoiding the complexity of list-of-tensors or padded batch management.
- **Input Flexibility**: It handles images of any resolution automatically without requiring manual padding or alignment for batching.
- **Horizontal Scaling**: Instead of "Vertical Scaling" (increasing batch size), `cvmd` promotes "Horizontal Scaling" via **Ray**. By running multiple model instances in parallel, you can achieve high throughput while keeping the inference logic simple and robust.

## Installation

```bash
pip install cvmd
```

## Quick Start

You can build a model using the `build` function (convenient for dynamic names) or by importing the model class directly (better for IDE support).

```python
import imageio.v3 as iio
from cvmd import build, Yolov11Detect

# Option 1: Build by name
model = build("yolov11det", weights="yolo11l.torchscript", device="cuda")

# Option 2: Direct import
# model = Yolov11Detect(weights="yolo11l.torchscript", device="cuda")

model.load_model()

# Read image (HWC, RGB)
image = iio.imread("image.jpg")

# Perform inference
results = model(image)
# results: [x1, y1, x2, y2, confidence, class]
```

## Core API

### Model Building and Management

`cvmd` provides a registration mechanism to manage different models. While the `build` pattern is convenient for dynamic model creation, you can also import model classes directly for better IDE support and type checking.

- `list_models()`: List all registered model names.
- `build(model_name_or_cls, **kwargs)`: Build a model instance by name or class.
- `register_model(*names)`: Decorator to register custom model classes into `cvmd`.

### Supported Models

Currently supported model series (primarily loaded via TorchScript):

| Model Series | Task | Registered Names |
| :--- | :--- | :--- |
| **YOLOv12** | Detection / Segmentation | `yolov12det`, `yolov12seg` |
| **YOLOv11** | Detection / Segmentation | `yolov11det`, `yolov11seg` |
| **YOLOv8** | Detection / Segmentation | `yolov8det`, `yolov8seg` |
| **YOLOv5** | Detection / Segmentation | `yolov5det`, `yolov5seg` |
| **DETR** | Detection | `detr` |
| **Deformable DETR** | Detection | `deformabledetr` (To be implemented) |

### Inference Interface

All model classes follow a unified calling convention:

#### Detection Models (`*Detect`)
- **Input**: `image` (np.ndarray, HWC, RGB)
- **Output**: `results` (np.ndarray, shape=(N, 6))
    - Format per row: `[x1, y1, x2, y2, confidence, class]`

#### Segmentation Models (`*Segment`)
- **Input**: `image` (np.ndarray, HWC, RGB)
- **Output**: `(detections, masks)`
    - `detections`: (np.ndarray, shape=(N, 6)), same format as above.
    - `masks`: (np.ndarray, shape=(N, H, W)), boolean masks.

## Utility Functions

### Sliding Window Inference

For large image inference, you can use `detect_with_windows`:

```python
from cvmd.utils.windows import detect_with_windows

# Define windows [x1, y1, x2, y2]
windows = [[0, 0, 640, 640], [320, 320, 960, 960]]

results = detect_with_windows(
    image, 
    windows, 
    model, 
    merge=True, 
    merge_iou=0.2
)
```

### Distributed Inference with Ray

`cvmd` includes a utility for distributed inference using [Ray](https://www.ray.io/). This is useful for processing large batches of images across multiple GPUs.

```python
from cvmd.utils.ray_infer import ray_infer_iter, InferActor

# Define your custom handler
def my_handler(task, model_config, runs_config):
    model = model_config["model"]
    image = task["image"]
    return model(image)

# Run distributed inference
tasks = [{"image": img} for img in my_images]
results = ray_infer_iter(
    InferActor,
    tasks,
    num_actors=4,
    actor_kwargs={
        "model_config": {"model_name": "yolov11det", "weights": "yolo11l.torchscript"},
        "handler": my_handler
    }
)

for r in results:
    print(r)
```

## Examples & Tests

You can find more usage examples in the [test/](test/) directory:

- [test_detect_with_windows.py](test/test_detect_with_windows.py): Sliding window inference example.
- [test_ray.py](test/test_ray.py): Distributed inference with Ray.
- [test_yolov11_detect.py](test/test_yolov11_detect.py): YOLOv11 detection example.
- [test_yolov11_segment.py](test/test_yolov11_segment.py): YOLOv11 segmentation example.

## Development

```bash
git clone <this repository>
cd cvmd
uv sync --dev
```

