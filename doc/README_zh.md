# CVMD

[English](../README.md)

> 一个以 TorchScript 为优先、采用统一 NumPy API 的计算机视觉推理工具库。

## 为什么使用 CVMD

- **统一推理接口**：YOLO 和 DETR 风格模型都使用同一个 `model(image)` 调用方式。
- **面向部署**：直接加载 TorchScript 权重，减少对训练代码仓库的依赖。
- **便于切换模型**：通过 `build(...)` 切换不同架构，同时保持一致的输入输出约定。
- **易于扩展**：可以先做简单单图推理，后续再扩展到滑窗推理或基于 Ray 的分布式推理。

## 安装

```bash
pip install cvmd
```

## 快速开始

```python
import imageio.v3 as iio
from cvmd import build

model = build("yolov11det", weights="yolo11l.torchscript", device="cuda")
model.load_model()

image = iio.imread("image.jpg")
results = model(image)
# results: [x1, y1, x2, y2, confidence, class]
```

## 当前支持的模型

| 模型系列 | 任务 | 注册名称 |
| :--- | :--- | :--- |
| **YOLOv12** | 检测 / 分割 | `yolov12det`, `yolov12seg` |
| **YOLOv11** | 检测 / 分割 | `yolov11det`, `yolov11seg` |
| **YOLOv8** | 检测 / 分割 | `yolov8det`, `yolov8seg` |
| **YOLOv5** | 检测 / 分割 | `yolov5det`, `yolov5seg` |
| **DETR** | 检测 | `detr` |
| **RF-DETR** | 检测 | `rfdetr`, `rfdetrdetect`, `rf-detr` |
| **Deformable DETR** | 检测 | `deformabledetr`, `deformable_detr`, `deformable-detr` |

## 核心 API

- `build(model_name_or_cls, **kwargs)`：按名称或类创建模型实例。
- `list_models()`：列出当前已注册模型。
- `register_model(*names)`：注册自定义模型类。

检测模型返回：

```python
# np.ndarray, shape=(N, 6)
# [x1, y1, x2, y2, confidence, class]
```

分割模型返回：

```python
# (detections, masks)
# detections: np.ndarray, shape=(N, 6)
# masks: np.ndarray, shape=(N, H, W)
```

## 更多文档

- [English README](../README.md)
- [使用指南](guide_zh.md)
- [示例与测试](../test/)

## 开发

```bash
git clone <this repository>
cd cvmd
uv sync --dev
```
