from __future__ import annotations
from typing import Dict, Type, Callable, Any

_MODEL_REGISTRY: Dict[str, Type] = {}   # lower(name) -> class


def register_model(*names: str) -> Callable[[Type], Type]:
    if not names:
        raise ValueError("register_model() requires at least one name")

    def _wrap(cls: Type) -> Type:
        for n in names:
            key = n.lower()
            if key in _MODEL_REGISTRY and _MODEL_REGISTRY[key] is not cls:
                raise KeyError(
                    f"Model name '{n}' already registered by {_MODEL_REGISTRY[key]}"
                )
            _MODEL_REGISTRY[key] = cls

        cls.MODEL_NAMES = tuple(sorted({n.lower() for n in names}))

        # ✅ 正确的打印位置：这里才能看到 cls
        # print(f"Registering model: {names} -> {cls}")

        return cls

    return _wrap


def build(model: str | Type, *args: Any, **kwargs: Any):
    """
    build("yolov8det", weights=..., device=...)
    build(Yolov8Detect, weights=..., device=...)
    """
    if isinstance(model, str):
        key = model.lower()
        if key not in _MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{model}'. Available: {list_models()}")
        cls = _MODEL_REGISTRY[key]
    else:
        cls = model
    # print(f"Building model: {cls}")
    return cls(*args, **kwargs)


def list_models():
    """Return all registered names (lowercased)."""
    return sorted(_MODEL_REGISTRY.keys())
