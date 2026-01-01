import pytest
from pathlib import Path
import imageio.v3 as iio
import numpy as np

@pytest.fixture
def save_dir():
    path = Path("temp/test_runs")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture
def coco_images():
    coco_dir = Path("temp/datasets/coco128/images/train2017")
    if not coco_dir.exists():
        return []
    return list(coco_dir.rglob("*.jpg"))

@pytest.fixture
def sample_image(coco_images):
    if coco_images:
        return iio.imread(coco_images[0], mode="RGB")
    return np.zeros((640, 640, 3), dtype=np.uint8)

@pytest.fixture
def blank_image():
    return 255 * np.ones((1024, 800, 3), dtype=np.uint8)
