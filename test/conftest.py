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
def sample_image():
    # Try to find a real image in coco128, otherwise create a dummy one
    coco_path = Path("temp/datasets/coco128/images/train2017/000000000074.jpg")
    if coco_path.exists():
        return iio.imread(coco_path, mode="RGB")
    return np.zeros((640, 640, 3), dtype=np.uint8)

@pytest.fixture
def blank_image():
    return 255 * np.ones((1024, 800, 3), dtype=np.uint8)
