import math
import random
from pathlib import Path
from PIL import Image
import imageio.v3 as iio

def make_mosaic(
    img_dir: str,
    num_images: int = 64,
    thumb_size: tuple = (480, 480),
    seed: int = 0,
    out_path: str = "./mosaic.jpg"
):
    img_dir = Path(img_dir)
    all_images = sorted(img_dir.rglob("*.jpg"))
    total = len(all_images)
    if total == 0:
        raise RuntimeError(f"No jpg found in {img_dir}")

    n = min(num_images, total)
    random.seed(seed)
    selected = random.sample(all_images, n)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    W, H = thumb_size
    mosaic_w = cols * W
    mosaic_h = rows * H

    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), (0, 0, 0))

    for idx, img_path in enumerate(selected):
        arr = iio.imread(img_path, mode="RGB")
        im = Image.fromarray(arr)
        im = im.resize((W, H), Image.BILINEAR)

        r = idx // cols
        c = idx % cols
        x = c * W
        y = r * H
        mosaic.paste(im, (x, y))

    mosaic.save(out_path, quality=95)
    return Path(out_path).resolve()
