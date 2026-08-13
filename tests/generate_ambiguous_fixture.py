"""Generate a deterministic, non-crack surface fixture for gate testing."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def main() -> None:
    rng = np.random.default_rng(20260813)
    height, width = 640, 640
    base = rng.normal(132, 13, size=(height, width, 1))
    grain = rng.normal(0, 5, size=(height, width, 3))
    rgb = np.clip(base + grain + np.array([4, 2, 0]), 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(1.2))

    stain = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(stain)
    draw.ellipse((70, 130, 410, 410), fill=(53, 45, 38, 55))
    draw.ellipse((290, 250, 610, 590), fill=(62, 55, 48, 42))
    draw.line(
        [(0, 520), (190, 470), (410, 500), (640, 430)], fill=(35, 32, 30, 70), width=34
    )
    stain = stain.filter(ImageFilter.GaussianBlur(28))
    image = Image.alpha_composite(image.convert("RGBA"), stain).convert("RGB")

    output = Path("test_images/ambiguous_02.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "JPEG", quality=94, optimize=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
