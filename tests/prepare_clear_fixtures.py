"""Create documented crack-focused crops from licensed source photographs."""

from pathlib import Path

from PIL import Image

CROPS = {
    "clear_01_original.jpg": (
        (0, 280, 480, 640),
        "clear_01_commons_difficult.jpg",
    ),
    "clear_03_original.jpg": (
        (300, 220, 1050, 1900),
        "clear_03_commons_difficult.jpg",
    ),
}


def main() -> None:
    source_dir = Path("test_images/source_images")
    destination_dir = Path("test_images/difficult_cases")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_name, (crop_box, destination_name) in CROPS.items():
        destination = destination_dir / destination_name
        with Image.open(source_dir / source_name) as image:
            image = image.convert("RGB").crop(crop_box)
            image.save(destination, "JPEG", quality=95, optimize=True)
        print(f"{source_name} {crop_box} -> {destination}")


if __name__ == "__main__":
    main()
