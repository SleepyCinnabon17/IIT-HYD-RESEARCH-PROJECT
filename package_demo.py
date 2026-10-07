"""Build a source distribution from an explicit list of public app files."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
FILES = (
    "app.py", "pipeline.py", "inspection_view.py", "deployment.py", "run_demo.py",
    "ui.css", "requirements.txt", "START_WINDOWS.bat", "TEST_AND_SHARE.md", "package_demo.py",
    "models/MODEL_SOURCE.json", "models/VLM_SOURCE.json",
    "models/gate_calibration.json", "models/detection_operating_point.json",
)


def main():
    for name in FILES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    destination = ROOT / "dist" / "Crack-Inspection-Demo.zip"
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, f"Crack-Inspection-Demo/{name}")
    print(f"Created {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
