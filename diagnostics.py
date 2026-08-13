"""Fast, non-inference preflight checks for the inspection project."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CORE_IMPORTS = (
    "torch",
    "torchvision",
    "ultralytics",
    "transformers",
    "accelerate",
    "PIL",
    "gradio",
    "einops",
    "huggingface_hub",
)
REQUIRED_DIRECTORIES = ("models", "test_images", "evidence", "tests")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_preflight(root: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    errors: list[str] = []

    package_versions: dict[str, str] = {}
    for module_name in CORE_IMPORTS:
        try:
            importlib.import_module(module_name)
            distribution = "pillow" if module_name == "PIL" else module_name
            package_versions[module_name] = importlib.metadata.version(distribution)
        except Exception as exc:  # noqa: BLE001 - diagnostics must report every import failure
            errors.append(f"Import {module_name}: {type(exc).__name__}: {exc}")
    checks["packages"] = package_versions

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        checks["runtime"] = {
            "python": sys.version.split()[0],
            "device": "cuda" if cuda_available else "cpu",
            "cuda_available": cuda_available,
            "dtype": "float16" if cuda_available else "float32",
        }
    except Exception as exc:  # noqa: BLE001 - preflight reports an unusable runtime
        checks["runtime"] = {
            "python": sys.version.split()[0],
            "error": f"{type(exc).__name__}: {exc}",
        }
        errors.append("Torch runtime inspection failed.")

    missing_directories = [
        name for name in REQUIRED_DIRECTORIES if not (root / name).is_dir()
    ]
    checks["project_directories"] = {
        "required": list(REQUIRED_DIRECTORIES),
        "missing": missing_directories,
    }
    if missing_directories:
        errors.append(f"Missing directories: {', '.join(missing_directories)}")

    metadata_path = root / "models" / "MODEL_SOURCE.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            checkpoint = root / metadata["local_filename"]
            checkpoint_check = {
                "path": str(checkpoint),
                "exists": checkpoint.is_file(),
                "expected_size": metadata["size_bytes"],
                "expected_sha256": metadata["sha256"],
            }
            if checkpoint.is_file():
                checkpoint_check["actual_size"] = checkpoint.stat().st_size
                checkpoint_check["actual_sha256"] = _sha256(checkpoint)
                checkpoint_check["verified"] = (
                    checkpoint_check["actual_size"] == checkpoint_check["expected_size"]
                    and checkpoint_check["actual_sha256"]
                    == checkpoint_check["expected_sha256"]
                )
            else:
                checkpoint_check["verified"] = False
            checks["detector_checkpoint"] = checkpoint_check
            if not checkpoint_check["verified"]:
                errors.append(
                    "Detector checkpoint is missing or does not match MODEL_SOURCE.json."
                )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            checks["detector_checkpoint"] = {
                "verified": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append("Detector metadata is unreadable or incomplete.")
    else:
        errors.append("models/MODEL_SOURCE.json is missing.")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    checks["pip_check"] = {
        "passed": pip_check.returncode == 0,
        "output": (pip_check.stdout or pip_check.stderr).strip(),
    }
    if pip_check.returncode != 0:
        errors.append("pip check reported dependency conflicts.")

    return {
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root.resolve()),
        "checks": checks,
        "errors": errors,
        "note": "Preflight does not load detector or VLM weights and is not an inference test.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fast project preflight diagnostics."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--save-json", type=Path)
    args = parser.parse_args()
    report = run_preflight(args.project_root.resolve())
    rendered = json.dumps(report, indent=2)
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
