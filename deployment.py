"""Cloud entrypoint: acquire the pinned detector, then serve the inspection app."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def prepare_detector() -> Path:
    from huggingface_hub import hf_hub_download

    root = Path(__file__).resolve().parent
    source = json.loads((root / "models/MODEL_SOURCE.json").read_text(encoding="utf-8"))
    destination = root / source["local_filename"]
    if not destination.exists():
        downloaded = hf_hub_download(
            repo_id=source["repository"],
            filename=source["source_filename"],
            revision=source["revision"],
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".download")
        shutil.copyfile(downloaded, temporary)
        with temporary.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != source["sha256"]:
            temporary.unlink()
            raise RuntimeError("Downloaded detector checksum does not match the pinned model.")
        temporary.replace(destination)
    with destination.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != source["sha256"]:
        raise RuntimeError("Detector checksum mismatch; refusing to serve a different model.")
    return destination


def main() -> None:
    import torch

    os.chdir(Path(__file__).resolve().parent)
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
    prepare_detector()
    from app import build_demo

    build_demo().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_api=False,
        max_file_size="20mb",
    )


if __name__ == "__main__":
    main()
