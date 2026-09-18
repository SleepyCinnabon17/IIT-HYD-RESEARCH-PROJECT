import hashlib
import json
from unittest.mock import Mock

import huggingface_hub
import pytest

import deployment


@pytest.fixture
def model_source(tmp_path, monkeypatch):
    monkeypatch.setattr(deployment, "__file__", str(tmp_path / "deployment.py"))
    models = tmp_path / "models"
    models.mkdir()
    source = {
        "repository": "example/detector",
        "revision": "pinned-commit",
        "source_filename": "weights/best.pt",
        "local_filename": "models/detector.pt",
        "sha256": hashlib.sha256(b"expected weights").hexdigest(),
    }
    (models / "MODEL_SOURCE.json").write_text(json.dumps(source))
    return tmp_path, source


def test_existing_model_is_verified_without_network(model_source, monkeypatch):
    root, _ = model_source
    path = root / "models/detector.pt"
    path.write_bytes(b"expected weights")
    download = Mock(side_effect=AssertionError("Should reuse verified weights"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert deployment.prepare_detector() == path
    download.assert_not_called()


def test_new_model_uses_pinned_revision(model_source, monkeypatch):
    root, source = model_source
    cached = root / "cached.pt"
    cached.write_bytes(b"expected weights")
    download = Mock(return_value=str(cached))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert deployment.prepare_detector().read_bytes() == b"expected weights"
    download.assert_called_once_with(
        repo_id=source["repository"], filename=source["source_filename"],
        revision=source["revision"],
    )


def test_bad_download_never_becomes_active_model(model_source, monkeypatch):
    root, _ = model_source
    cached = root / "cached.pt"
    cached.write_bytes(b"incorrect weights")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", Mock(return_value=str(cached)))
    with pytest.raises(RuntimeError, match="checksum"):
        deployment.prepare_detector()
    assert not (root / "models/detector.pt").exists()
    assert not (root / "models/detector.download").exists()
