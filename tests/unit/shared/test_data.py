from __future__ import annotations

from pathlib import Path

import pytest

from intensity.config import REPO_ROOT, resolve_path
from intensity.data import bucket, bucket_uri, data_dir, data_path, resolve_data


def test_data_dir_defaults_to_repo_data(monkeypatch):
    monkeypatch.delenv("INTENSITY_DATA_DIR", raising=False)
    assert data_dir() == REPO_ROOT / "data"
    assert resolve_path("data/events/football") == REPO_ROOT / "data" / "events" / "football"


def test_data_paths_follow_intensity_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("INTENSITY_DATA_DIR", str(tmp_path))
    assert resolve_path("data/samples/x.mp4") == tmp_path / "samples" / "x.mp4"
    assert resolve_path("data") == tmp_path
    assert data_path("samples/x.mp4") == resolve_data("data/samples/x.mp4")
    # Non-data config paths stay repo-relative, absolute paths are untouched.
    assert resolve_path("config/feeds.yaml") == REPO_ROOT / "config" / "feeds.yaml"
    assert resolve_path("/abs/data/x") == Path("/abs/data/x")


def test_bucket_layout_mirrors_data_without_the_prefix(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "gs://example-bucket/")
    assert bucket() == "example-bucket"
    assert bucket_uri("samples/x.mp4") == "gs://example-bucket/samples/x.mp4"
    assert bucket_uri() == "gs://example-bucket/"


def test_bucket_uri_needs_gcs_bucket(monkeypatch, tmp_path):
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.setattr("intensity.data.ENV_FILE", tmp_path / "missing.env")
    assert bucket() is None
    with pytest.raises(RuntimeError, match="GCS_BUCKET"):
        bucket_uri("samples")


def test_bucket_falls_back_to_the_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=secret\nGCS_BUCKET=gs://from-env-file\n")
    monkeypatch.setattr("intensity.data.ENV_FILE", env)
    assert bucket() == "from-env-file"
