"""Unit tests for ``agile_assistant.rag.reranker.ensure_reranker_model_downloaded``.

The reranker snapshot is mirrored from Yandex Cloud Object Storage at
``s3://{s3_models_bucket}/{s3_models_path}/{reranker_model}/`` to
``embedding_model_cache_dir`` on first use — same idiom as
``ensure_embedding_model_downloaded``. These tests pin the contract so
that the celery-worker no longer hits HuggingFace Hub at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agile_assistant.rag import reranker as rr


@pytest.fixture
def configured_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Wire settings so the cache lands in tmp_path and S3 is on."""
    monkeypatch.setattr(rr.settings, "s3_models_bucket", "quant-models-agile")
    monkeypatch.setattr(rr.settings, "s3_models_path", "models")
    monkeypatch.setattr(rr.settings, "reranker_model", "bge-reranker-v2-m3")
    monkeypatch.setattr(rr.settings, "embedding_model_cache_dir", str(tmp_path))
    monkeypatch.setattr(rr.settings, "s3_endpoint", "https://storage.yandexcloud.net")
    return tmp_path


def _fake_paginator(pages: list[list[dict[str, Any]]]) -> MagicMock:
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": page} for page in pages]
    return paginator


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``boto3.session.Session`` so no AWS calls happen."""
    s3 = MagicMock(name="s3_client")
    s3.get_paginator.return_value = _fake_paginator([[]])
    s3.download_file = MagicMock(return_value=None)
    session = MagicMock(client=MagicMock(return_value=s3))

    import boto3

    monkeypatch.setattr(boto3.session, "Session", MagicMock(return_value=session))
    return s3


@pytest.mark.unit
class TestEnsureRerankerModelDownloaded:
    def test_returns_local_path_after_download(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        fake_s3.get_paginator.return_value = _fake_paginator(
            [
                [
                    {"Key": "models/bge-reranker-v2-m3/config.json"},
                    {"Key": "models/bge-reranker-v2-m3/tokenizer.json"},
                    {"Key": "models/bge-reranker-v2-m3/pytorch_model.bin"},
                ]
            ]
        )
        target_dir = configured_settings / "bge-reranker-v2-m3"

        def _materialise(_bucket: str, _key: str, dest: str) -> None:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"x")

        fake_s3.download_file.side_effect = _materialise

        result = rr.ensure_reranker_model_downloaded()
        assert result == str(target_dir)
        assert fake_s3.download_file.call_count == 3

    def test_skips_download_when_marker_present(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # Pin: cached snapshot must short-circuit S3 — celery-worker
        # restart cost stays flat.
        target_dir = configured_settings / "bge-reranker-v2-m3"
        target_dir.mkdir(parents=True)
        (target_dir / "config.json").write_text("{}", encoding="utf-8")

        result = rr.ensure_reranker_model_downloaded()
        assert result == str(target_dir)
        fake_s3.download_file.assert_not_called()
        fake_s3.get_paginator.assert_not_called()

    def test_no_bucket_falls_back_to_hf_hub_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_s3: MagicMock,
    ) -> None:
        # Pin: when S3 is intentionally disabled, the value is forwarded
        # to HuggingFace as-is (back-compat for ad-hoc local runs).
        monkeypatch.setattr(rr.settings, "s3_models_bucket", None)
        monkeypatch.setattr(rr.settings, "reranker_model", "BAAI/bge-reranker-v2-m3")
        result = rr.ensure_reranker_model_downloaded()
        assert result == "BAAI/bge-reranker-v2-m3"
        fake_s3.download_file.assert_not_called()

    def test_missing_marker_after_sync_raises(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # Pin: protects against half-uploaded bucket where config.json is
        # absent — without this guard sentence-transformers would silently
        # try the Hub.
        fake_s3.get_paginator.return_value = _fake_paginator(
            [[{"Key": "models/bge-reranker-v2-m3/tokenizer.json"}]]
        )

        def _materialise(_bucket: str, _key: str, dest: str) -> None:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"x")

        fake_s3.download_file.side_effect = _materialise

        with pytest.raises(RuntimeError, match=r"config\.json is missing"):
            rr.ensure_reranker_model_downloaded()
