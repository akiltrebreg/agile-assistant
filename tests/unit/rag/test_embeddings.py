"""Unit tests for ``hse_prom_prog.rag.embeddings``.

The module's S3 downloader replaces the previous HuggingFace-Hub flow:
the embedding model snapshot lives in Yandex Cloud Object Storage at
``s3://{s3_models_bucket}/{s3_models_path}/{embedding_model}/`` and is
mirrored to ``embedding_model_cache_dir`` on first use.

Tests cover:

  * ``ensure_embedding_model_downloaded`` is idempotent — the second call
    skips the download when ``config.json`` is present (the canonical HF
    snapshot marker, same convention as ``download-model`` in compose).
  * ``s3_models_bucket=None`` short-circuits to the HuggingFace ID as a
    back-compat fallback for ad-hoc local runs.
  * ``_download_model_from_s3`` walks every object under the prefix,
    preserves nested paths, and skips zero-byte "directory" keys.
  * Empty bucket + missing ``config.json`` after sync both raise with
    a clear message — silent fallthrough would let HF Hub be queried
    without anyone noticing.
  * Truncation/normalization helpers behave as documented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from hse_prom_prog.rag import embeddings as emb

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def configured_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Wire settings so the cache lands in tmp_path and S3 is on."""
    monkeypatch.setattr(emb.settings, "s3_models_bucket", "quant-models-agile")
    monkeypatch.setattr(emb.settings, "s3_models_path", "models")
    monkeypatch.setattr(emb.settings, "embedding_model", "multilingual-e5-base")
    monkeypatch.setattr(emb.settings, "embedding_model_cache_dir", str(tmp_path))
    monkeypatch.setattr(emb.settings, "s3_endpoint", "https://storage.yandexcloud.net")
    return tmp_path


def _fake_paginator(pages: list[list[dict[str, Any]]]) -> MagicMock:
    """Build a paginator that yields the given list-of-pages."""
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": page} for page in pages]
    return paginator


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``boto3.session.Session`` so no AWS calls happen.

    Returns the s3 client mock so tests can inspect download_file calls
    and configure the paginator return value.
    """
    s3 = MagicMock(name="s3_client")
    # Default empty paginator — tests override.
    s3.get_paginator.return_value = _fake_paginator([[]])
    s3.download_file = MagicMock(return_value=None)
    session = MagicMock(client=MagicMock(return_value=s3))

    import boto3

    monkeypatch.setattr(boto3.session, "Session", MagicMock(return_value=session))
    return s3


# ===================================================================== #
# ensure_embedding_model_downloaded
# ===================================================================== #


@pytest.mark.unit
class TestEnsureEmbeddingModelDownloaded:
    def test_returns_local_path_after_download(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # Configure paginator with a tiny snapshot.
        fake_s3.get_paginator.return_value = _fake_paginator(
            [
                [
                    {"Key": "models/multilingual-e5-base/config.json"},
                    {"Key": "models/multilingual-e5-base/tokenizer.json"},
                    {"Key": "models/multilingual-e5-base/pytorch_model.bin"},
                ]
            ]
        )
        # Materialise marker via download_file side-effect.
        target_dir = configured_settings / "multilingual-e5-base"

        def _materialise(_bucket: str, _key: str, dest: str) -> None:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"x")

        fake_s3.download_file.side_effect = _materialise

        result = emb.ensure_embedding_model_downloaded()
        assert result == str(target_dir)
        # Pin: exactly 3 download_file calls (one per object in the page).
        assert fake_s3.download_file.call_count == 3

    def test_skips_download_when_marker_present(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # Pre-create the marker — second run must skip S3 entirely.
        # Pin: this is the hot-path optimisation that keeps process restart
        # cost flat (no re-download on every container boot).
        target_dir = configured_settings / "multilingual-e5-base"
        target_dir.mkdir(parents=True)
        (target_dir / "config.json").write_text("{}", encoding="utf-8")

        result = emb.ensure_embedding_model_downloaded()
        assert result == str(target_dir)
        fake_s3.download_file.assert_not_called()
        fake_s3.get_paginator.assert_not_called()

    def test_no_bucket_falls_back_to_hf_hub_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_s3: MagicMock,
    ) -> None:
        # Pin: when S3 is intentionally disabled, the value is forwarded
        # to HuggingFace as-is. This preserves the ad-hoc local dev flow
        # for contributors without Yandex creds.
        monkeypatch.setattr(emb.settings, "s3_models_bucket", None)
        monkeypatch.setattr(emb.settings, "embedding_model", "intfloat/multilingual-e5-base")
        result = emb.ensure_embedding_model_downloaded()
        assert result == "intfloat/multilingual-e5-base"
        fake_s3.download_file.assert_not_called()

    def test_empty_bucket_string_also_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_s3: MagicMock,
    ) -> None:
        # Pin: the truthiness check accepts both None and "" — env-var
        # users typically set ``S3_MODELS_BUCKET=`` (empty string) to
        # disable, not ``S3_MODELS_BUCKET=null``.
        monkeypatch.setattr(emb.settings, "s3_models_bucket", "")
        monkeypatch.setattr(emb.settings, "embedding_model", "intfloat/x")
        assert emb.ensure_embedding_model_downloaded() == "intfloat/x"

    def test_missing_marker_after_sync_raises(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # Pin: the post-sync check protects against a half-populated bucket
        # where the snapshot is missing config.json. Without this guard,
        # HuggingFace would silently fall back to a Hub lookup, defeating
        # the air-gap goal.
        fake_s3.get_paginator.return_value = _fake_paginator(
            [[{"Key": "models/multilingual-e5-base/tokenizer.json"}]]
        )

        def _materialise(_bucket: str, _key: str, dest: str) -> None:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"x")

        fake_s3.download_file.side_effect = _materialise

        with pytest.raises(RuntimeError, match=r"config\.json is missing"):
            emb.ensure_embedding_model_downloaded()


# ===================================================================== #
# _download_model_from_s3
# ===================================================================== #


@pytest.mark.unit
class TestDownloadModelFromS3:
    def test_iterates_paginator_pages(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # Pin: pagination is critical — a single page caps at 1000 keys
        # and HF snapshots can have more. Two pages, two files each.
        fake_s3.get_paginator.return_value = _fake_paginator(
            [
                [{"Key": "models/m/a.bin"}, {"Key": "models/m/b.bin"}],
                [{"Key": "models/m/c.bin"}],
            ]
        )
        emb._download_model_from_s3(
            bucket="quant-models-agile",
            prefix="models/m",
            local_dir=configured_settings / "m",
        )
        assert fake_s3.download_file.call_count == 3

    def test_preserves_nested_subdirectories(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # Pin: a key like ``models/m/sub/file`` lands at
        # ``{local_dir}/sub/file`` — flattening would silently break HF
        # tokenisers that look up files by relative path.
        fake_s3.get_paginator.return_value = _fake_paginator(
            [[{"Key": "models/m/sub/dir/file.json"}]]
        )
        emb._download_model_from_s3(
            bucket="b",
            prefix="models/m",
            local_dir=configured_settings / "m",
        )
        # Check the local_path argument to download_file.
        _bucket, _key, dest = fake_s3.download_file.call_args.args
        assert dest.endswith("/m/sub/dir/file.json")

    def test_skips_directory_marker_keys(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # Some S3 implementations (and console-uploaded folders) include a
        # zero-byte key matching the prefix itself, e.g. ``models/m/``.
        # Pin: we skip these — otherwise download_file would write a
        # zero-byte file at the *directory's* path and crash mkdir.
        fake_s3.get_paginator.return_value = _fake_paginator(
            [
                [
                    {"Key": "models/m/"},  # the directory marker
                    {"Key": "models/m/config.json"},
                ]
            ]
        )
        emb._download_model_from_s3(
            bucket="b",
            prefix="models/m",
            local_dir=configured_settings / "m",
        )
        assert fake_s3.download_file.call_count == 1
        _, key, _ = fake_s3.download_file.call_args.args
        assert key == "models/m/config.json"

    def test_empty_prefix_raises(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # No objects at the prefix → wrong bucket or typo. Pin: raise
        # with a clear message instead of silently leaving an empty cache
        # dir that HF will then try (and fail) to read from.
        fake_s3.get_paginator.return_value = _fake_paginator([[]])
        with pytest.raises(RuntimeError, match="No objects found"):
            emb._download_model_from_s3(
                bucket="b",
                prefix="models/missing",
                local_dir=configured_settings / "missing",
            )

    def test_uses_yandex_endpoint_and_creds_from_env(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin: same pattern as load_csv — Yandex endpoint + creds from env,
        # not from Pydantic settings. AWS rotates creds via docker-compose
        # env block, NOT via the application config file.
        captured: dict[str, Any] = {}

        def _capture(**kw: Any) -> MagicMock:
            captured["session"] = kw
            session = MagicMock()

            def _client(service: str, **client_kw: Any) -> MagicMock:
                captured["client"] = {"service": service, **client_kw}
                return fake_s3

            session.client = _client
            return session

        import boto3

        monkeypatch.setattr(boto3.session, "Session", _capture)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setattr(emb.settings, "s3_endpoint", "https://storage.yandexcloud.net")

        fake_s3.get_paginator.return_value = _fake_paginator([[{"Key": "models/m/x.bin"}]])
        emb._download_model_from_s3(
            bucket="b",
            prefix="models/m",
            local_dir=configured_settings / "m",
        )

        assert captured["session"]["aws_access_key_id"] == "AKIA-test"
        assert captured["session"]["aws_secret_access_key"] == "secret"
        # Region default — the deployment lives in ru-central1.
        assert captured["session"]["region_name"] == "ru-central1"
        # Custom endpoint MUST reach boto — without it the client would
        # try the AWS endpoint and 403.
        assert captured["client"]["endpoint_url"] == "https://storage.yandexcloud.net"

    def test_creates_local_dir_if_missing(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
    ) -> None:
        # Pin: parent directory creation is the SUT's job, not the
        # caller's — first-boot containers don't have ``/app/models``.
        fresh = configured_settings / "deep" / "nested" / "m"
        assert not fresh.exists()
        fake_s3.get_paginator.return_value = _fake_paginator([[{"Key": "models/m/config.json"}]])
        emb._download_model_from_s3(
            bucket="b",
            prefix="models/m",
            local_dir=fresh,
        )
        assert fresh.exists()


# ===================================================================== #
# get_embeddings — wired-through integration with the downloader
# ===================================================================== #


@pytest.mark.unit
class TestGetEmbeddings:
    def test_passes_local_path_to_huggingface_embeddings(
        self,
        configured_settings: Path,
        fake_s3: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pre-cache the marker so download is skipped — keep the test
        # focused on the wiring between the helper and HuggingFaceEmbeddings.
        target_dir = configured_settings / "multilingual-e5-base"
        target_dir.mkdir(parents=True)
        (target_dir / "config.json").write_text("{}", encoding="utf-8")

        captured: dict[str, Any] = {}

        def _factory(**kw: Any) -> MagicMock:
            captured.update(kw)
            return MagicMock(name="embeddings")

        monkeypatch.setattr(emb, "HuggingFaceEmbeddings", _factory)
        emb.get_embeddings()
        # Pin: the LOCAL filesystem path reaches HF — not the bare folder
        # name. A regression that passed only ``embedding_model`` would
        # send HF to the Hub for ``multilingual-e5-base`` (not a real ID).
        assert captured["model_name"] == str(target_dir)
        # Pin the keyword args that have been stable across versions —
        # CPU device + normalize_embeddings True is what every retriever
        # downstream relies on.
        assert captured["model_kwargs"]["device"] == "cpu"
        assert captured["encode_kwargs"]["normalize_embeddings"] is True


# ===================================================================== #
# Truncation helpers — pinned numeric semantics
# ===================================================================== #


@pytest.mark.unit
class TestTruncateAndNormalize:
    def test_truncates_to_target_dim(self) -> None:
        v = [3.0, 4.0, 5.0, 6.0]
        out = emb.truncate_and_normalize(v, dim=2)
        assert len(out) == 2
        # After truncation [3,4] → norm 5 → [0.6, 0.8]
        assert out[0] == pytest.approx(0.6)
        assert out[1] == pytest.approx(0.8)

    def test_zero_vector_returns_zero(self) -> None:
        # Pin: a zero vector stays zero — the norm guard prevents division
        # by zero. Cosine similarity of zero against anything is undefined
        # but Qdrant accepts the value without error.
        out = emb.truncate_and_normalize([0.0, 0.0, 0.0], dim=2)
        assert out == [0.0, 0.0]


@pytest.mark.unit
class TestGetTargetDim:
    def test_returns_setting_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(emb.settings, "embedding_dimension", 256)
        assert emb.get_target_dim(full_dim=768) == 256

    def test_returns_full_dim_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(emb.settings, "embedding_dimension", None)
        assert emb.get_target_dim(full_dim=768) == 768


@pytest.mark.unit
class TestTruncateVectors:
    def test_no_op_when_target_geq_full(self) -> None:
        # Pin: when target_dim >= full_dim, the SUT returns the input
        # *unchanged* (same object), avoiding a copy on every embed call.
        vectors = [[1.0, 2.0, 3.0]]
        out = emb.truncate_vectors(vectors, target_dim=3, full_dim=3)
        assert out is vectors

    def test_truncates_each_vector(self) -> None:
        out = emb.truncate_vectors(
            [[3.0, 4.0, 99.0], [5.0, 12.0, 99.0]],
            target_dim=2,
            full_dim=3,
        )
        # Each row becomes its first-2 dims, L2-renormalised.
        assert np.linalg.norm(out[0]) == pytest.approx(1.0)
        assert np.linalg.norm(out[1]) == pytest.approx(1.0)


@pytest.mark.unit
class TestTruncateVector:
    def test_no_op_when_target_geq_full(self) -> None:
        v = [1.0, 2.0]
        assert emb.truncate_vector(v, target_dim=2, full_dim=2) is v

    def test_truncates_when_target_smaller(self) -> None:
        out = emb.truncate_vector([3.0, 4.0, 99.0], target_dim=2, full_dim=3)
        assert len(out) == 2
        assert out[0] == pytest.approx(0.6)
        assert out[1] == pytest.approx(0.8)
