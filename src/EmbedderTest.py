"""
Test suite for the Embedder class and EmbedderConfig model.

Tests cover:
- __init__: model loading, device assignment, max_seq_length, empty model
  list, reducer creation when dimensionality reduction is enabled/disabled.
- embed(): encode parameters, per-model output, empty model list,
  reducer applied when term count exceeds n_components, reducer skipped
  when term count does not exceed n_components, reducer skipped when
  dimensionality reduction is disabled.

Run with:
    pytest src/EmbedderTest.py -v

Run from the repository root so that the default config path
(./config/config.yaml) resolves correctly.
"""

from __future__ import annotations

import sys
import os

import numpy  as np
import pytest
import yaml
from unittest.mock import MagicMock, patch

from .Embedder import Embedder

# Resolve the actual module object Embedder lives in via sys.modules,
# rather than building a dotted string like "src.Embedder" for @patch to
# resolve via attribute lookup. The latter breaks here: this package's
# __init__.py does `from .Embedder import Embedder`, which rebinds the
# name "Embedder" inside the src package's namespace from "the submodule"
# to "the class" (since the module file and the class share the same
# name) — so patch("src.Embedder.SentenceTransformer") ends up resolving
# "src.Embedder" to the class, not the module, and fails to find
# SentenceTransformer on it. sys.modules[...] always returns the true
# module object regardless of that shadowing.
_EMBEDDER_MODULE = sys.modules[Embedder.__module__]


def _write_config(tmp_path, **overrides):
    """
    Write an "embedder" section config YAML to a temp file, with
    sensible defaults for every field Embedder reads, overridden by
    whatever the caller passes in.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory to write the config into.
    **overrides
        Any EmbedderConfig field to override from the defaults.

    Returns
    -------
    str
        Absolute path to the written config file.
    """
    cfg = {
        "model_id"                : ["fake/test-model"],
        "device"                  : "cpu",
        "batch_size"              : 64,
        "max_seq_length"          : 128,
        "normalization"           : True,
        "dimensionality_reduction": False,
        "n_components"            : 2,
        "n_neighbors"             : 5,
        "min_dist"                : 0.2,
        "metric"                  : "cosine",
        "random_state"            : 42,
    }
    cfg.update(overrides)
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"embedder": cfg}, f)
    return str(path)


class TestEmbedderInit:
    """
    Tests for Embedder.__init__.

    Verifies that models are loaded with the correct parameters, that the
    UMAP reducer is created only when dimensionality_reduction is True, and
    that an empty model list results in no SentenceTransformer calls.
    """

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_loads_each_configured_model(self, mock_st, tmp_path):
        config_path = _write_config(tmp_path, model_id=["model-a", "model-b"])
        mock_st.side_effect = lambda model_id, device: MagicMock(name=model_id)

        embedder = Embedder(config=config_path)

        assert mock_st.call_count == 2
        assert set(embedder.models.keys()) == {"model-a", "model-b"}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_passes_configured_device_to_each_model(self, mock_st, tmp_path):
        config_path = _write_config(tmp_path, model_id=["model-a"], device="cuda")
        mock_st.return_value = MagicMock()

        Embedder(config=config_path)

        mock_st.assert_called_once_with("model-a", device="cuda")

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_sets_max_seq_length_on_each_loaded_model(self, mock_st, tmp_path):
        config_path = _write_config(tmp_path, model_id=["model-a"], max_seq_length=256)
        fake_model = MagicMock()
        mock_st.return_value = fake_model

        embedder = Embedder(config=config_path)

        assert fake_model.max_seq_length == 256
        assert embedder.models["model-a"] is fake_model

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_with_no_configured_models_leaves_models_empty(self, mock_st, tmp_path):
        config_path = _write_config(tmp_path, model_id=[])

        embedder = Embedder(config=config_path)

        assert embedder.models == {}
        mock_st.assert_not_called()

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_creates_a_reducer_when_dimensionality_reduction_enabled(
        self, mock_st, mock_umap, tmp_path
    ):
        """
        When dimensionality_reduction is True, UMAP must be constructed
        with the exact parameters from the config.
        """
        mock_st.return_value = MagicMock()
        config_path = _write_config(
            tmp_path,
            model_id              = ["model-a"],
            dimensionality_reduction = True,
            n_components          = 3,
            n_neighbors           = 10,
            min_dist              = 0.5,
            metric                = "euclidean",
            random_state          = 7,
        )

        embedder = Embedder(config=config_path)

        mock_umap.UMAP.assert_called_once_with(
            n_components = 3,
            n_neighbors  = 10,
            min_dist     = 0.5,
            metric       = "euclidean",
            random_state = 7,
        )
        assert embedder.reducer is mock_umap.UMAP.return_value

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_does_not_create_a_reducer_when_dimensionality_reduction_disabled(
        self, mock_st, tmp_path
    ):
        mock_st.return_value = MagicMock()
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=False
        )

        embedder = Embedder(config=config_path)

        assert embedder.reducer is None


class TestEmbed:
    """
    Tests for Embedder.embed().

    Verifies that encode() is called with the correct parameters, that
    results are keyed by model ID, that the reducer is applied only when
    the term count strictly exceeds n_components, and that an empty model
    registry or an empty terms list returns an empty dict.
    """

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_calls_encode_with_configured_parameters(self, mock_st, tmp_path):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((2, 4))
        mock_st.return_value = fake_model
        config_path = _write_config(
            tmp_path, model_id=["model-a"], batch_size=16, normalization=False
        )
        embedder = Embedder(config=config_path)

        embedder.embed(["term one", "term two"])

        fake_model.encode.assert_called_once_with(
            ["term one", "term two"],
            batch_size           = 16,
            show_progress_bar    = False,
            normalize_embeddings = False,
            convert_to_numpy     = True,
        )

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_returns_one_entry_per_model(self, mock_st, tmp_path):
        model_a = MagicMock()
        model_a.encode.return_value = np.ones((3, 4))
        model_b = MagicMock()
        model_b.encode.return_value = np.zeros((3, 4))
        mock_st.side_effect = [model_a, model_b]
        config_path = _write_config(tmp_path, model_id=["model-a", "model-b"])
        embedder = Embedder(config=config_path)

        result = embedder.embed(["a", "b", "c"])

        assert set(result.keys()) == {"model-a", "model-b"}
        np.testing.assert_array_equal(result["model-a"], model_a.encode.return_value)
        np.testing.assert_array_equal(result["model-b"], model_b.encode.return_value)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_with_no_loaded_models_returns_empty_dict(self, mock_st, tmp_path):
        config_path = _write_config(tmp_path, model_id=[])
        embedder = Embedder(config=config_path)

        assert embedder.embed(["term"]) == {}

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_applies_reducer_when_enabled_and_term_count_exceeds_n_components(
        self, mock_st, mock_umap, tmp_path
    ):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones((5, 384))
        mock_st.return_value = fake_model
        fake_reducer = MagicMock()
        fake_reducer.fit_transform.return_value = np.ones((5, 2))
        mock_umap.UMAP.return_value = fake_reducer

        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=True, n_components=2
        )
        embedder = Embedder(config=config_path)

        result = embedder.embed(["a", "b", "c", "d", "e"])

        fake_reducer.fit_transform.assert_called_once()
        np.testing.assert_array_equal(
            result["model-a"], fake_reducer.fit_transform.return_value
        )

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_skips_reducer_when_term_count_does_not_exceed_n_components(
        self, mock_st, mock_umap, tmp_path
    ):
        """
        UMAP requires strictly more input points than n_components. Exactly
        n_components terms must not trigger the reducer.
        """
        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones((2, 384))
        mock_st.return_value = fake_model
        fake_reducer = MagicMock()
        mock_umap.UMAP.return_value = fake_reducer

        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=True, n_components=2
        )
        embedder = Embedder(config=config_path)

        # Exactly 2 terms == n_components; the guard uses strict greater-than,
        # so the reducer must NOT be applied here.
        result = embedder.embed(["a", "b"])

        fake_reducer.fit_transform.assert_not_called()
        np.testing.assert_array_equal(result["model-a"], fake_model.encode.return_value)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_does_not_apply_reducer_when_dimensionality_reduction_disabled(
        self, mock_st, tmp_path
    ):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones((10, 384))
        mock_st.return_value = fake_model
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=False
        )
        embedder = Embedder(config=config_path)

        result = embedder.embed(["a"] * 10)

        np.testing.assert_array_equal(result["model-a"], fake_model.encode.return_value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))