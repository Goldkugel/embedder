"""
Test suite for the Embedder class and EmbedderConfig model.

Tests cover:
- EmbedderConfig: blank model IDs dropped, random_state=None accepted,
  out-of-range values rejected.
- __init__: model loading, device assignment, max_seq_length, empty model
  list, missing config section, logger configuration, one reducer per model
  when dimensionality reduction is enabled, none when disabled.
- embed(): encode parameters, per-model output, empty model list, empty
  terms, term coercion, numpy array and single-string input, reducer applied only when there
  are at least n_components + 2 terms, reducer skipped when dimensionality
  reduction is disabled.

SentenceTransformer, UMAP and Logger are mocked, so no models are
downloaded, no GPU is required and no log files are written.

Run with:
    pytest tests/EmbedderTest.py -v
"""

from __future__ import annotations

import sys

import numpy  as np
import pytest
import yaml
from pydantic      import ValidationError
from unittest.mock import MagicMock, patch

from embedder import Embedder, EmbedderConfig

# Resolve the actual module object Embedder lives in via sys.modules,
# rather than building a dotted string like "embedder.Embedder" for @patch
# to resolve via attribute lookup. The latter breaks here: this package's
# __init__.py does `from .Embedder import Embedder`, which rebinds the
# name "Embedder" inside the package namespace from "the submodule" to
# "the class" (since the module file and the class share the same name) —
# so patch("embedder.Embedder.SentenceTransformer") ends up resolving
# "embedder.Embedder" to the class, not the module, and fails to find
# SentenceTransformer on it. sys.modules[...] always returns the true
# module object regardless of that shadowing.
_EMBEDDER_MODULE = sys.modules[Embedder.__module__]


@pytest.fixture(autouse=True)
def mock_logger():
    """
    Replace Logger in the Embedder module for every test, so tests neither
    depend on ./config/config.yaml nor write to the real log file.
    """
    with patch.object(_EMBEDDER_MODULE, "Logger") as mock:
        yield mock


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


class TestEmbedderConfig:
    """
    Tests for EmbedderConfig validation.
    """

    def test_blank_model_ids_are_dropped(self):
        config = EmbedderConfig(model_id=["", "  ", "model-a"])

        assert config.model_id == ["model-a"]

    def test_random_state_accepts_none(self):
        assert EmbedderConfig(random_state=None).random_state is None

    @pytest.mark.parametrize(
        "field, value",
        [
            ("batch_size",     0),
            ("max_seq_length", 0),
            ("n_components",   0),
            ("n_neighbors",    1),
            ("min_dist",      -0.1),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field, value):
        with pytest.raises(ValidationError):
            EmbedderConfig(**{field: value})


class TestEmbedderInit:
    """
    Tests for Embedder.__init__.

    Verifies that models are loaded with the correct parameters, that one
    UMAP reducer per model is created only when dimensionality_reduction is
    True, and that an empty model list results in no SentenceTransformer calls.
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

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_with_blank_model_id_placeholder_loads_nothing(self, mock_st, tmp_path):
        config_path = _write_config(tmp_path, model_id=[""])

        embedder = Embedder(config=config_path)

        assert embedder.models == {}
        mock_st.assert_not_called()

    def test_init_raises_when_embedder_section_is_missing(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("logger:\n  folder: ./logs/\n")

        with pytest.raises(ValueError, match="embedder"):
            Embedder(config=str(path))

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_configures_unconfigured_logger_from_the_same_file(
        self, mock_st, mock_logger, tmp_path
    ):
        mock_logger.config = None
        config_path = _write_config(tmp_path, model_id=[])

        Embedder(config=config_path)

        mock_logger.assert_called_once_with(config_path)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_keeps_an_already_configured_logger(
        self, mock_st, mock_logger, tmp_path
    ):
        mock_logger.config = MagicMock()
        config_path = _write_config(tmp_path, model_id=[])

        Embedder(config=config_path)

        mock_logger.assert_called_once_with()

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_creates_one_reducer_per_model_when_dimensionality_reduction_enabled(
        self, mock_st, mock_umap, tmp_path
    ):
        """
        When dimensionality_reduction is True, one UMAP reducer per model
        must be constructed with the exact parameters from the config.
        """
        mock_st.return_value = MagicMock()
        reducer_a, reducer_b = MagicMock(), MagicMock()
        mock_umap.UMAP.side_effect = [reducer_a, reducer_b]
        config_path = _write_config(
            tmp_path,
            model_id                 = ["model-a", "model-b"],
            dimensionality_reduction = True,
            n_components             = 3,
            n_neighbors              = 10,
            min_dist                 = 0.5,
            metric                   = "euclidean",
            random_state             = None,
        )

        embedder = Embedder(config=config_path)

        assert mock_umap.UMAP.call_count == 2
        mock_umap.UMAP.assert_called_with(
            n_components = 3,
            n_neighbors  = 10,
            min_dist     = 0.5,
            metric       = "euclidean",
            random_state = None,
        )
        assert embedder.reducers == {"model-a": reducer_a, "model-b": reducer_b}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_does_not_create_reducers_when_dimensionality_reduction_disabled(
        self, mock_st, tmp_path
    ):
        mock_st.return_value = MagicMock()
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=False
        )

        embedder = Embedder(config=config_path)

        assert embedder.reducers == {}


class TestEmbed:
    """
    Tests for Embedder.embed().

    Verifies that encode() is called with the correct parameters, that
    results are keyed by model ID, that terms are coerced to strings, that
    each model's reducer is applied only when there are at least
    n_components + 2 terms, and that an empty model registry or an empty
    terms list returns an empty dict.
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
    def test_embed_coerces_terms_to_strings(self, mock_st, tmp_path):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((6, 4))
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"])
        embedder = Embedder(config=config_path)

        embedder.embed([None, float("nan"), 0, 0.0, False, "term"])

        encoded = fake_model.encode.call_args.args[0]
        assert encoded == ["", "", "0", "0.0", "False", "term"]

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_accepts_a_numpy_array(self, mock_st, tmp_path):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((2, 4))
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"])
        embedder = Embedder(config=config_path)

        result = embedder.embed(np.array(["a", "b"]))

        assert fake_model.encode.call_args.args[0] == ["a", "b"]
        assert set(result.keys()) == {"model-a"}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_treats_a_single_string_as_one_term(self, mock_st, tmp_path):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((1, 4))
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"])
        embedder = Embedder(config=config_path)

        embedder.embed("kidney")

        assert fake_model.encode.call_args.args[0] == ["kidney"]

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

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_with_no_terms_returns_empty_dict(self, mock_st, tmp_path):
        fake_model = MagicMock()
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"])
        embedder = Embedder(config=config_path)

        assert embedder.embed([]) == {}
        fake_model.encode.assert_not_called()

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_applies_each_models_own_reducer(
        self, mock_st, mock_umap, tmp_path
    ):
        """
        With n_components = 2, four terms (n_components + 2) is the smallest
        input that gets reduced. Each model's output must go through its own
        reducer.
        """
        model_a, model_b = MagicMock(), MagicMock()
        model_a.encode.return_value = np.ones((4, 384))
        model_b.encode.return_value = np.zeros((4, 384))
        mock_st.side_effect = [model_a, model_b]
        reducer_a, reducer_b = MagicMock(), MagicMock()
        reducer_a.fit_transform.return_value = np.ones((4, 2))
        reducer_b.fit_transform.return_value = np.zeros((4, 2))
        mock_umap.UMAP.side_effect = [reducer_a, reducer_b]

        config_path = _write_config(
            tmp_path,
            model_id                 = ["model-a", "model-b"],
            dimensionality_reduction = True,
            n_components             = 2,
        )
        embedder = Embedder(config=config_path)

        result = embedder.embed(["a", "b", "c", "d"])

        reducer_a.fit_transform.assert_called_once_with(model_a.encode.return_value)
        reducer_b.fit_transform.assert_called_once_with(model_b.encode.return_value)
        np.testing.assert_array_equal(result["model-a"], reducer_a.fit_transform.return_value)
        np.testing.assert_array_equal(result["model-b"], reducer_b.fit_transform.return_value)

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_skips_reducer_when_term_count_is_below_n_components_plus_two(
        self, mock_st, mock_umap, tmp_path
    ):
        """
        UMAP's spectral initialisation fails with n_components + 1 points
        (scipy raises "k >= N"), so 3 terms with n_components = 2 must be
        returned unreduced.
        """
        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones((3, 384))
        mock_st.return_value = fake_model
        fake_reducer = MagicMock()
        mock_umap.UMAP.return_value = fake_reducer

        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=True, n_components=2
        )
        embedder = Embedder(config=config_path)

        result = embedder.embed(["a", "b", "c"])

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
