"""
Test suite for the Embedder class and EmbedderConfig model.

Tests cover:
- EmbedderConfig: blank model IDs dropped, random_state=None accepted,
  pooling default and allowed values, out-of-range values rejected,
  revisions for unknown models rejected, backend and int8 quantisation
  options, lowercase options.
- __init__: configuration from a YAML path, a pathlib.Path, a dict (full
  configuration or embedder section) or an EmbedderConfig; model loading,
  device, revision, backend and pooling assignment, max_seq_length, CPU
  thread limit, lowercasing decision, empty model list, missing config
  section, logger configuration, one reducer per model when dimensionality
  reduction is enabled, none when disabled.
- _apply_pooling(): pooling module replaced, other modules kept.
- embed(): encode parameters, per-model full-dimensional output, empty
  model list, empty terms, term coercion, whitespace collapsing and
  lowercasing, numpy array and single-string input, duplicate terms encoded
  once, JSON cache (reuse, only missing terms encoded, rounding, repair of
  an interrupted append), chunked encoding with progress messages and
  checkpoints.
- save_cache() / append_cache() / load_cache(): exact round trip, rounding,
  valid JSON after appends, no file left half-written when a save is
  interrupted, complete entries recovered after an interrupted append.
- reduce(): each model's own reducer applied, reduction skipped below
  n_components + 2 rows, error when dimensionality reduction is disabled.
- Slow smoke tests with a real model and real UMAP (see TestRealModel).

Apart from the smoke test, SentenceTransformer, UMAP and Logger are mocked,
so no models are downloaded and no log files are written.

Run with:
    pytest tests/EmbedderTest.py -v           # fast tests only (default)
    pytest tests/EmbedderTest.py -v -m slow   # real-model smoke tests only
"""

from __future__ import annotations

import json
import pathlib
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

# The real pooling function, kept before the autouse fixture below replaces
# it, so TestApplyPooling can still test it.
_REAL_APPLY_POOLING = Embedder._apply_pooling


@pytest.fixture(autouse=True)
def mock_logger():
    """
    Replace Logger in the Embedder module for every test, so tests neither
    depend on ./config/config.yaml nor write to the real log file.
    """
    with patch.object(_EMBEDDER_MODULE, "Logger") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_apply_pooling():
    """
    Replace Embedder._apply_pooling for every test: the MagicMock models
    used here have no real pooling module to swap. TestApplyPooling tests
    the real function separately.
    """
    with patch.object(Embedder, "_apply_pooling") as mock:
        yield mock


def _fake_model():
    """
    Return a MagicMock SentenceTransformer whose encode() returns one
    4-dimensional float32 row per input term, derived from the term's
    length, so results can be checked without a real model.
    """
    model = MagicMock()
    model.encode.side_effect = lambda terms, **kwargs: np.array(
        [[float(len(t))] * 4 for t in terms], dtype=np.float32
    ).reshape(len(terms), 4)
    return model


def _write_config(tmp_path, **overrides):
    """
    Write an "embedder" section config YAML to a temp file, with
    sensible defaults for the commonly overridden fields, overridden or
    extended by whatever the caller passes in. revision and cache_folder
    are left at their EmbedderConfig defaults (no pin, no cache) unless
    passed in.

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

    def test_pooling_defaults_to_mean(self):
        assert EmbedderConfig().pooling == "mean"

    def test_unknown_pooling_mode_is_rejected(self):
        with pytest.raises(ValidationError):
            EmbedderConfig(pooling="average")

    def test_revision_for_unknown_model_is_rejected(self):
        with pytest.raises(ValidationError, match="not in model_id"):
            EmbedderConfig(model_id=["model-a"], revision={"model-b": "abc123"})

    def test_backend_defaults_to_torch_without_quantization(self):
        config = EmbedderConfig()

        assert config.backend == "torch"
        assert config.quantize_int8 is False

    def test_int8_quantization_requires_the_openvino_backend(self):
        with pytest.raises(ValidationError, match="openvino"):
            EmbedderConfig(quantize_int8=True)

        assert EmbedderConfig(backend="openvino", quantize_int8=True).quantize_int8

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValidationError):
            EmbedderConfig(backend="onnx")

    @pytest.mark.parametrize("value", ["auto", True, False])
    def test_lowercase_accepts_auto_and_booleans(self, value):
        assert EmbedderConfig(lowercase=value).lowercase == value

    @pytest.mark.parametrize(
        "field, value",
        [
            ("batch_size",     0),
            ("max_seq_length", 0),
            ("n_components",   0),
            ("n_neighbors",    1),
            ("min_dist",      -0.1),
            ("progress_interval", 0),
            ("num_threads",    0),
            ("cache_decimals", 0),
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
        mock_st.side_effect = lambda model_id, **kwargs: MagicMock(name=model_id)

        embedder = Embedder(config=config_path)

        assert mock_st.call_count == 2
        assert set(embedder.models.keys()) == {"model-a", "model-b"}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_passes_configured_device_to_each_model(self, mock_st, tmp_path):
        config_path = _write_config(tmp_path, model_id=["model-a"], device="cuda")
        mock_st.return_value = MagicMock()

        Embedder(config=config_path)

        mock_st.assert_called_once_with("model-a", device="cuda", revision=None)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_passes_each_models_pinned_revision(self, mock_st, tmp_path):
        config_path = _write_config(
            tmp_path, model_id=["model-a", "model-b"], revision={"model-a": "abc123"}
        )
        mock_st.return_value = MagicMock()

        Embedder(config=config_path)

        assert mock_st.call_args_list[0].kwargs["revision"] == "abc123"
        assert mock_st.call_args_list[1].kwargs["revision"] is None

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_applies_configured_pooling_to_each_model(
        self, mock_st, mock_apply_pooling, tmp_path
    ):
        model_a, model_b = MagicMock(), MagicMock()
        mock_st.side_effect = [model_a, model_b]
        config_path = _write_config(
            tmp_path, model_id=["model-a", "model-b"], pooling="cls"
        )

        Embedder(config=config_path)

        assert mock_apply_pooling.call_args_list == [
            ((model_a, "cls"),), ((model_b, "cls"),)
        ]

    @patch.object(_EMBEDDER_MODULE, "torch")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_limits_cpu_threads_when_configured(self, mock_st, mock_torch, tmp_path):
        config_path = _write_config(tmp_path, model_id=[], num_threads=4)

        Embedder(config=config_path)

        mock_torch.set_num_threads.assert_called_once_with(4)

    @patch.object(_EMBEDDER_MODULE, "torch")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_keeps_default_threads_when_unset(self, mock_st, mock_torch, tmp_path):
        config_path = _write_config(tmp_path, model_id=[])

        Embedder(config=config_path)

        mock_torch.set_num_threads.assert_not_called()

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

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_accepts_an_embedder_config(self, mock_st):
        config = EmbedderConfig(model_id=["model-a"], batch_size=7)

        embedder = Embedder(config=config)

        assert embedder.config is config
        mock_st.assert_called_once_with("model-a", device="cpu", revision=None)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_accepts_a_dict_of_embedder_settings(self, mock_st):
        embedder = Embedder(config={"model_id": ["model-a"], "batch_size": 7})

        assert embedder.config.batch_size == 7
        assert set(embedder.models) == {"model-a"}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_accepts_a_full_configuration_dict(self, mock_st, mock_logger):
        mock_logger.config = None
        logger_settings = {
            "folder"       : "./logs/",
            "file_name"    : "out.log",
            "header_char"  : "-",
            "header_length": 60,
            "log_runtime"  : True,
            "format"       : "%Y-%m-%d %H:%M:%S",
            "encoding"     : "utf-8",
        }

        embedder = Embedder(
            config={"logger": logger_settings, "embedder": {"model_id": ["model-a"]}}
        )

        assert set(embedder.models) == {"model-a"}
        # The unconfigured logger is configured from the dict's logger section.
        assert mock_logger.config.file_name == "out.log"
        mock_logger.assert_called_once_with()

    def test_init_rejects_unknown_keys_in_a_dict(self):
        with pytest.raises(ValidationError):
            Embedder(config={"model_id": [], "batch_sise": 7})

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_accepts_a_pathlib_path(self, mock_st, mock_logger, tmp_path):
        mock_logger.config = None
        config_path = pathlib.Path(_write_config(tmp_path, model_id=["model-a"]))

        embedder = Embedder(config=config_path)

        assert set(embedder.models) == {"model-a"}
        mock_logger.assert_called_once_with(str(config_path))

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_passes_openvino_backend_threads_and_int8(self, mock_st, tmp_path):
        config_path = _write_config(
            tmp_path, model_id=["model-a"], backend="openvino",
            quantize_int8=True, num_threads=4,
        )

        Embedder(config=config_path)

        mock_st.assert_called_once_with(
            "model-a",
            device       = "cpu",
            revision     = None,
            backend      = "openvino",
            model_kwargs = {"ov_config": {"INFERENCE_NUM_THREADS": "4"}, "load_in_8bit": True},
        )

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_lowercases_automatically_for_uncased_tokenizers(self, mock_st, tmp_path):
        uncased, cased = MagicMock(), MagicMock()
        uncased.tokenizer.do_lower_case = True
        cased.tokenizer.do_lower_case   = False
        mock_st.side_effect = [uncased, cased]
        config_path = _write_config(tmp_path, model_id=["uncased", "cased"])

        embedder = Embedder(config=config_path)

        assert embedder.lowercase == {"uncased": True, "cased": False}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_lowercase_setting_overrides_the_tokenizer(self, mock_st, tmp_path):
        model = MagicMock()
        model.tokenizer.do_lower_case = True
        mock_st.return_value = model
        config_path = _write_config(tmp_path, model_id=["model-a"], lowercase=False)

        embedder = Embedder(config=config_path)

        assert embedder.lowercase == {"model-a": False}

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
    full-dimensional results are keyed by model ID, that terms are coerced
    to strings and encoded once each, that the JSON cache is reused and
    only missing terms are encoded, that encoding runs in chunks with a
    progress message and checkpoint after each, and that an empty model
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
    def test_embed_coerces_terms_to_strings(self, mock_st, tmp_path):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"])
        embedder = Embedder(config=config_path)

        result = embedder.embed([None, float("nan"), 0, 0.0, False, "term"])

        # None and NaN both become "", which is encoded only once.
        encoded = fake_model.encode.call_args.args[0]
        assert encoded == ["", "0", "0.0", "False", "term"]
        assert result["model-a"].shape == (6, 4)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_encodes_duplicate_terms_once_and_keeps_input_order(
        self, mock_st, tmp_path
    ):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"])
        embedder = Embedder(config=config_path)

        result = embedder.embed(["bb", "a", "bb"])

        assert fake_model.encode.call_args.args[0] == ["bb", "a"]
        np.testing.assert_array_equal(result["model-a"][:, 0], [2.0, 1.0, 2.0])

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_collapses_whitespace_and_lowercases_when_enabled(
        self, mock_st, tmp_path
    ):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"], lowercase=True)
        embedder = Embedder(config=config_path)

        result = embedder.embed(["Renal  cyst", " renal cyst ", "Renal\tCyst"])

        assert fake_model.encode.call_args.args[0] == ["renal cyst"]
        assert result["model-a"].shape == (3, 4)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_keeps_case_when_lowercasing_is_disabled(self, mock_st, tmp_path):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"], lowercase=False)
        embedder = Embedder(config=config_path)

        embedder.embed(["Seizure", "seizure"])

        assert fake_model.encode.call_args.args[0] == ["Seizure", "seizure"]

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
    def test_embed_never_reduces_even_when_reduction_is_enabled(
        self, mock_st, mock_umap, tmp_path
    ):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.ones((10, 384))
        mock_st.return_value = fake_model
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=True
        )
        embedder = Embedder(config=config_path)

        result = embedder.embed(["a"] * 10)

        mock_umap.UMAP.return_value.fit_transform.assert_not_called()
        np.testing.assert_array_equal(result["model-a"], fake_model.encode.return_value)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_serves_a_repeated_call_from_the_cache(self, mock_st, tmp_path):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        cache_folder = tmp_path / "cache"
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(cache_folder)
        )

        first  = Embedder(config=config_path).embed(["a", "bb"])
        # A new Embedder, as in a later run, reads the cache from disk.
        second = Embedder(config=config_path).embed(["a", "bb"])

        fake_model.encode.assert_called_once()
        assert len(list(cache_folder.glob("*.json"))) == 1
        np.testing.assert_array_equal(first["model-a"], second["model-a"])

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_only_encodes_terms_missing_from_the_cache(self, mock_st, tmp_path):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        cache_folder = tmp_path / "cache"
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(cache_folder)
        )
        embedder = Embedder(config=config_path)

        embedder.embed(["a", "bb"])
        result = embedder.embed(["bb", "ccc", "a"])

        assert fake_model.encode.call_args.args[0] == ["ccc"]
        np.testing.assert_array_equal(result["model-a"][:, 0], [2.0, 3.0, 1.0])
        cache_file, = cache_folder.glob("*.json")
        assert set(json.loads(cache_file.read_text())) == {"a", "bb", "ccc"}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_uses_a_new_cache_file_when_pooling_changes(self, mock_st, tmp_path):
        mock_st.return_value = _fake_model()
        cache_folder = tmp_path / "cache"

        for pooling in ("mean", "cls"):
            config_path = _write_config(
                tmp_path, model_id=["model-a"], cache_folder=str(cache_folder), pooling=pooling
            )
            Embedder(config=config_path).embed(["a"])

        assert len(list(cache_folder.glob("*.json"))) == 2

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_writes_no_cache_when_cache_folder_is_unset(self, mock_st, tmp_path):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"])
        embedder = Embedder(config=config_path)

        embedder.embed(["a"])
        embedder.embed(["a"])

        assert fake_model.encode.call_count == 2
        assert list(tmp_path.glob("**/*.json")) == []

    @patch.object(Embedder, "append_cache")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_logs_and_appends_a_checkpoint_after_every_chunk(
        self, mock_st, mock_append, mock_logger, tmp_path
    ):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(tmp_path / "cache"),
            progress_interval=2,
        )
        embedder = Embedder(config=config_path)

        result = embedder.embed(["a", "b", "c", "d", "e"])

        chunks = [c.args[0] for c in fake_model.encode.call_args_list]
        assert chunks == [["a", "b"], ["c", "d"], ["e"]]
        # Each checkpoint appends only the terms of its own chunk.
        appended = [list(c.args[1]) for c in mock_append.call_args_list]
        assert appended == [["a", "b"], ["c", "d"], ["e"]]
        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        assert any("encoded 2/5" in m for m in logged)
        assert any("encoded 5/5" in m for m in logged)
        assert result["model-a"].shape == (5, 4)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_rounds_cached_values_and_returns_them_rounded(self, mock_st, tmp_path):
        fake_model = MagicMock()
        fake_model.encode.side_effect = lambda terms, **kwargs: np.full(
            (len(terms), 4), 1 / 3, dtype=np.float32
        )
        mock_st.return_value = fake_model
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(tmp_path / "cache"),
            cache_decimals=2,
        )

        first  = Embedder(config=config_path).embed(["a"])["model-a"]
        second = Embedder(config=config_path).embed(["a"])["model-a"]

        np.testing.assert_array_equal(first, np.float32(0.33))
        np.testing.assert_array_equal(first, second)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_does_not_round_without_a_cache(self, mock_st, tmp_path):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.full((1, 4), 1 / 3, dtype=np.float32)
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"], cache_decimals=2)

        result = Embedder(config=config_path).embed(["a"])["model-a"]

        np.testing.assert_array_equal(result, np.float32(1 / 3))

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_repairs_a_cache_file_cut_off_by_an_interrupted_append(
        self, mock_st, tmp_path
    ):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        cache_folder = tmp_path / "cache"
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(cache_folder)
        )
        Embedder(config=config_path).embed(["a", "bb"])
        cache_file, = cache_folder.glob("*.json")
        # Simulate a job killed during an append: closing brace gone and
        # the last line only half written.
        text = cache_file.read_text().rstrip().rstrip("}")
        cache_file.write_text(text + ',\n"ccc": [3.0,3.')

        result = Embedder(config=config_path).embed(["a", "ccc"])

        # "a" survived, the half-written "ccc" is encoded again, and the
        # file is valid JSON again.
        assert fake_model.encode.call_args.args[0] == ["ccc"]
        np.testing.assert_array_equal(result["model-a"][:, 0], [1.0, 3.0])
        assert set(json.loads(cache_file.read_text())) == {"a", "bb", "ccc"}

class TestReduce:
    """
    Tests for Embedder.reduce().

    Verifies that each model's embeddings go through that model's own
    reducer, that inputs with fewer than n_components + 2 rows are returned
    unreduced, and that reduce() refuses to run when dimensionality
    reduction is disabled.
    """

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_reduce_applies_each_models_own_reducer(
        self, mock_st, mock_umap, tmp_path
    ):
        """
        With n_components = 2, four rows (n_components + 2) is the smallest
        input that gets reduced.
        """
        mock_st.return_value = MagicMock()
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
        full = {"model-a": np.ones((4, 384)), "model-b": np.zeros((4, 384))}

        result = embedder.reduce(full)

        reducer_a.fit_transform.assert_called_once_with(full["model-a"])
        reducer_b.fit_transform.assert_called_once_with(full["model-b"])
        np.testing.assert_array_equal(result["model-a"], reducer_a.fit_transform.return_value)
        np.testing.assert_array_equal(result["model-b"], reducer_b.fit_transform.return_value)

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_reduce_skips_inputs_below_n_components_plus_two(
        self, mock_st, mock_umap, tmp_path
    ):
        """
        UMAP's spectral initialisation fails with n_components + 1 points
        (scipy raises "k >= N"), so 3 rows with n_components = 2 must be
        returned unreduced.
        """
        mock_st.return_value = MagicMock()
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=True, n_components=2
        )
        embedder = Embedder(config=config_path)
        full = {"model-a": np.ones((3, 384))}

        result = embedder.reduce(full)

        mock_umap.UMAP.return_value.fit_transform.assert_not_called()
        np.testing.assert_array_equal(result["model-a"], full["model-a"])

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_reduce_raises_when_dimensionality_reduction_disabled(self, mock_st, tmp_path):
        mock_st.return_value = MagicMock()
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=False
        )
        embedder = Embedder(config=config_path)

        with pytest.raises(RuntimeError, match="disabled"):
            embedder.reduce({"model-a": np.ones((10, 384))})


class TestCacheFile:
    """
    Tests for Embedder.save_cache() and Embedder.load_cache().
    """

    def test_round_trip_is_exact_and_file_is_valid_json(self, tmp_path):
        path = str(tmp_path / "sub" / "cache.json")
        rng = np.random.default_rng(0)
        cache = {
            "Seizure"           : rng.normal(size=8).astype(np.float32),
            'Quote " and ünïcode': rng.normal(size=8).astype(np.float32),
            ""                  : rng.normal(size=8).astype(np.float32),
        }

        Embedder.save_cache(path, cache)
        loaded = Embedder.load_cache(path)

        assert set(json.loads(open(path, encoding="utf-8").read())) == set(cache)
        for term, vector in cache.items():
            assert loaded[term].dtype == np.float32
            np.testing.assert_array_equal(loaded[term], vector)

    def test_interrupted_save_keeps_the_previous_file(self, tmp_path):
        path = str(tmp_path / "cache.json")
        Embedder.save_cache(path, {"a": np.ones(4, dtype=np.float32)})

        # A value that cannot be converted to numbers makes the second save
        # fail halfway through writing.
        with pytest.raises(ValueError):
            Embedder.save_cache(
                path, {"a": np.ones(4, dtype=np.float32), "b": ["not", "numbers"]}
            )

        assert set(Embedder.load_cache(path)) == {"a"}

    def test_successful_save_leaves_no_tmp_file(self, tmp_path):
        path = tmp_path / "cache.json"

        Embedder.save_cache(str(path), {"a": np.ones(4, dtype=np.float32)})

        assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]

    def test_save_rounds_to_the_given_decimals(self, tmp_path):
        path = str(tmp_path / "cache.json")

        Embedder.save_cache(path, {"a": np.array([1 / 3, -2 / 3], dtype=np.float32)}, decimals=3)

        assert json.loads(open(path, encoding="utf-8").read()) == {"a": [0.333, -0.667]}

    def test_append_adds_entries_and_keeps_valid_json(self, tmp_path):
        path = str(tmp_path / "cache.json")
        Embedder.save_cache(path, {"a": np.ones(3, dtype=np.float32)})

        Embedder.append_cache(path, {"b": np.zeros(3, dtype=np.float32)})
        Embedder.append_cache(path, {"c": np.full(3, 2, dtype=np.float32),
                                     'd "quoted"': np.ones(3, dtype=np.float32)})

        assert set(json.loads(open(path, encoding="utf-8").read())) == {"a", "b", "c", 'd "quoted"'}
        loaded = Embedder.load_cache(path)
        np.testing.assert_array_equal(loaded["c"], [2.0, 2.0, 2.0])

    def test_append_to_an_empty_cache_and_to_a_missing_file(self, tmp_path):
        empty   = str(tmp_path / "empty.json")
        missing = str(tmp_path / "sub" / "missing.json")
        Embedder.save_cache(empty, {})

        Embedder.append_cache(empty,   {"a": np.ones(2, dtype=np.float32)})
        Embedder.append_cache(missing, {"a": np.ones(2, dtype=np.float32)})

        for path in (empty, missing):
            assert set(json.loads(open(path, encoding="utf-8").read())) == {"a"}

    def test_append_refuses_a_file_without_closing_brace(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text('{\n"a": [1.0]')

        with pytest.raises(ValueError, match="closing|cut off"):
            Embedder.append_cache(str(path), {"b": np.ones(1, dtype=np.float32)})

    def test_load_recovers_complete_entries_after_an_interrupted_append(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text('{\n"a": [1.0,2.0],\n"b": [3.0,4.0],\n"c": [5.')

        loaded = Embedder.load_cache(str(path))

        assert set(loaded) == {"a", "b"}

    def test_load_rejects_an_unreadable_file(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("not json")

        with pytest.raises(ValueError, match="could not be read"):
            Embedder.load_cache(str(path))

    def test_load_of_a_missing_file_returns_an_empty_cache(self, tmp_path):
        assert Embedder.load_cache(str(tmp_path / "missing.json")) == {}

    def test_load_accepts_json_in_another_layout(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text('{"a": [1.5, 2.5], "b": [3, 4]}')

        loaded = Embedder.load_cache(str(path))

        np.testing.assert_array_equal(loaded["a"], [1.5, 2.5])
        np.testing.assert_array_equal(loaded["b"], [3.0, 4.0])


class TestApplyPooling:
    """
    Tests for the real Embedder._apply_pooling(), using real
    sentence-transformers Pooling modules (no model download needed).
    """

    def test_replaces_the_pooling_module_and_keeps_the_others(self):
        Pooling = _EMBEDDER_MODULE.Pooling
        transformer, normalize = MagicMock(), MagicMock()
        model = [transformer, Pooling(384, pooling_mode="mean"), normalize]

        _REAL_APPLY_POOLING(model, "cls")

        assert model[0] is transformer and model[2] is normalize
        assert isinstance(model[1], Pooling)
        assert model[1].get_config_dict()["pooling_mode"] == "cls"
        assert model[1].get_config_dict()["embedding_dimension"] == 384

    def test_raises_when_the_model_has_no_pooling_module(self):
        with pytest.raises(ValueError, match="no pooling module"):
            _REAL_APPLY_POOLING([MagicMock()], "cls")


@pytest.mark.slow
class TestRealModel:
    """
    Smoke test against the real sentence-transformers and UMAP libraries,
    to catch API changes the mocked tests cannot see. Downloads a small
    model (~90 MB) on first use and runs on the CPU. Deselected by default;
    run with `pytest -m slow`.
    """

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    def test_embed_reduce_and_cache_with_a_real_model(self, tmp_path):
        cache_folder = tmp_path / "cache"
        config_path = _write_config(
            tmp_path,
            model_id                 = [self.MODEL_ID],
            revision                 = {self.MODEL_ID: self.REVISION},
            cache_folder             = str(cache_folder),
            dimensionality_reduction = True,
            n_components             = 2,
            # Below the six terms, so UMAP does not have to lower it.
            n_neighbors              = 3,
        )
        terms = [
            "Microcephaly",
            "Decreased skull size",
            "Abnormality of the kidney",
            "Renal dysplasia",
            "Seizure",
            "Epileptic fit",
        ]
        embedder = Embedder(config=config_path)

        full = embedder.embed(terms)[self.MODEL_ID]

        # all-MiniLM-L6-v2 produces 384-dimensional, L2-normalised vectors.
        assert full.shape == (len(terms), 384)
        np.testing.assert_allclose(np.linalg.norm(full, axis=1), 1.0, rtol=1e-5)

        # Synonyms must be closer to each other than to an unrelated term.
        similarity = full @ full.T
        assert similarity[4, 5] > similarity[4, 2]

        # A repeated call is served from the JSON cache with identical vectors.
        assert len(list(cache_folder.glob("*.json"))) == 1
        np.testing.assert_array_equal(embedder.embed(terms)[self.MODEL_ID], full)

        reduced = embedder.reduce({self.MODEL_ID: full})[self.MODEL_ID]
        assert reduced.shape == (len(terms), 2)

    def test_pooling_setting_changes_the_vectors(self, tmp_path, mock_apply_pooling):
        # Use the real pooling swap here instead of the autouse mock.
        mock_apply_pooling.side_effect = _REAL_APPLY_POOLING
        vectors = {}
        for pooling in ("mean", "cls"):
            config_path = _write_config(
                tmp_path,
                model_id = [self.MODEL_ID],
                revision = {self.MODEL_ID: self.REVISION},
                pooling  = pooling,
            )
            vectors[pooling] = Embedder(config=config_path).embed(["Seizure"])[self.MODEL_ID]

        assert vectors["mean"].shape == vectors["cls"].shape == (1, 384)
        assert not np.allclose(vectors["mean"], vectors["cls"])

    def test_openvino_backend_matches_torch(self, tmp_path):
        # Needs the "openvino" extra; skipped where it is not installed.
        pytest.importorskip("optimum.intel")
        terms = ["Microcephaly", "Seizure", "Renal dysplasia"]
        vectors = {}
        for backend, int8 in (("torch", False), ("openvino", False), ("openvino", True)):
            config_path = _write_config(
                tmp_path,
                model_id      = [self.MODEL_ID],
                revision      = {self.MODEL_ID: self.REVISION},
                backend       = backend,
                quantize_int8 = int8,
            )
            vectors[backend, int8] = Embedder(config=config_path).embed(terms)[self.MODEL_ID]

        reference = vectors["torch", False]
        # Full precision: the same vectors up to float noise.
        np.testing.assert_allclose(vectors["openvino", False], reference, atol=1e-5)
        # int8 weights: close, but not identical.
        cosine = (vectors["openvino", True] * reference).sum(axis=1)
        assert cosine.min() > 0.99


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
