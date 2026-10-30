"""
Test suite for the Embedder class and EmbedderConfig model.

Tests cover:
- EmbedderConfig: blank model IDs dropped, random_state=None accepted,
  pooling default and allowed values, out-of-range values rejected,
  revisions for unknown models rejected, backend and int8 quantisation
  options, lowercase options, duplicate model IDs rejected, revision keys
  stripped, UMAP metric names checked, the shipped config/config.yaml is
  valid.
- __init__: configuration from a YAML path (including the default path),
  a pathlib.Path, a dict (full configuration or embedder section) or an
  EmbedderConfig; missing, empty and invalid configuration, missing logger
  section; pooling warning, max_seq_length checked against the model;
  model loading, device, revision, backend and pooling assignment,
  max_seq_length, CPU
  thread limit, lowercasing decision, empty model list, logger
  configuration, one reducer per model when dimensionality reduction is
  enabled, none when disabled, no registries shared between instances.
- _apply_pooling(): pooling module replaced, other modules kept, previous
  mode returned.
- embed(): encode parameters, per-model full-dimensional output, empty
  model list, empty or None terms, term coercion (including pandas missing
  values), whitespace collapsing and lowercasing, numpy array, tuple,
  generator, array and single-string input, duplicate terms encoded once,
  NaN and infinite vectors returned and logged but not cached, JSON
  cache (reuse, only missing terms encoded, rounding, repair of an
  interrupted append, which settings start a new cache file, file names,
  one file per model), chunked encoding with progress messages and
  checkpoints.
- save_cache() / append_cache() / load_cache(): exact round trip, rounding,
  valid JSON after appends (also to a pretty-printed file), duplicates,
  NaN and infinity refused, malformed files and entries rejected,
  no file left half-written when a save is interrupted, complete entries
  recovered after an interrupted append or a broken line.
- reduce(): each model's own reducer applied, reduction skipped below
  n_components + 2 rows, error when dimensionality reduction is disabled
  or a model was not loaded.
- Package exports.
- Slow smoke tests with a real model and real UMAP (see TestRealModel).

Apart from the smoke test, SentenceTransformer, UMAP and Logger are mocked,
so no models are downloaded and no log files are written.

Run from the repository root with:
    pytest -v           # fast tests only (default)
    pytest -v -m slow   # real-model smoke tests only
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy  as np
import pytest
import yaml
from pydantic      import ValidationError
from unittest.mock import MagicMock, patch

import embedder
from embedder import Embedder, EmbedderConfig
from logger   import LoggerConfig

# The configuration file shipped with the repository.
_SHIPPED_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "config" / "config.yaml"

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


# Settings for the "logger" section of test configurations.
_LOGGER_SETTINGS = {
    "folder"       : "./logs/",
    "file_name"    : "out.log",
    "header_char"  : "-",
    "header_length": 60,
    "log_runtime"  : True,
    "format"       : "%Y-%m-%d %H:%M:%S",
    "encoding"     : "utf-8",
}


def _write_config(tmp_path, logger=None, **overrides):
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
    logger : dict or None, optional
        Content of a "logger" section; None writes no such section.
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
    data = {"embedder": cfg}
    if logger is not None:
        data["logger"] = logger
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
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

    def test_model_ids_are_stripped(self):
        assert EmbedderConfig(model_id=["  model-a "]).model_id == ["model-a"]

    def test_duplicate_model_ids_are_rejected(self):
        with pytest.raises(ValidationError, match="more than once"):
            EmbedderConfig(model_id=["model-a", " model-a "])

    def test_revision_keys_and_values_are_stripped(self):
        config = EmbedderConfig(model_id=[" model-a "], revision={" model-a ": " abc123 "})

        assert config.revision == {"model-a": "abc123"}

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan", "correlation"])
    def test_known_umap_metrics_are_accepted(self, metric):
        assert EmbedderConfig(metric=metric).metric == metric

    def test_unknown_umap_metric_is_rejected(self):
        with pytest.raises(ValidationError, match="unknown UMAP metric 'cosin'"):
            EmbedderConfig(metric="cosin")

    def test_shipped_config_file_is_valid(self):
        """
        config/config.yaml must stay valid for both packages that read it,
        since it is the documented starting point for users.
        """
        data = yaml.safe_load(_SHIPPED_CONFIG.read_text(encoding="utf-8"))

        config = EmbedderConfig.model_validate(data["embedder"])
        LoggerConfig.model_validate(data["logger"])

        assert config.model_id == ["cambridgeltl/SapBERT-from-PubMedBERT-fulltext"]
        assert set(config.revision) == set(config.model_id)


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
        embedder = Embedder(
            config={"logger": _LOGGER_SETTINGS, "embedder": {"model_id": ["model-a"]}}
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
        config_path = pathlib.Path(
            _write_config(tmp_path, logger=_LOGGER_SETTINGS, model_id=["model-a"])
        )

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
        config_path = _write_config(tmp_path, logger=_LOGGER_SETTINGS, model_id=[])

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

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_reads_the_default_config_path(self, mock_st, tmp_path, monkeypatch):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "config.yaml").write_text(
            "embedder:\n  model_id: [model-a]\n  batch_size: 7\n"
        )
        monkeypatch.chdir(tmp_path)

        embedder = Embedder()

        assert embedder.config.batch_size == 7
        assert set(embedder.models) == {"model-a"}

    def test_init_raises_when_the_config_file_is_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Embedder(config=str(tmp_path / "missing.yaml"))

    def test_init_raises_for_an_empty_config_file(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("")

        with pytest.raises(ValueError, match="no 'embedder' section"):
            Embedder(config=str(path))

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_uses_defaults_for_an_empty_embedder_section(self, mock_st, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("embedder:\n")

        embedder = Embedder(config=str(path))

        assert embedder.config == EmbedderConfig()
        mock_st.assert_not_called()

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_uses_defaults_for_an_empty_section_in_a_dict(self, mock_st):
        embedder = Embedder(config={"embedder": None})

        assert embedder.config == EmbedderConfig()

    def test_init_rejects_invalid_values_from_a_file(self, tmp_path):
        config_path = _write_config(tmp_path, batch_size=0)

        with pytest.raises(ValidationError, match="batch_size"):
            Embedder(config=config_path)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_instances_do_not_share_model_registries(self, mock_st, tmp_path):
        first  = Embedder(config=_write_config(tmp_path, model_id=["model-a"]))
        second = Embedder(config=_write_config(tmp_path, model_id=["model-b"]))

        assert set(first.models)  == {"model-a"}
        assert set(second.models) == {"model-b"}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_keeps_case_when_the_model_has_no_tokenizer(self, mock_st, tmp_path):
        model = MagicMock()
        model.tokenizer = None
        mock_st.return_value = model

        embedder = Embedder(config=_write_config(tmp_path, model_id=["model-a"]))

        assert embedder.lowercase == {"model-a": False}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_raises_before_loading_when_the_logger_section_is_missing(
        self, mock_st, mock_logger, tmp_path
    ):
        mock_logger.config = None
        config_path = _write_config(tmp_path, model_id=["model-a"])

        with pytest.raises(ValueError, match="no \\(or an empty\\) 'logger' section"):
            Embedder(config=config_path)

        mock_st.assert_not_called()

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_warns_when_pooling_replaces_the_models_own(
        self, mock_st, mock_apply_pooling, mock_logger, tmp_path
    ):
        mock_apply_pooling.return_value = "cls"

        Embedder(config=_write_config(tmp_path, model_id=["model-a"], pooling="mean"))

        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        assert any(
            "Warning: 'model-a': the configured 'mean' pooling replaces the model's own "
            "'cls' pooling." in m
            for m in logged
        )

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_does_not_warn_when_pooling_is_unchanged(
        self, mock_st, mock_apply_pooling, mock_logger, tmp_path
    ):
        mock_apply_pooling.return_value = "mean"

        Embedder(config=_write_config(tmp_path, model_id=["model-a"], pooling="mean"))

        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        assert not any("Warning" in m for m in logged)

    @pytest.mark.parametrize("max_seq_length, accepted", [(512, True), (513, False)])
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_checks_max_seq_length_against_the_model(
        self, mock_st, tmp_path, max_seq_length, accepted
    ):
        model = MagicMock()
        model[0].auto_model.config.max_position_embeddings = 512
        mock_st.return_value = model
        config_path = _write_config(
            tmp_path, model_id=["model-a"], max_seq_length=max_seq_length
        )

        if accepted:
            assert Embedder(config=config_path).models["model-a"].max_seq_length == 512
        else:
            with pytest.raises(ValueError, match="larger than the 512 tokens"):
                Embedder(config=config_path)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_passes_openvino_backend_without_extra_settings(self, mock_st, tmp_path):
        config_path = _write_config(tmp_path, model_id=["model-a"], backend="openvino")

        Embedder(config=config_path)

        mock_st.assert_called_once_with(
            "model-a", device="cpu", revision=None, backend="openvino", model_kwargs={}
        )


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
    def test_embed_treats_pandas_missing_values_as_empty(self, mock_st, tmp_path):
        pd = pytest.importorskip("pandas")
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        config_path = _write_config(tmp_path, model_id=["model-a"])
        embedder = Embedder(config=config_path)

        terms = pd.Series(["term", pd.NA, pd.NaT, np.float32("nan"), None], dtype=object)
        result = embedder.embed(terms)

        assert fake_model.encode.call_args.args[0] == ["term", ""]
        assert result["model-a"].shape == (5, 4)

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
        self, mock_st, mock_logger, tmp_path
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
        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        assert f"Warning: 'model-a': cache file '{cache_file}' could only be read in " \
               f"part; rewriting it with the 2 entries that could be read." in logged

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_with_none_returns_empty_dict(self, mock_st, tmp_path):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        embedder = Embedder(config=_write_config(tmp_path, model_id=["model-a"]))

        assert embedder.embed(None) == {}
        fake_model.encode.assert_not_called()

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_accepts_a_tuple_and_a_generator(self, mock_st, tmp_path):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        embedder = Embedder(config=_write_config(tmp_path, model_id=["model-a"]))

        from_tuple     = embedder.embed(("a", "bb"))["model-a"]
        from_generator = embedder.embed(t for t in ("a", "bb"))["model-a"]

        np.testing.assert_array_equal(from_tuple, from_generator)
        np.testing.assert_array_equal(from_tuple[:, 0], [1.0, 2.0])

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_serves_case_variants_from_one_cache_entry(self, mock_st, tmp_path):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        cache_folder = tmp_path / "cache"
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(cache_folder), lowercase=True
        )

        Embedder(config=config_path).embed(["Seizure"])
        Embedder(config=config_path).embed(["SEIZURE", "  seizure "])

        fake_model.encode.assert_called_once()
        cache_file, = cache_folder.glob("*.json")
        assert set(json.loads(cache_file.read_text())) == {"seizure"}

    @pytest.mark.parametrize(
        "changed",
        [
            {"revision": {"model-a": "abc123"}},
            {"normalization": False},
            {"max_seq_length": 64},
            {"backend": "openvino", "quantize_int8": True},
        ],
    )
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_uses_a_new_cache_file_when_output_settings_change(
        self, mock_st, tmp_path, changed
    ):
        mock_st.return_value = _fake_model()
        cache_folder = tmp_path / "cache"

        for overrides in ({}, changed):
            config_path = _write_config(
                tmp_path, model_id=["model-a"], cache_folder=str(cache_folder), **overrides
            )
            Embedder(config=config_path).embed(["a"])

        assert len(list(cache_folder.glob("*.json"))) == 2

    @pytest.mark.parametrize(
        "changed",
        [
            {"backend": "openvino"},
            {"lowercase": True},
            {"cache_decimals": 3},
            {"batch_size": 8},
            {"progress_interval": 5},
        ],
    )
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_keeps_the_cache_file_when_other_settings_change(
        self, mock_st, tmp_path, changed
    ):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        cache_folder = tmp_path / "cache"

        for overrides in ({}, changed):
            config_path = _write_config(
                tmp_path, model_id=["model-a"], cache_folder=str(cache_folder), **overrides
            )
            Embedder(config=config_path).embed(["a"])

        assert len(list(cache_folder.glob("*.json"))) == 1
        fake_model.encode.assert_called_once()

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_names_cache_files_safely_for_the_file_system(self, mock_st, tmp_path):
        mock_st.return_value = _fake_model()
        cache_folder = tmp_path / "cache"
        config_path = _write_config(
            tmp_path, model_id=["org/model name"], cache_folder=str(cache_folder)
        )

        Embedder(config=config_path).embed(["a"])

        # One file directly in the cache folder, not a subfolder per "/".
        cache_file, = cache_folder.iterdir()
        assert cache_file.is_file()
        assert cache_file.name.startswith("org_model_name-")

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_keeps_one_cache_file_per_model(self, mock_st, tmp_path):
        mock_st.side_effect = [_fake_model(), _fake_model()]
        cache_folder = tmp_path / "cache"
        config_path = _write_config(
            tmp_path, model_id=["model-a", "model-b"], cache_folder=str(cache_folder)
        )

        Embedder(config=config_path).embed(["a", "bb"])

        files = sorted(cache_folder.glob("*.json"))
        assert [f.name.split("-")[1] for f in files] == ["a", "b"]
        for cache_file in files:
            assert set(json.loads(cache_file.read_text())) == {"a", "bb"}

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_logs_cache_hits_and_does_not_rewrite_an_intact_cache(
        self, mock_st, mock_logger, tmp_path
    ):
        mock_st.return_value = _fake_model()
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(tmp_path / "cache")
        )
        Embedder(config=config_path).embed(["a"])
        mock_logger.return_value.log.reset_mock()

        Embedder(config=config_path).embed(["a", "bb", "bb"])

        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        assert any("1 distinct term(s) loaded from cache, 1 to encode" in m for m in logged)
        assert not any("read in part" in m for m in logged)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_keeps_the_text_of_an_array_term(self, mock_st, tmp_path):
        fake_model = _fake_model()
        mock_st.return_value = fake_model
        embedder = Embedder(config=_write_config(tmp_path, model_id=["model-a"]))

        embedder.embed([np.array([1, 2])])

        assert fake_model.encode.call_args.args[0] == ["[1 2]"]

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_returns_but_does_not_cache_non_finite_vectors(
        self, mock_st, mock_logger, tmp_path
    ):
        fake_model = MagicMock()
        fake_model.encode.side_effect = lambda terms, **kwargs: np.array(
            [[np.nan if t == "bad" else 1.0] * 4 for t in terms], dtype=np.float32
        )
        mock_st.return_value = fake_model
        cache_folder = tmp_path / "cache"
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(cache_folder)
        )

        result = Embedder(config=config_path).embed(["good", "bad"])["model-a"]

        assert np.isnan(result[1]).all() and np.isfinite(result[0]).all()
        cache_file, = cache_folder.glob("*.json")
        assert set(json.loads(cache_file.read_text())) == {"good"}
        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        assert any("NaN or infinite values for 1 term(s) ('bad'); they are not cached" in m
                   for m in logged)

        # The next run finds an intact cache and only encodes "bad" again.
        mock_logger.return_value.log.reset_mock()
        Embedder(config=config_path).embed(["good", "bad"])

        assert fake_model.encode.call_args.args[0] == ["bad"]
        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        assert not any("read in part" in m for m in logged)

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_warns_about_non_finite_vectors_without_a_cache(
        self, mock_st, mock_logger, tmp_path
    ):
        fake_model = MagicMock()
        fake_model.encode.side_effect = lambda terms, **kwargs: np.full(
            (len(terms), 4), np.inf, dtype=np.float32
        )
        mock_st.return_value = fake_model
        terms = [f"t{i}" for i in range(7)]

        Embedder(config=_write_config(tmp_path, model_id=["model-a"])).embed(terms)

        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        assert any(
            m.endswith("for 7 term(s) ('t0', 't1', 't2', 't3', 't4', ...).") for m in logged
        )

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_embed_keeps_exact_values_when_cache_decimals_is_none(self, mock_st, tmp_path):
        fake_model = MagicMock()
        fake_model.encode.side_effect = lambda terms, **kwargs: np.full(
            (len(terms), 4), 1 / 3, dtype=np.float32
        )
        mock_st.return_value = fake_model
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(tmp_path / "cache"),
            cache_decimals=None,
        )

        first  = Embedder(config=config_path).embed(["a"])["model-a"]
        second = Embedder(config=config_path).embed(["a"])["model-a"]

        np.testing.assert_array_equal(first, np.float32(1 / 3))
        np.testing.assert_array_equal(second, first)


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
        self, mock_st, mock_umap, mock_logger, tmp_path
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
        mock_logger.return_value.log.assert_called_with(
            "Warning: 'model-a': 3 row(s) given, at least 4 needed for dimensionality "
            "reduction; returning the full-dimensional embeddings."
        )

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_reduce_raises_when_dimensionality_reduction_disabled(self, mock_st, tmp_path):
        mock_st.return_value = MagicMock()
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=False
        )
        embedder = Embedder(config=config_path)

        with pytest.raises(RuntimeError, match="disabled"):
            embedder.reduce({"model-a": np.ones((10, 384))})

    @pytest.mark.parametrize("rows, reduced", [(4, False), (5, True)])
    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_reduce_threshold_follows_n_components(
        self, mock_st, mock_umap, tmp_path, rows, reduced
    ):
        """
        With n_components = 3, five rows (n_components + 2) is the smallest
        input that gets reduced.
        """
        reducer = mock_umap.UMAP.return_value
        reducer.fit_transform.return_value = np.zeros((rows, 3))
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=True, n_components=3
        )
        embedder = Embedder(config=config_path)

        result = embedder.reduce({"model-a": np.ones((rows, 384))})

        assert reducer.fit_transform.called is reduced
        assert result["model-a"].shape == ((rows, 3) if reduced else (rows, 384))

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_reduce_raises_for_a_model_that_was_not_loaded(
        self, mock_st, mock_umap, tmp_path
    ):
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=True
        )
        embedder = Embedder(config=config_path)

        with pytest.raises(KeyError, match="model-b"):
            embedder.reduce({"model-b": np.ones((10, 384))})

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_reduce_of_an_empty_dict_returns_an_empty_dict(
        self, mock_st, mock_umap, tmp_path
    ):
        config_path = _write_config(
            tmp_path, model_id=["model-a"], dimensionality_reduction=True
        )

        assert Embedder(config=config_path).reduce({}) == {}


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

    def test_save_to_a_file_name_without_folder(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        Embedder.save_cache("cache.json", {"a": np.ones(2, dtype=np.float32)})

        assert set(Embedder.load_cache(str(tmp_path / "cache.json"))) == {"a"}

    def test_save_keeps_non_ascii_terms_readable(self, tmp_path):
        path = tmp_path / "cache.json"

        Embedder.save_cache(str(path), {"Ödem": np.ones(2, dtype=np.float32)})

        assert '"Ödem"' in path.read_text(encoding="utf-8")

    def test_append_of_no_entries_leaves_the_file_unchanged(self, tmp_path):
        path = tmp_path / "cache.json"
        Embedder.save_cache(str(path), {"a": np.ones(2, dtype=np.float32)})
        before = path.read_bytes()

        Embedder.append_cache(str(path), {})
        Embedder.append_cache(str(tmp_path / "missing.json"), {})

        assert path.read_bytes() == before
        assert not (tmp_path / "missing.json").exists()

    def test_append_rounds_to_the_given_decimals(self, tmp_path):
        path = str(tmp_path / "cache.json")
        Embedder.save_cache(path, {})

        Embedder.append_cache(path, {"a": np.array([1 / 3], dtype=np.float32)}, decimals=2)

        assert json.loads(open(path, encoding="utf-8").read()) == {"a": [0.33]}

    def test_appended_duplicate_term_wins_when_loaded(self, tmp_path):
        path = str(tmp_path / "cache.json")
        Embedder.save_cache(path, {"a": np.zeros(2, dtype=np.float32)})

        Embedder.append_cache(path, {"a": np.ones(2, dtype=np.float32)})

        np.testing.assert_array_equal(Embedder.load_cache(path)["a"], [1.0, 1.0])

    def test_append_to_a_pretty_printed_file(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"a": [1.0, 2.0]}, indent=2))

        Embedder.append_cache(str(path), {"b": np.array([3.0, 4.0], dtype=np.float32)})

        assert json.loads(path.read_text()) == {"a": [1.0, 2.0], "b": [3.0, 4.0]}
        np.testing.assert_array_equal(Embedder.load_cache(str(path))["b"], [3.0, 4.0])

    def test_save_refuses_non_finite_values_and_keeps_the_previous_file(self, tmp_path):
        path = str(tmp_path / "cache.json")
        Embedder.save_cache(path, {"a": np.ones(2, dtype=np.float32)})

        with pytest.raises(ValueError, match="NaN or infinity"):
            Embedder.save_cache(path, {"b": np.array([np.nan, 1.0], dtype=np.float32)})
        with pytest.raises(ValueError, match="NaN or infinity"):
            Embedder.append_cache(path, {"c": np.array([np.inf], dtype=np.float32)})

        assert set(json.loads(open(path, encoding="utf-8").read())) == {"a"}

    @pytest.mark.parametrize(
        "text",
        [
            "[1, 2]",                       # not a JSON object
            '{"a": 5}',                     # a number instead of a list
            '{"a": {"x": 1}}',              # a nested object
            '{"a": []}',                    # an empty list
            '{"a": [NaN, 1.0]}',            # Python's json accepts NaN
            '{"a": ["x", "y"]}',            # strings
            "",                             # an empty file
        ],
    )
    def test_load_rejects_files_that_are_not_term_to_vector_objects(self, tmp_path, text):
        path = tmp_path / "cache.json"
        path.write_text(text)

        with pytest.raises(ValueError, match="could not be read"):
            Embedder.load_cache(str(path))

    def test_line_with_a_number_instead_of_a_list_counts_as_broken(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text('{\n"a": 5,\n"b": [1.0,2.0]\n}\n')

        cache, intact = Embedder._read_cache(str(path))

        assert set(cache) == {"b"}
        assert intact is False

    def test_broken_line_before_the_closing_brace_marks_the_file_incomplete(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text('{\n"a": [1.0],\n"b": [2.\n"c": [3.0]\n}\n')

        cache, intact = Embedder._read_cache(str(path))

        assert set(cache) == {"a", "c"}
        assert intact is False


class TestLogMessages:
    """
    Tests that the log messages follow one structure: start and end of a
    step as "<Action>..." and "<Action> completed.", messages about one
    model as "'<model_id>': ...", and warnings prefixed with "Warning: ".
    """

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "torch")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_messages_and_their_order(
        self, mock_st, mock_torch, mock_umap, mock_apply_pooling, mock_logger, tmp_path
    ):
        model = _fake_model()
        model.tokenizer.do_lower_case = True
        mock_st.return_value = model
        mock_apply_pooling.return_value = "cls"
        config_path = _write_config(
            tmp_path, model_id=["model-a"], revision={"model-a": "abc123"},
            num_threads=2, dimensionality_reduction=True,
        )

        Embedder(config=config_path)

        assert [c.args[0] for c in mock_logger.return_value.log.call_args_list] == [
            "Limiting the model(s) to 2 CPU thread(s).",
            "Loading model(s)...",
            "Loading embedding model 'model-a' (revision 'abc123', pooling 'mean', "
            "backend 'torch', device 'cpu')...",
            "Warning: 'model-a': the configured 'mean' pooling replaces the model's own "
            "'cls' pooling.",
            "Loading embedding model 'model-a' completed (terms are lowercased).",
            "Loading model(s) completed.",
            "Creating UMAP reducer(s)...",
            "Creating UMAP reducer(s) completed.",
        ]

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_init_messages_without_models(self, mock_st, mock_logger, tmp_path):
        Embedder(config=_write_config(tmp_path, model_id=[], dimensionality_reduction=True))

        assert [c.args[0] for c in mock_logger.return_value.log.call_args_list] == [
            "Loading model(s)...",
            "No model specified.",
            "Loading model(s) completed.",
        ]

    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_unpinned_openvino_int8_model_is_described_in_the_loading_message(
        self, mock_st, mock_logger, tmp_path
    ):
        Embedder(config=_write_config(
            tmp_path, model_id=["model-a"], backend="openvino", quantize_int8=True
        ))

        mock_logger.return_value.log.assert_any_call(
            "Loading embedding model 'model-a' (revision 'not pinned', pooling 'mean', "
            "backend 'openvino int8', device 'cpu')..."
        )

    @patch.object(_EMBEDDER_MODULE, "umap")
    @patch.object(_EMBEDDER_MODULE, "SentenceTransformer")
    def test_all_messages_follow_the_structure(
        self, mock_st, mock_umap, mock_apply_pooling, mock_logger, tmp_path
    ):
        """
        Run every path that logs (pooling warning, cache hits, progress,
        non-finite vectors, cache repair, skipped reduction) and check the
        structure of each message.
        """
        model = MagicMock()
        model.encode.side_effect = lambda terms, **kwargs: np.array(
            [[np.nan if t == "bad" else 1.0] * 4 for t in terms], dtype=np.float32
        )
        mock_st.return_value = model
        mock_apply_pooling.return_value = "cls"
        cache_folder = tmp_path / "cache"
        config_path = _write_config(
            tmp_path, model_id=["model-a"], cache_folder=str(cache_folder),
            dimensionality_reduction=True, progress_interval=1,
        )
        embedder = Embedder(config=config_path)
        embedder.embed(["good", "bad"])
        cache_file, = cache_folder.glob("*.json")
        cache_file.write_text(cache_file.read_text().rstrip().rstrip("}"))
        embedder.reduce(embedder.embed(["good", "other"]))

        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        warnings = [m for m in logged if m.startswith("Warning: ")]
        # Pooling, NaN, cache repair and skipped reduction.
        assert len(warnings) == 4
        for message in logged:
            assert message.endswith((".", "...")), message
            assert "Warning" not in message or message in warnings, message
        for message in warnings:
            assert re.match(r"Warning: 'model-a': \S", message), message
        for message in logged:
            if "encoded" in message or "from cache" in message:
                assert message.startswith("'model-a': "), message


class TestPackage:
    """
    Tests for what the embedder package exports.
    """

    def test_exports_embedder_and_config(self):
        assert embedder.__all__ == ["Embedder", "EmbedderConfig"]
        assert embedder.Embedder is Embedder
        assert embedder.EmbedderConfig is EmbedderConfig


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

    def test_returns_the_pooling_mode_the_model_came_with(self):
        Pooling = _EMBEDDER_MODULE.Pooling
        model = [MagicMock(), Pooling(384, pooling_mode="cls")]

        assert _REAL_APPLY_POOLING(model, "mean") == "cls"

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

    def test_pooling_warning_and_max_seq_length_check_with_a_real_model(
        self, tmp_path, mock_apply_pooling, mock_logger
    ):
        mock_apply_pooling.side_effect = _REAL_APPLY_POOLING
        # all-MiniLM-L6-v2 was trained with mean pooling and has 512
        # position embeddings.
        Embedder(config=_write_config(
            tmp_path, model_id=[self.MODEL_ID], revision={self.MODEL_ID: self.REVISION},
            pooling="cls", max_seq_length=512,
        ))

        logged = [c.args[0] for c in mock_logger.return_value.log.call_args_list]
        assert any("the configured 'cls' pooling replaces the model's own 'mean'" in m
                   for m in logged)

        with pytest.raises(ValueError, match="larger than the 512 tokens"):
            Embedder(config=_write_config(
                tmp_path, model_id=[self.MODEL_ID], revision={self.MODEL_ID: self.REVISION},
                max_seq_length=1024,
            ))

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
