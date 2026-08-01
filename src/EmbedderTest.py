import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from Embedder import Embedder
from EmbedderConfig import EmbedderConfig


@pytest.fixture
def create_yaml_config(tmp_path):
    """Helper fixture to write YAML string content to a temporary file."""
    def _create(yaml_content: str) -> str:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")
        return str(config_file)
    return _create


@pytest.fixture
def standard_yaml_str():
    """Return a standard raw YAML configuration string with multiple models."""
    return """
embedder:
  model_id:
    - "mock/model-a"
    - "mock/model-b"
  device: "cpu"
  batch_size: 32
  max_seq_length: 64
  normalization: true
  dimensionality_reduction: true
  n_components: 2
  n_neighbors: 5
  min_dist: 0.2
  metric: "cosine"
  random_state: 42
"""


@pytest.fixture
def single_model_yaml_str():
    """Return a YAML configuration string with a single model."""
    return """
embedder:
  model_id:
    - "mock/model-a"
  device: "cpu"
  batch_size: 32
  max_seq_length: 64
  normalization: true
  dimensionality_reduction: true
  n_components: 2
  n_neighbors: 5
  min_dist: 0.2
  metric: "cosine"
  random_state: 42
"""


class TestEmbedderInit:

    @patch("Embedder.Logger")
    @patch("Embedder.umap.UMAP")
    @patch("Embedder.SentenceTransformer")
    def test_init_loads_config_and_initializes_models(
        self, mock_sentence_transformer, mock_umap, mock_logger, create_yaml_config, standard_yaml_str
    ):
        config_path = create_yaml_config(standard_yaml_str)
        mock_sentence_transformer.side_effect = lambda model_id, device=None: MagicMock()

        embedder = Embedder(config_path)

        assert isinstance(embedder.config, EmbedderConfig)
        assert embedder.config.model_id == ["mock/model-a", "mock/model-b"]
        assert len(embedder.models) == 2
        assert "mock/model-a" in embedder.models
        assert "mock/model-b" in embedder.models
        assert mock_sentence_transformer.call_count == 2

    @patch("Embedder.Logger")
    @patch("Embedder.umap.UMAP")
    @patch("Embedder.SentenceTransformer")
    def test_init_sets_max_seq_length_on_loaded_models(
        self, mock_sentence_transformer, mock_umap, mock_logger, create_yaml_config, standard_yaml_str
    ):
        config_path = create_yaml_config(standard_yaml_str)
        mock_sentence_transformer.side_effect = lambda model_id, device=None: MagicMock()

        embedder = Embedder(config_path)

        assert len(embedder.models) == 2
        for model_id, model_obj in embedder.models.items():
            assert model_obj.max_seq_length == 64

    @patch("Embedder.Logger")
    @patch("Embedder.umap.UMAP")
    @patch("Embedder.SentenceTransformer")
    def test_init_initializes_umap_reducer_when_reduction_is_enabled(
        self, mock_sentence_transformer, mock_umap, mock_logger, create_yaml_config, single_model_yaml_str
    ):
        config_path = create_yaml_config(single_model_yaml_str)

        embedder = Embedder(config_path)

        assert embedder.reducer is not None
        mock_umap.assert_called_once_with(
            n_components=2,
            n_neighbors=5,
            min_dist=0.2,
            metric="cosine",
            random_state=42,
        )

    @patch("Embedder.Logger")
    @patch("Embedder.SentenceTransformer")
    def test_init_skips_reducer_when_dimensionality_reduction_is_false(
        self, mock_sentence_transformer, mock_logger, create_yaml_config
    ):
        yaml_no_reduction = """
embedder:
  model_id: ["mock/model-a"]
  dimensionality_reduction: false
"""
        config_path = create_yaml_config(yaml_no_reduction)

        embedder = Embedder(config_path)

        assert embedder.reducer is None

    @patch("Embedder.Logger")
    def test_init_handles_empty_model_id_list(
        self, mock_logger, create_yaml_config
    ):
        yaml_no_models = """
embedder:
  model_id: []
"""
        config_path = create_yaml_config(yaml_no_models)

        embedder = Embedder(config_path)

        assert len(embedder.models) == 0
        assert embedder.reducer is None


class TestEmbedderEmbed:

    @patch("Embedder.Logger")
    @patch("Embedder.umap.UMAP")
    @patch("Embedder.SentenceTransformer")
    def test_embed_returns_embeddings_dictionary_for_all_configured_models(
        self, mock_sentence_transformer, mock_umap, mock_logger, create_yaml_config, standard_yaml_str
    ):
        config_path = create_yaml_config(standard_yaml_str)

        # Create distinct mock models for each model entry
        def create_mock_model(model_id, device=None):
            m = MagicMock()
            m.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            return m

        mock_sentence_transformer.side_effect = create_mock_model

        # Mock UMAP fit_transform behavior
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mock_umap.return_value = mock_reducer

        embedder = Embedder(config_path)
        sample_terms = ["term_1", "term_2", "term_3"]

        results = embedder.embed(sample_terms)

        assert isinstance(results, dict)
        assert set(results.keys()) == {"mock/model-a", "mock/model-b"}
        assert results["mock/model-a"].shape == (3, 2)
        assert results["mock/model-b"].shape == (3, 2)

    @patch("Embedder.Logger")
    @patch("Embedder.SentenceTransformer")
    def test_embed_passes_configured_batch_size_and_normalization_to_model(
        self, mock_sentence_transformer, mock_logger, create_yaml_config
    ):
        yaml_str = """
embedder:
  model_id: ["mock/model-a"]
  batch_size: 16
  normalization: true
  dimensionality_reduction: false
"""
        config_path = create_yaml_config(yaml_str)
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        embedder = Embedder(config_path)
        embedder.embed(["term_1", "term_2"])

        mock_model.encode.assert_called_once_with(
            ["term_1", "term_2"],
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    @patch("Embedder.Logger")
    @patch("Embedder.umap.UMAP")
    @patch("Embedder.SentenceTransformer")
    def test_embed_skips_reduction_if_sample_count_is_less_than_or_equal_to_n_components(
        self, mock_sentence_transformer, mock_umap, mock_logger, create_yaml_config, single_model_yaml_str
    ):
        config_path = create_yaml_config(single_model_yaml_str)

        # Return 2 samples when n_components is 2 (len(embeddings) <= n_components)
        mock_model = MagicMock()
        raw_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = raw_embeddings
        mock_sentence_transformer.return_value = mock_model

        mock_reducer = MagicMock()
        mock_umap.return_value = mock_reducer

        embedder = Embedder(config_path)
        results = embedder.embed(["term_1", "term_2"])

        # fit_transform should NOT be called because sample count (2) <= n_components (2)
        mock_reducer.fit_transform.assert_not_called()
        np.testing.assert_array_equal(results["mock/model-a"], raw_embeddings)

    @patch("Embedder.Logger")
    def test_embed_returns_empty_dict_when_no_models_are_configured(
        self, mock_logger, create_yaml_config
    ):
        yaml_no_models = """
embedder:
  model_id: []
"""
        config_path = create_yaml_config(yaml_no_models)

        embedder = Embedder(config_path)
        results = embedder.embed(["term_1", "term_2"])

        assert results == {}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))