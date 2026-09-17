"""
Core embedding engine for the embedder package.

Provides the Embedder class, which loads one or more SentenceTransformer
models from Hugging Face (or local paths), encodes lists of text terms into
dense vector representations, and optionally applies UMAP dimensionality
reduction to the resulting embeddings.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from .EmbedderConfig       import EmbedderConfig
from logger                import Logger
import yaml
import umap

# Key under which embedder settings are expected to live in the YAML config file.
configuration_section: str = "embedder"

# Default path to the config file, used if no path is explicitly passed in.
standard_directory: str    = "./config/config.yaml"


class Embedder:
    """
    Core embedding engine responsible for encoding textual terms into dense
    vector representations and optionally applying dimensionality reduction.

    One or more SentenceTransformer models are loaded at construction time.
    Each call to embed() runs all loaded models and returns their outputs as
    a dictionary keyed by model ID. If dimensionality reduction is enabled,
    UMAP is applied to each model's output before it is returned.
    """

    def __init__(self, config: str = standard_directory):
        """
        Load and validate configuration, then initialise all configured
        SentenceTransformer models and (if enabled) the UMAP reducer.

        Parameters
        ----------
        config : str, optional
            Path to the YAML configuration file. Defaults to
            ./config/config.yaml relative to the working directory.
        """
        # Initialise instance attributes explicitly so each Embedder instance
        # has its own model registry and reducer, rather than sharing
        # class-level mutable defaults.
        self.models : dict = {}
        self.reducer       = None

        # Load and validate configuration from the YAML file.
        with open(config, "r") as f:
            data = yaml.safe_load(f)
        self.config = EmbedderConfig.model_validate(data[configuration_section])

        l = Logger()
        l.log("Loading model(s)...")

        if self.config.model_id:
            # Load each configured SentenceTransformer model and store it
            # in self.models, keyed by its model ID, so embed() can iterate
            # over all of them and return per-model results.
            for model_id in self.config.model_id:
                l.log(f"Loading embedding model '{model_id}' on device '{self.config.device}'...")
                model = SentenceTransformer(model_id, device=self.config.device)

                # Override the model's default max sequence length with the
                # configured value — inputs longer than this are truncated.
                model.max_seq_length = self.config.max_seq_length
                self.models[model_id] = model
                l.log(f"Loading embedding model '{model_id}' on device '{self.config.device}' completed.")

            if self.config.dimensionality_reduction:
                # Initialise the UMAP reducer with the configured parameters.
                # The reducer is fitted lazily inside embed() (via fit_transform),
                # so it learns the manifold structure from each batch of embeddings
                # rather than from a separate training set.
                l.log("Loading reducer...")
                self.reducer = umap.UMAP(
                    n_components = self.config.n_components,
                    n_neighbors  = self.config.n_neighbors,
                    min_dist     = self.config.min_dist,
                    metric       = self.config.metric,
                    random_state = self.config.random_state,
                )
                l.log("Loading reducer completed.")
        else:
            l.log("No model specified.")

        l.log("Loading model(s) completed.")

    def embed(self, terms: list) -> dict:
        """
        Encode a list of text terms into dense vector embeddings.

        Each loaded model produces one set of embeddings. If dimensionality
        reduction is enabled and the number of terms exceeds `n_components`,
        UMAP is applied to reduce each model's output before it is returned.

        Parameters
        ----------
        terms : list
            List of text strings to embed. Non-string and None values are
            coerced to strings; None is converted to an empty string.

        Returns
        -------
        dict
            Dictionary mapping each model ID to a numpy array of shape
            (len(terms), embedding_dim), where embedding_dim is either the
            model's native dimension or `n_components` if reduction is applied.
            Returns an empty dict if no models are loaded or `terms` is empty.
        """
        ret = {}

        if self.models and terms:
            # Coerce all term values to strings so the model never receives
            # unexpected types (e.g. None, int, float) from upstream callers.
            terms = [str(t) if t else "" for t in terms]

            for model_name, model in self.models.items():

                # Encode all terms in a single batched call. batch_size
                # controls how many terms are processed per forward pass —
                # larger values are faster on GPU but use more memory.
                embeddings = model.encode(
                    terms,
                    batch_size           = self.config.batch_size,
                    show_progress_bar    = False,
                    normalize_embeddings = self.config.normalization,
                    convert_to_numpy     = True,
                )

                # Apply dimensionality reduction only when enabled and when
                # there are strictly more terms than target dimensions —
                # UMAP requires at least n_components + 1 input points.
                if (
                    self.config.dimensionality_reduction
                    and self.reducer is not None
                    and len(embeddings) > self.config.n_components
                ):
                    embeddings = self.reducer.fit_transform(embeddings)

                ret[model_name] = embeddings

        return ret