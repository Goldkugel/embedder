"""
Core embedding engine for the embedder package.

Provides the Embedder class, which loads one or more SentenceTransformer
models from Hugging Face (or local paths), encodes lists of text terms into
dense vector representations, and optionally applies UMAP dimensionality
reduction to the resulting embeddings.
"""

from __future__ import annotations

from collections.abc       import Iterable
from sentence_transformers import SentenceTransformer
from .EmbedderConfig       import EmbedderConfig
from logger                import Logger
import math
import yaml
import umap

# Key under which embedder settings are expected to live in the YAML config file.
configuration_section: str = "embedder"

# Default path to the config file, used if no path is explicitly passed in.
standard_directory: str    = "./config/config.yaml"


def _to_text(term) -> str:
    """
    Coerce a single term to a string for the model. None and NaN (e.g.
    missing values from a pandas column) become an empty string; every
    other value, including falsy ones like 0 or False, keeps its string form.
    """
    if term is None or (isinstance(term, float) and math.isnan(term)):
        return ""
    return str(term)


class Embedder:
    """
    Core embedding engine responsible for encoding textual terms into dense
    vector representations and optionally applying dimensionality reduction.

    One or more SentenceTransformer models are loaded at construction time.
    Each call to embed() runs all loaded models and returns their outputs as
    a dictionary keyed by model ID. If dimensionality reduction is enabled,
    each model has its own UMAP reducer, which is fitted afresh on every
    embed() call. Reduced coordinates are therefore only comparable within
    the output of a single call, not across calls.
    """

    def __init__(self, config: str = standard_directory):
        """
        Load and validate configuration, then initialise all configured
        SentenceTransformer models and (if enabled) one UMAP reducer per model.

        Parameters
        ----------
        config : str, optional
            Path to the YAML configuration file. Defaults to
            ./config/config.yaml relative to the working directory.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If the configuration file has no "embedder" section.
        pydantic.ValidationError
            If the "embedder" section contains unknown keys or invalid values.
        """
        # Initialise instance attributes explicitly so each Embedder instance
        # has its own model and reducer registries, rather than sharing
        # class-level mutable defaults.
        self.models   : dict = {}
        self.reducers : dict = {}

        # Load and validate configuration from the YAML file.
        # An empty file parses to None, so fall back to an empty dict to get
        # the clearer "missing section" error below instead of a TypeError.
        with open(config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if configuration_section not in data:
            raise ValueError(
                f"Configuration file '{config}' has no '{configuration_section}' section."
            )
        # A section that is present but empty (`embedder:` with nothing
        # below it) also parses to None; treat it as "use all defaults".
        self.config = EmbedderConfig.model_validate(data[configuration_section] or {})

        # The logger is a process-wide singleton. If the entry point has not
        # configured it yet, configure it from the same file as the embedder
        # instead of falling back to ./config/config.yaml.
        l = Logger() if Logger.config is not None else Logger(config)
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
                # Initialise one UMAP reducer per model with the configured
                # parameters, so fitting on one model's embeddings never
                # overwrites the fitted state belonging to another model.
                # Each reducer is fitted inside embed() (via fit_transform).
                l.log("Loading reducer(s)...")
                for model_id in self.models:
                    self.reducers[model_id] = umap.UMAP(
                        n_components = self.config.n_components,
                        n_neighbors  = self.config.n_neighbors,
                        min_dist     = self.config.min_dist,
                        metric       = self.config.metric,
                        random_state = self.config.random_state,
                    )
                l.log("Loading reducer(s) completed.")
        else:
            l.log("No model specified.")

        l.log("Loading model(s) completed.")

    def embed(self, terms: str | Iterable) -> dict:
        """
        Encode a sequence of text terms into dense vector embeddings.

        Each loaded model produces one set of embeddings. If dimensionality
        reduction is enabled, UMAP is applied to each model's output, provided
        there are at least `n_components + 2` terms (UMAP's spectral
        initialisation fails on fewer points). With fewer terms, the
        full-dimensional embeddings are returned and a message is logged.

        Parameters
        ----------
        terms : str or Iterable
            Text strings to embed — a list, tuple, numpy array or pandas
            Series. A single string is treated as one term. None and NaN
            become an empty string; any other value is converted with str().

        Returns
        -------
        dict
            Dictionary mapping each model ID to a numpy array of shape
            (len(terms), embedding_dim), where embedding_dim is either the
            model's native dimension or `n_components` if reduction is applied.
            Returns an empty dict if no models are loaded or `terms` is empty.
        """
        ret = {}

        # Materialise the input as a plain list so numpy arrays and pandas
        # Series work too (their truth value is ambiguous in an `if`). A bare
        # string is wrapped first, since list("abc") would split it into
        # single characters and embed each one separately.
        if isinstance(terms, str):
            terms = [terms]
        terms = list(terms) if terms is not None else []

        if self.models and terms:
            # Coerce all term values to strings so the model never receives
            # unexpected types (e.g. None, NaN, int) from upstream callers.
            terms = [_to_text(t) for t in terms]

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

                # self.reducers is empty when dimensionality reduction is
                # disabled, so this lookup also acts as the on/off switch.
                reducer = self.reducers.get(model_name)
                if reducer is not None:
                    # UMAP's spectral initialisation needs strictly more
                    # points than n_components + 1, so smaller inputs are
                    # returned unreduced.
                    if len(embeddings) > self.config.n_components + 1:
                        embeddings = reducer.fit_transform(embeddings)
                    else:
                        Logger().log(
                            f"Skipping dimensionality reduction for '{model_name}': "
                            f"{len(embeddings)} term(s) given, at least "
                            f"{self.config.n_components + 2} needed. Returning "
                            f"full-dimensional embeddings."
                        )

                ret[model_name] = embeddings

        return ret
