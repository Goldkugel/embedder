"""
Core embedding engine for the embedder package.

Provides the Embedder class, which loads one or more SentenceTransformer
models from Hugging Face (or local paths), encodes lists of text terms into
dense vector representations (optionally cached on disk as JSON), and can
project those embeddings down with UMAP dimensionality reduction.
"""

from __future__ import annotations

from collections.abc       import Iterable
from sentence_transformers import SentenceTransformer
from .EmbedderConfig       import EmbedderConfig
from logger                import Logger
import hashlib
import json
import math
import os
import re
import numpy as np
import torch
import yaml
import umap

# sentence-transformers 6 moved the Pooling module and deprecated the old
# import path; fall back to it for the older versions pyproject.toml allows.
try:
    from sentence_transformers.sentence_transformer.modules import Pooling
except ImportError:
    from sentence_transformers.models import Pooling

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
    vector representations and optionally reducing their dimensionality.

    One or more SentenceTransformer models are loaded at construction time.
    Each call to embed() runs all loaded models and returns their
    full-dimensional outputs as a dictionary keyed by model ID. If
    dimensionality reduction is enabled, reduce() projects such a dictionary
    with one UMAP reducer per model. Each reducer is fitted afresh on every
    reduce() call, so reduced coordinates are only comparable within the
    output of a single call, not across calls.
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
            If the configuration file has no "embedder" section, or a
            loaded model has no pooling module to configure.
        pydantic.ValidationError
            If the "embedder" section contains unknown keys or invalid values.
        """
        # Initialise instance attributes explicitly so each Embedder instance
        # has its own model and reducer registries, rather than sharing
        # class-level mutable defaults.
        self.models   : dict = {}
        self.reducers : dict = {}

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

        # Limit PyTorch's CPU threads before any model runs, so a large job
        # does not occupy every core of a shared machine.
        if self.config.num_threads is not None:
            torch.set_num_threads(self.config.num_threads)
            l.log(f"Limiting PyTorch to {self.config.num_threads} CPU thread(s).")

        l.log("Loading model(s)...")

        if self.config.model_id:
            # Load each configured SentenceTransformer model and store it
            # in self.models, keyed by its model ID, so embed() can iterate
            # over all of them and return per-model results.
            for model_id in self.config.model_id:
                # None (no entry in the config) loads the latest revision.
                revision = self.config.revision.get(model_id)
                l.log(
                    f"Loading embedding model '{model_id}' (revision "
                    f"'{revision or 'latest'}', pooling '{self.config.pooling}') "
                    f"on device '{self.config.device}'..."
                )
                model = SentenceTransformer(
                    model_id, device=self.config.device, revision=revision
                )
                Embedder._apply_pooling(model, self.config.pooling)

                # Override the model's default max sequence length with the
                # configured value — inputs longer than this are truncated.
                model.max_seq_length = self.config.max_seq_length
                self.models[model_id] = model
                l.log(f"Loading embedding model '{model_id}' completed.")

            if self.config.dimensionality_reduction:
                # Initialise one UMAP reducer per model with the configured
                # parameters, so fitting on one model's embeddings never
                # overwrites the fitted state belonging to another model.
                # Each reducer is fitted inside reduce() (via fit_transform).
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
        Encode a sequence of text terms into full-dimensional embeddings.

        Each loaded model produces one set of embeddings. Every distinct
        term is encoded only once per call. If `cache_folder` is configured,
        terms already in the model's cache file are loaded from it instead
        of being encoded, and newly encoded terms are added to it.

        Terms are encoded in chunks of `progress_interval`. After each chunk
        a progress message is logged and, if caching is enabled, the cache
        file is saved, so an interrupted run resumes where it stopped.

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
            (len(terms), embedding_dim), where embedding_dim is the model's
            native dimension. Returns an empty dict if no models are loaded
            or `terms` is empty.
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
                # cache maps each term to its embedding. Without caching it
                # starts empty and only lives for this call, which still
                # avoids encoding a repeated term twice.
                cache_path = self._cache_path(model_name)
                cache = Embedder.load_cache(cache_path) if cache_path is not None else {}

                # Distinct terms not in the cache yet, in first-seen order.
                missing = [t for t in dict.fromkeys(terms) if t not in cache]
                if cache_path is not None:
                    Logger().log(
                        f"'{model_name}': {len(set(terms)) - len(missing)} distinct "
                        f"term(s) loaded from cache, {len(missing)} to encode."
                    )

                self._encode_into_cache(model_name, model, missing, cache, cache_path)

                # Assemble the output in the caller's order, including
                # duplicates, from the (now complete) cache.
                ret[model_name] = np.stack([cache[t] for t in terms])

        return ret

    def reduce(self, embeddings: dict) -> dict:
        """
        Project embeddings returned by embed() down to `n_components`
        dimensions with UMAP, using each model's own reducer.

        Each reducer is fitted afresh on every call, so the reduced
        coordinates are only comparable within one call. UMAP's spectral
        initialisation needs at least `n_components + 2` points; a model's
        embeddings with fewer rows are returned unchanged and a message is
        logged.

        Parameters
        ----------
        embeddings : dict
            Dictionary mapping model IDs to numpy arrays, as returned by
            embed().

        Returns
        -------
        dict
            Dictionary with the same keys, mapping each model ID to a numpy
            array of shape (rows, n_components).

        Raises
        ------
        RuntimeError
            If dimensionality_reduction is disabled in the configuration.
        KeyError
            If `embeddings` contains a model ID this Embedder did not load.
        """
        if not self.config.dimensionality_reduction:
            raise RuntimeError(
                "Dimensionality reduction is disabled; set "
                "'dimensionality_reduction: True' in the configuration to use reduce()."
            )

        ret = {}
        for model_name, vectors in embeddings.items():
            reducer = self.reducers[model_name]

            # UMAP's spectral initialisation needs strictly more points than
            # n_components + 1, so smaller inputs are returned unreduced.
            if len(vectors) > self.config.n_components + 1:
                ret[model_name] = reducer.fit_transform(vectors)
            else:
                Logger().log(
                    f"Skipping dimensionality reduction for '{model_name}': "
                    f"{len(vectors)} row(s) given, at least "
                    f"{self.config.n_components + 2} needed. Returning "
                    f"full-dimensional embeddings."
                )
                ret[model_name] = vectors

        return ret

    @staticmethod
    def save_cache(path: str, cache: dict) -> None:
        """
        Write an embedding cache to a JSON file without ever leaving a
        half-written file behind.

        The data is first written to `<path>.tmp`, flushed to disk, and only
        then renamed to `path`. The rename replaces the old file in a single
        step, so if the process is killed mid-write, `path` still holds the
        previous complete version and only the .tmp file is incomplete (it
        is overwritten by the next save).

        The file is a JSON object mapping each term to its embedding (a list
        of numbers), written with one term per line so load_cache() can read
        large files line by line.

        Parameters
        ----------
        path : str
            Destination JSON file. Its folder is created if missing.
        cache : dict
            Mapping from term (str) to embedding (1-D numpy array).
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("{\n")
            for i, (term, vector) in enumerate(cache.items()):
                separator = ",\n" if i < len(cache) - 1 else "\n"
                # astype(str) on float32 gives the shortest text that reads
                # back to exactly the same float32 value, which keeps the
                # file much smaller than Python's float64 repr would.
                numbers = ",".join(np.asarray(vector, dtype=np.float32).astype(str))
                f.write(f"{json.dumps(term, ensure_ascii=False)}: [{numbers}]{separator}")
            f.write("}\n")
            # Make sure the data is on disk before the rename makes it visible.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @staticmethod
    def load_cache(path: str) -> dict:
        """
        Read an embedding cache written by save_cache().

        Returns an empty dict if the file does not exist. Files in
        save_cache()'s one-term-per-line layout are parsed line by line, so
        a large cache never has to exist as one big Python object; any other
        valid JSON object of the same shape is read with json.load().

        Parameters
        ----------
        path : str
            JSON file to read.

        Returns
        -------
        dict
            Mapping from term (str) to embedding (1-D float32 numpy array).
        """
        if not os.path.exists(path):
            return {}

        cache = {}
        with open(path, "r", encoding="utf-8") as f:
            if f.readline().strip() == "{":
                for line in f:
                    line = line.strip().rstrip(",")
                    if line in ("", "}"):
                        continue
                    (term, vector), = json.loads("{" + line + "}").items()
                    cache[term] = np.asarray(vector, dtype=np.float32)
                return cache

        # Not in save_cache()'s layout (e.g. edited by hand): read it whole.
        with open(path, "r", encoding="utf-8") as f:
            return {t: np.asarray(v, dtype=np.float32) for t, v in json.load(f).items()}

    def _encode_into_cache(
        self, model_name: str, model, terms: list[str], cache: dict, cache_path: str | None
    ) -> None:
        """
        Encode `terms` with `model` in chunks of `progress_interval` and add
        each embedding to `cache`. After every chunk, log the progress and
        (if caching is enabled) save the cache, so the work done so far
        survives an interrupted run.
        """
        total = len(terms)
        interval = self.config.progress_interval

        for start in range(0, total, interval):
            chunk = terms[start:start + interval]

            # batch_size controls how many terms are processed per forward
            # pass — larger values are faster but use more memory.
            vectors = model.encode(
                chunk,
                batch_size           = self.config.batch_size,
                show_progress_bar    = False,
                normalize_embeddings = self.config.normalization,
                convert_to_numpy     = True,
            )
            cache.update(zip(chunk, vectors))

            done = start + len(chunk)
            Logger().log(
                f"'{model_name}': encoded {done}/{total} term(s) ({100 * done / total:.1f}%)."
            )
            if cache_path is not None:
                Embedder.save_cache(cache_path, cache)

    def _cache_path(self, model_name: str) -> str | None:
        """
        Build the cache file path for one model, or return None if caching
        is disabled.

        The file name is the model ID (made safe for file systems) plus a
        SHA-256 hash over every setting that changes the output vectors:
        model ID, revision, pooling, max_seq_length and normalization.
        Changing any of these starts a new file instead of mixing vectors
        computed in different ways.
        """
        if self.config.cache_folder is None:
            return None

        key = json.dumps(
            {
                "model_id"      : model_name,
                "revision"      : self.config.revision.get(model_name),
                "pooling"       : self.config.pooling,
                "max_seq_length": self.config.max_seq_length,
                "normalization" : self.config.normalization,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        return os.path.join(self.config.cache_folder, f"{safe_name}-{digest}.json")

    @staticmethod
    def _apply_pooling(model, pooling: str) -> None:
        """
        Replace the pooling module of a loaded SentenceTransformer with one
        using `pooling`, keeping every other module (e.g. a Normalize step)
        unchanged.

        Raises
        ------
        ValueError
            If the model has no pooling module.
        """
        for index, module in enumerate(model):
            if isinstance(module, Pooling):
                # sentence-transformers 6 renamed this attribute.
                dimension = getattr(module, "embedding_dimension", None) or module.word_embedding_dimension
                model[index] = Pooling(dimension, pooling_mode=pooling)
                return
        raise ValueError("The model has no pooling module to configure.")
