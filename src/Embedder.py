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

# Sections of the YAML config file, and its default path.
configuration_section: str = "embedder"
logger_section: str        = "logger"
standard_directory: str    = "./config/config.yaml"


def _to_text(term) -> str:
    """
    Coerce a term to a string. Missing values (None, NaN, pd.NA, NaT) become
    "", every other value (including 0, False and arrays) its str() form.
    """
    if term is None:
        return ""
    try:
        # NaN and NaT are the only common values not equal to themselves.
        if term != term:
            return ""
    except TypeError:
        # pd.NA: comparing it gives pd.NA again, whose truth value raises.
        return ""
    except ValueError:
        # A numpy array compares element-wise, and the truth value of the
        # result raises; it is not a missing value, so keep its text.
        pass
    return str(term)


def _section(data: dict, key: str) -> dict | None:
    """
    Return the section `key` of a parsed configuration: None if it is
    missing, and an empty dict if it is present but empty (`logger:` with
    nothing below it parses to None).
    """
    if key not in data:
        return None
    return data[key] or {}


def _clean_term(term: str, lowercase: bool) -> str:
    """
    Collapse whitespace and optionally lowercase the term. The result is both
    the encoded text and the cache key, so "Renal  cyst" and "Renal cyst" are
    encoded and cached once.
    """
    term = " ".join(term.split())
    return term.lower() if lowercase else term


class Embedder:
    """
    Encodes text terms into dense vectors and optionally reduces their
    dimensionality.

    The configured SentenceTransformer models are loaded at construction.
    embed() returns the full-dimensional vectors of every model, keyed by
    model ID; reduce() projects them with one UMAP reducer per model.
    """

    def __init__(
        self, config: str | os.PathLike | dict | EmbedderConfig = standard_directory
    ):
        """
        Load and validate configuration, then initialise all configured
        SentenceTransformer models and (if enabled) one UMAP reducer per model.

        Parameters
        ----------
        config : str, os.PathLike, dict or EmbedderConfig, optional
            Where the settings come from:
            - a path to a YAML configuration file with an "embedder"
              section (default: ./config/config.yaml relative to the
              working directory);
            - a dict, either the content of such a file (with "embedder"
              and optionally "logger" keys) or just the embedder settings
              themselves;
            - a ready EmbedderConfig, which is used as is.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If the configuration file has no "embedder" section; if the
            logger still has to be configured from that file and it has no
            "logger" section; if a loaded model has no pooling module to
            configure; or if a model's max_seq_length exceeds the number of
            positions it supports.
        pydantic.ValidationError
            If the embedder settings contain unknown keys or invalid values.
        """
        self.models    : dict = {}
        self.reducers  : dict = {}
        # Per model ID: whether terms are lowercased (see `lowercase`).
        self.lowercase : dict = {}

        self.config, logger_path, logger_settings = Embedder._resolveConfig(config)

        # The logger is a process-wide singleton. If it is not configured yet,
        # configure it from the same file or dict as the embedder; otherwise
        # Logger() falls back to ./config/config.yaml.
        if Logger.config is None and logger_path is not None:
            # The logger package would only raise a bare KeyError('logger').
            if logger_settings is None:
                raise ValueError(
                    f"Configuration file '{logger_path}' has no "
                    f"'{logger_section}' section, and the logger has not been "
                    f"configured yet. Add the section, or configure the logger "
                    f"first with Logger(<path>)."
                )
            l = Logger(logger_path)
        elif Logger.config is None and logger_settings is not None:
            l = Logger(logger_settings)
        else:
            l = Logger()

        # Set before any model runs; OpenVINO gets its limit in
        # _backend_arguments().
        if self.config.num_threads is not None:
            torch.set_num_threads(self.config.num_threads)
            l.log(f"Limiting the model(s) to {self.config.num_threads} CPU thread(s).")

        l.log("Loading model(s)...")

        if not self.config.model_id:
            l.log("No model specified.")
        else:
            for model_id in self.config.model_id:
                # None loads the latest revision, or whatever a local path holds.
                revision       = self.config.revision.get(model_id)
                pooling        = self.config.model_setting("pooling", model_id)
                max_seq_length = self.config.model_setting("max_seq_length", model_id)
                lowercase      = self.config.model_setting("lowercase", model_id)
                # One row per setting, so a long model ID and revision hash
                # stay readable.
                l.log(f"Loading embedding model '{model_id}'...")
                l.log(f"Revision: '{revision or 'not pinned'}'.")
                l.log(f"Pooling: '{pooling}'.")
                l.log(f"Backend: '{self.config.backend}"
                      f"{' int8' if self.config.quantize_int8 else ''}'.")
                l.log(f"Device: '{self.config.device}'.")
                model = SentenceTransformer(
                    model_id,
                    device   = self.config.device,
                    revision = revision,
                    **self._backend_arguments(),
                )
                previous = Embedder._apply_pooling(model, pooling)
                if isinstance(previous, str) and previous != pooling:
                    # Not an error (SapBERT gets sentence-transformers' "mean"
                    # fallback but was trained with "cls"), but worse vectors.
                    l.log(
                        f"'{Embedder.getModelName(model_id)}': pooling '{pooling}' replaces "
                        f"the model's own '{previous}'.",
                        level = "warning",
                    )

                # A longer max_seq_length than the model has position
                # embeddings for would only fail once a long term arrives.
                limit = Embedder._max_positions(model)
                if limit is not None and max_seq_length > limit:
                    raise ValueError(
                        f"max_seq_length {max_seq_length} is larger than "
                        f"the {limit} tokens model '{model_id}' supports."
                    )

                model.max_seq_length = max_seq_length
                self.models[model_id] = model

                if lowercase == "auto":
                    # Uncased tokenizers (e.g. SapBERT) lowercase anyway, so this
                    # only avoids encoding and caching "Seizure" and "seizure"
                    # twice. `is True` keeps tokenizers without the attribute
                    # (or test mocks) cased.
                    tokenizer = getattr(model, "tokenizer", None)
                    self.lowercase[model_id] = getattr(tokenizer, "do_lower_case", None) is True
                else:
                    self.lowercase[model_id] = lowercase
                l.log(f"Terms are {'lowercased' if self.lowercase[model_id] else 'kept cased'}.")
                l.log(f"Loading embedding model '{model_id}' completed.")

        l.log("Loading model(s) completed.")

        if self.config.dimensionality_reduction and self.models:
            # One reducer per model, so fitting one never overwrites another's
            # state. They are fitted in reduce().
            l.log("Creating UMAP reducer(s)...")
            for model_id in self.models:
                self.reducers[model_id] = umap.UMAP(
                    n_components = self.config.n_components,
                    n_neighbors  = self.config.n_neighbors,
                    min_dist     = self.config.min_dist,
                    metric       = self.config.metric,
                    random_state = self.config.random_state,
                )
            l.log("Creating UMAP reducer(s) completed.")

    def embed(self, terms: str | Iterable) -> dict:
        """
        Encode a sequence of text terms into full-dimensional embeddings.

        Whitespace in each term is collapsed and, depending on `lowercase`,
        the term is lowercased; terms equal after this are encoded once. With
        `cache_folder` set, cached terms are loaded instead of encoded, and
        new ones are appended to the cache every `progress_interval` terms,
        rounded to `cache_decimals` (the returned vectors are rounded alike).

        Parameters
        ----------
        terms : str or Iterable
            Text strings to embed — a list, tuple, numpy array or pandas
            Series. A single string is treated as one term. Missing values
            (None, NaN, pd.NA, NaT) become an empty string; any other value
            is converted with str().

        Returns
        -------
        dict
            Dictionary mapping each model ID to a numpy array of shape
            (len(terms), embedding_dim), where embedding_dim is the model's
            native dimension. Returns an empty dict if no models are loaded
            or `terms` is empty.
        """
        ret = {}

        # A plain list, since the truth value of an array or Series is
        # ambiguous; a bare string is wrapped, as list("abc") would split it.
        if isinstance(terms, str):
            terms = [terms]
        terms = list(terms) if terms is not None else []

        if self.models and terms:
            terms = [_to_text(t) for t in terms]

            for model_name, model in self.models.items():
                # The text actually encoded, which is also the cache key.
                keys = [_clean_term(t, self.lowercase[model_name]) for t in terms]

                # Without caching, `cache` only lives for this call, which
                # still avoids encoding a repeated term twice.
                cache_path = self._cache_path(model_name)
                cache = {}
                if cache_path is not None:
                    cache, intact = Embedder._read_cache(cache_path)
                    if not intact:
                        # An interrupted append or a malformed line: rewrite
                        # the file so new entries can be appended.
                        Logger().log(
                            f"'{Embedder.getModelName(model_name)}': cache file '{cache_path}' could only "
                            f"be read in part.",
                            level = "warning",
                        )
                        Logger().log(
                            f"'{Embedder.getModelName(model_name)}': rewriting the cache file with the {len(cache)} "
                            f"entries that could be read.",
                            level = "warning",
                        )
                        Embedder.save_cache(cache_path, cache)

                # Distinct keys not in the cache yet, in first-seen order.
                missing = [k for k in dict.fromkeys(keys) if k not in cache]
                if cache_path is not None:
                    Logger().log(
                        f"'{Embedder.getModelName(model_name)}': {len(set(keys)) - len(missing)} distinct "
                        f"term(s) loaded from cache, {len(missing)} to encode."
                    )

                self._encode_into_cache(model_name, model, missing, cache, cache_path)

                # In the caller's order, duplicates included.
                ret[model_name] = np.stack([cache[k] for k in keys])

        return ret

    def reduce(self, embeddings: dict) -> dict:
        """
        Project embeddings returned by embed() down to `n_components`
        dimensions with UMAP, using each model's own reducer.

        Each reducer is fitted afresh on every call, so the reduced
        coordinates are only comparable within one call. UMAP's spectral
        initialisation needs at least `n_components + 2` rows; smaller inputs
        are returned unchanged, with a warning.

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
            if len(vectors) > self.config.n_components + 1:
                ret[model_name] = reducer.fit_transform(vectors)
            else:
                Logger().log(
                    f"'{Embedder.getModelName(model_name)}': {len(vectors)} row(s) given, at least "
                    f"{self.config.n_components + 2} needed for dimensionality reduction.",
                    level = "warning",
                )
                Logger().log(
                    f"'{Embedder.getModelName(model_name)}': returning the full-dimensional embeddings.",
                    level = "warning",
                )
                ret[model_name] = vectors

        return ret

    @staticmethod
    def getModelName(model_id: str) -> str:
        """
        The name of a model in the logs: its ID without the organisation,
        e.g. "SapBERT-from-PubMedBERT-fulltext" for
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" (the last part of a
        local path).
        """
        return str(model_id).rstrip("/").rsplit("/", 1)[-1]

    @staticmethod
    def save_cache(path: str, cache: dict, decimals: int | None = None) -> None:
        """
        Write a complete embedding cache to a JSON file.

        The file is written to `<path>.tmp` and then renamed, so a killed
        process never leaves `path` half-written. It holds one term per line,
        so load_cache() can read it line by line and append_cache() can
        extend it cheaply.

        Parameters
        ----------
        path : str
            Destination JSON file. Its folder is created if missing.
        cache : dict
            Mapping from term (str) to embedding (1-D numpy array).
        decimals : int or None, optional
            Round every value to this many decimals before writing (smaller
            file). None writes the exact float32 values.
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("{\n")
            for i, (term, vector) in enumerate(cache.items()):
                separator = ",\n" if i < len(cache) - 1 else "\n"
                f.write(Embedder._cache_entry(term, vector, decimals) + separator)
            f.write("}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @staticmethod
    def append_cache(path: str, entries: dict, decimals: int | None = None) -> None:
        """
        Add entries to a cache file written by save_cache(), without
        rewriting the existing ones (the closing brace is replaced by the new
        entries and a new brace). A missing file is created with save_cache().

        An append is not atomic: a killed process can leave an incomplete last
        line, which load_cache() skips and embed() repairs. Duplicates are not
        checked; the later line wins when the file is read.

        Parameters
        ----------
        path : str
            Cache file to extend.
        entries : dict
            Mapping from term (str) to embedding (1-D numpy array).
        decimals : int or None, optional
            Round every value to this many decimals before writing. None
            writes the exact float32 values.

        Raises
        ------
        ValueError
            If the file does not end with a closing brace (e.g. after an
            interrupted append), since appending would then corrupt it.
        """
        if not entries:
            return
        if not os.path.exists(path):
            Embedder.save_cache(path, entries, decimals)
            return

        # Build the text first, so a value that cannot be written raises
        # before the file is touched.
        text = ",\n".join(
            Embedder._cache_entry(term, vector, decimals) for term, vector in entries.items()
        )

        with open(path, "r+b") as f:
            # Only the end of the file is needed to find the closing brace.
            size = f.seek(0, os.SEEK_END)
            start = max(0, size - 4096)
            f.seek(start)
            tail = f.read().rstrip()
            if not tail.endswith(b"}"):
                raise ValueError(
                    f"Cache file '{path}' does not end with '}}'; it was probably "
                    f"cut off. Rewrite it with save_cache(path, load_cache(path))."
                )
            before = tail[:-1].rstrip()

            # Cut after the last entry (or the opening brace) and append.
            f.seek(start + len(before))
            f.truncate()
            separator = "\n" if before.endswith(b"{") else ",\n"
            f.write(f"{separator}{text}\n}}\n".encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def load_cache(path: str) -> dict:
        """
        Read an embedding cache written by save_cache() / append_cache().

        Returns an empty dict if the file does not exist. The one-term-per-line
        layout is parsed line by line (an incomplete last line from an
        interrupted append is skipped); other JSON objects are read whole.

        Parameters
        ----------
        path : str
            JSON file to read.

        Returns
        -------
        dict
            Mapping from term (str) to embedding (1-D float32 numpy array).

        Raises
        ------
        ValueError
            If the file is neither valid JSON nor in the one-term-per-line
            layout.
        """
        return Embedder._read_cache(path)[0]

    @staticmethod
    def _read_cache(path: str) -> tuple[dict, bool]:
        """
        Read a cache file like load_cache(), and also report whether it was
        complete (True), or cut off by an interrupted append (False).
        """
        if not os.path.exists(path):
            return {}, True

        with open(path, "r", encoding="utf-8") as f:
            line_layout = f.readline().strip() == "{"

        if line_layout:
            cache, closed, broken_lines = {}, False, 0
            with open(path, "r", encoding="utf-8") as f:
                f.readline()
                # Each line is one `"term": [...]` pair; in braces it is a
                # JSON object of its own.
                for line in f:
                    line = line.strip().rstrip(",")
                    if line == "":
                        continue
                    if line == "}":
                        closed = True
                        continue
                    try:
                        (term, vector), = json.loads("{" + line + "}").items()
                        cache[term] = Embedder._cache_vector(vector)
                    except ValueError:
                        broken_lines += 1
            if closed and broken_lines == 0:
                return cache, True

        # Not in the line layout (e.g. edited by hand, or pretty-printed):
        # read it whole.
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            return {t: Embedder._cache_vector(v) for t, v in data.items()}, True
        except ValueError as e:
            if not line_layout:
                raise ValueError(f"Cache file '{path}' could not be read ({e}).") from e

        # The line layout without a closing brace (or with a broken last
        # line): an interrupted append. Keep every complete entry.
        return cache, False

    @staticmethod
    def _cache_vector(value) -> np.ndarray:
        """
        Convert one value read from a cache file to a float32 vector.

        Raises
        ------
        ValueError
            If the value is not a non-empty list of finite numbers (e.g. a
            single number in a file edited by hand), which would otherwise
            only fail later, when embed() stacks the vectors.
        """
        try:
            vector = np.asarray(value, dtype=np.float32)
        except TypeError:
            # e.g. a nested object, which numpy rejects with a TypeError.
            raise ValueError("an embedding is not a list of finite numbers") from None
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("an embedding is not a list of finite numbers")
        return vector

    @staticmethod
    def _cache_entry(term: str, vector, decimals: int | None) -> str:
        """
        Format one cache entry as `"term": [v1,v2,...]` (without a trailing
        comma or newline).

        Raises
        ------
        ValueError
            If the vector contains NaN or infinity, which JSON cannot hold
            (the entry would make its line, and the file, unreadable).
        """
        vector = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ValueError(f"Embedding of {term!r} contains NaN or infinity.")
        if decimals is not None:
            vector = np.round(vector, decimals)
        # astype(str) on float32 gives the shortest text that reads back to
        # exactly the same float32 value, which keeps the file much smaller
        # than Python's float64 repr would.
        return f"{json.dumps(term, ensure_ascii=False)}: [{','.join(vector.astype(str))}]"

    def _encode_into_cache(
        self, model_name: str, model, terms: list[str], cache: dict, cache_path: str | None
    ) -> None:
        """
        Encode `terms` in chunks of `progress_interval` into `cache`, logging
        progress and, with caching, appending each chunk to the cache file
        (rounded to `cache_decimals`, so `cache` equals what is read back).
        Vectors with NaN or infinity (e.g. a float16 overflow on a GPU) are
        returned but logged and not written, as JSON cannot hold them.
        """
        total = len(terms)
        interval = self.config.progress_interval

        for start in range(0, total, interval):
            chunk = terms[start:start + interval]

            vectors = model.encode(
                chunk,
                batch_size           = self.config.batch_size,
                show_progress_bar    = False,
                normalize_embeddings = self.config.normalization,
                convert_to_numpy     = True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if cache_path is not None and self.config.cache_decimals is not None:
                vectors = np.round(vectors, self.config.cache_decimals)
            entries = dict(zip(chunk, vectors))
            cache.update(entries)

            invalid = [t for t, v in entries.items() if not np.isfinite(v).all()]
            if invalid:
                shown = ", ".join(repr(t) for t in invalid[:5])
                Logger().log(
                    f"'{Embedder.getModelName(model_name)}': NaN or infinite values for "
                    f"{len(invalid)} term(s) ({shown}{', ...' if len(invalid) > 5 else ''}).",
                    level = "warning",
                )
                if cache_path is not None:
                    Logger().log(
                        f"'{Embedder.getModelName(model_name)}': these term(s) are not cached.",
                        level = "warning",
                    )
                for term in invalid:
                    del entries[term]

            done = start + len(chunk)
            Logger().log(
                f"'{Embedder.getModelName(model_name)}': encoded {done}/{total} term(s) ({100 * done / total:.1f}%)."
            )
            if cache_path is not None:
                Embedder.append_cache(cache_path, entries)

    def _cache_path(self, model_name: str) -> str | None:
        """
        Build the cache file path for one model, or return None if caching
        is disabled.

        The file name is the model ID plus a hash over the settings that
        change the vectors (model ID, and the model's revision, pooling and
        max_seq_length, normalization, int8). The backend is left out, as torch and OpenVINO
        without int8 agree to about 1e-6; so are lowercasing (it only changes
        the keys) and cache_decimals.
        """
        if self.config.cache_folder is None:
            return None

        key = json.dumps(
            {
                "model_id"      : model_name,
                "revision"      : self.config.revision.get(model_name),
                "pooling"       : self.config.model_setting("pooling", model_name),
                "max_seq_length": self.config.model_setting("max_seq_length", model_name),
                "normalization" : self.config.normalization,
                # Only when enabled, so older files keep their names.
                **({"int8": True} if self.config.quantize_int8 else {}),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        return os.path.join(self.config.cache_folder, f"{safe_name}-{digest}.json")

    def _backend_arguments(self) -> dict:
        """
        Build the extra SentenceTransformer() arguments for the configured
        backend. The default torch backend needs none, which also keeps
        sentence-transformers versions without a `backend` argument working.
        """
        if self.config.backend == "torch":
            return {}

        # On CPUs with bf16 support (e.g. AVX-512 BF16 or AMX) OpenVINO
        # computes in bf16 by default, which moves the vectors by about 1e-3.
        # Force f32 so they match torch and can share its cache file.
        ov_config = {"INFERENCE_PRECISION_HINT": "f32"}
        if self.config.num_threads is not None:
            # torch.set_num_threads() does not reach OpenVINO, which has its
            # own thread setting.
            ov_config["INFERENCE_NUM_THREADS"] = str(self.config.num_threads)
        model_kwargs = {"ov_config": ov_config}
        if self.config.quantize_int8:
            model_kwargs["load_in_8bit"] = True
        return {"backend": self.config.backend, "model_kwargs": model_kwargs}

    @staticmethod
    def _resolveConfig(config) -> tuple[EmbedderConfig, str | None, dict | None]:
        """
        Turn the `config` argument of __init__ into an EmbedderConfig.

        Returns the EmbedderConfig, the YAML file it came from (or None),
        and the "logger" section of that file or of a full configuration
        dict (an empty dict for an empty section, None if there is none),
        so __init__ can configure the logger from the same source.
        """
        if isinstance(config, EmbedderConfig):
            return config, None, None

        if isinstance(config, dict):
            # A full configuration has an "embedder" key, which is not a
            # valid setting itself, so the two kinds cannot be confused.
            if configuration_section in config:
                section = config[configuration_section] or {}
                return EmbedderConfig.model_validate(section), None, _section(config, logger_section)
            return EmbedderConfig.model_validate(config), None, None

        path = os.fspath(config)
        # An empty file parses to None; {} gives the clearer error below.
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if configuration_section not in data:
            raise ValueError(
                f"Configuration file '{path}' has no '{configuration_section}' section."
            )
        # An empty section (`embedder:`) parses to None: use the defaults.
        return (
            EmbedderConfig.model_validate(data[configuration_section] or {}),
            path,
            _section(data, logger_section),
        )

    @staticmethod
    def _apply_pooling(model, pooling: str) -> str | None:
        """
        Replace the pooling module of a loaded SentenceTransformer with one
        using `pooling`, keeping every other module (e.g. a Normalize step)
        unchanged.

        Returns
        -------
        str or None
            The pooling mode the model came with, or None if it cannot be
            read.

        Raises
        ------
        ValueError
            If the model has no pooling module.
        """
        for index, module in enumerate(model):
            if isinstance(module, Pooling):
                # sentence-transformers 6 stores the mode as one string;
                # older versions as flags, readable with get_pooling_mode_str().
                previous = module.get_config_dict().get("pooling_mode")
                if previous is None and hasattr(module, "get_pooling_mode_str"):
                    previous = module.get_pooling_mode_str()
                # sentence-transformers 6 renamed this attribute.
                dimension = getattr(module, "embedding_dimension", None) or module.word_embedding_dimension
                model[index] = Pooling(dimension, pooling_mode=pooling)
                return previous
        raise ValueError("The model has no pooling module to configure.")

    @staticmethod
    def _max_positions(model) -> int | None:
        """
        Return the number of position embeddings of a loaded model's
        transformer, the longest input in tokens it can take, or None if
        the model does not say (e.g. a model without a Transformer module).
        """
        transformer = model[0]
        config = getattr(getattr(transformer, "auto_model", None), "config", None)
        value = getattr(config, "max_position_embeddings", None)
        return value if isinstance(value, int) else None
