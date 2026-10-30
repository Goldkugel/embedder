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
from logger                import Logger, LoggerConfig
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

# Key under which embedder settings are expected to live in the YAML config file.
configuration_section: str = "embedder"

# Key of the logger settings in the same file (see the logger package).
logger_section: str        = "logger"

# Default path to the config file, used if no path is explicitly passed in.
standard_directory: str    = "./config/config.yaml"


def _to_text(term) -> str:
    """
    Coerce a single term to a string for the model. Missing values — None,
    NaN of any float type, and pandas' NA and NaT — become an empty string;
    every other value, including falsy ones like 0 or False and array-like
    values, keeps its string form.
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


def _clean_term(term: str, lowercase: bool) -> str:
    """
    Collapse runs of whitespace (including leading and trailing) into
    single spaces and, if `lowercase` is True, lowercase the term. The
    result is both the text the model encodes and the term's key in the
    cache, so "Renal  cyst" and "Renal cyst" are encoded and cached once.
    """
    term = " ".join(term.split())
    return term.lower() if lowercase else term


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
            If the configuration file has no "embedder" section, or a
            loaded model has no pooling module to configure.
        pydantic.ValidationError
            If the embedder settings contain unknown keys or invalid values.
        """
        # Initialise instance attributes explicitly so each Embedder instance
        # has its own model and reducer registries, rather than sharing
        # class-level mutable defaults.
        self.models    : dict = {}
        self.reducers  : dict = {}
        # Per model ID: whether terms are lowercased before encoding and
        # cache lookup (see the `lowercase` setting).
        self.lowercase : dict = {}

        self.config, logger_path, logger_settings = Embedder._resolveConfig(config)

        # The logger is a process-wide singleton. If the entry point has not
        # configured it yet, configure it from the same source as the
        # embedder: the same YAML file, or the "logger" key of the same
        # dict. Without either, Logger() falls back to ./config/config.yaml.
        if Logger.config is None and logger_path is not None:
            # Checked here because the logger package would only raise a
            # bare KeyError('logger').
            if logger_settings is None:
                raise ValueError(
                    f"Configuration file '{logger_path}' has no (or an empty) "
                    f"'{logger_section}' section, and the logger has not been "
                    f"configured yet. Add the section, or configure the logger "
                    f"first with Logger(<path>)."
                )
            l = Logger(logger_path)
        else:
            if Logger.config is None and logger_settings is not None:
                Logger.config = LoggerConfig.model_validate(logger_settings)
            l = Logger()

        # Limit PyTorch's CPU threads before any model runs, so a large job
        # does not occupy every core of a shared machine. The OpenVINO
        # backend gets the same limit in _backend_arguments().
        if self.config.num_threads is not None:
            torch.set_num_threads(self.config.num_threads)
            l.log(f"Limiting the model(s) to {self.config.num_threads} CPU thread(s).")

        l.log("Loading model(s)...")

        if not self.config.model_id:
            l.log("No model specified.")
        else:
            # Load each configured SentenceTransformer model and store it
            # in self.models, keyed by its model ID, so embed() can iterate
            # over all of them and return per-model results.
            for model_id in self.config.model_id:
                # None (no entry in the config) loads the latest revision
                # from the Hub, or whatever a local path contains.
                revision = self.config.revision.get(model_id)
                l.log(
                    f"Loading embedding model '{model_id}' (revision "
                    f"'{revision or 'not pinned'}', pooling '{self.config.pooling}', "
                    f"backend '{self.config.backend}"
                    f"{' int8' if self.config.quantize_int8 else ''}', "
                    f"device '{self.config.device}')..."
                )
                model = SentenceTransformer(
                    model_id,
                    device   = self.config.device,
                    revision = revision,
                    **self._backend_arguments(),
                )
                previous = Embedder._apply_pooling(model, self.config.pooling)
                if isinstance(previous, str) and previous != self.config.pooling:
                    # Not an error (e.g. SapBERT gets sentence-transformers'
                    # "mean" fallback, but was trained with "cls"), but a
                    # model trained with another pooling gives worse vectors.
                    l.log(
                        f"Warning: '{model_id}': the configured '{self.config.pooling}' "
                        f"pooling replaces the model's own '{previous}' pooling."
                    )

                # A longer max_seq_length than the model has position
                # embeddings for would only fail once a long term arrives.
                limit = Embedder._max_positions(model)
                if limit is not None and self.config.max_seq_length > limit:
                    raise ValueError(
                        f"max_seq_length {self.config.max_seq_length} is larger than "
                        f"the {limit} tokens model '{model_id}' supports."
                    )

                # Override the model's default max sequence length with the
                # configured value — inputs longer than this are truncated.
                model.max_seq_length = self.config.max_seq_length
                self.models[model_id] = model

                if self.config.lowercase == "auto":
                    # Uncased models (e.g. SapBERT) lowercase every input in
                    # their tokenizer, so "Seizure" and "seizure" give the
                    # same vector; lowercasing here as well avoids encoding
                    # and caching both. The `is True` check also keeps
                    # tokenizers without this attribute (or test mocks) cased.
                    tokenizer = getattr(model, "tokenizer", None)
                    self.lowercase[model_id] = getattr(tokenizer, "do_lower_case", None) is True
                else:
                    self.lowercase[model_id] = self.config.lowercase
                l.log(
                    f"Loading embedding model '{model_id}' completed (terms are "
                    f"{'lowercased' if self.lowercase[model_id] else 'kept cased'})."
                )

        l.log("Loading model(s) completed.")

        if self.config.dimensionality_reduction and self.models:
            # Initialise one UMAP reducer per model with the configured
            # parameters, so fitting on one model's embeddings never
            # overwrites the fitted state belonging to another model.
            # Each reducer is fitted inside reduce() (via fit_transform).
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

        Each loaded model produces one set of embeddings. Whitespace in each
        term is collapsed and, for models with lowercasing enabled (see the
        `lowercase` setting), the term is lowercased; terms that are equal
        after this are encoded only once per call. If `cache_folder` is
        configured, terms already in the model's cache file are loaded from
        it instead of being encoded, and newly encoded terms are appended to
        it, rounded to `cache_decimals` decimals (the returned vectors are
        rounded the same way, so they always equal what the cache holds).

        Terms are encoded in chunks of `progress_interval`. After each chunk
        a progress message is logged and, if caching is enabled, the chunk
        is appended to the cache file, so an interrupted run resumes where
        it stopped.

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

        # Materialise the input as a plain list so numpy arrays and pandas
        # Series work too (their truth value is ambiguous in an `if`). A bare
        # string is wrapped first, since list("abc") would split it into
        # single characters and embed each one separately.
        if isinstance(terms, str):
            terms = [terms]
        terms = list(terms) if terms is not None else []

        if self.models and terms:
            # Coerce all term values to strings so the model never receives
            # unexpected types (e.g. None, NaN, pd.NA, int) from upstream callers.
            terms = [_to_text(t) for t in terms]

            for model_name, model in self.models.items():
                # The text actually encoded, which is also the cache key.
                keys = [_clean_term(t, self.lowercase[model_name]) for t in terms]

                # cache maps each key to its embedding. Without caching it
                # starts empty and only lives for this call, which still
                # avoids encoding a repeated term twice.
                cache_path = self._cache_path(model_name)
                cache = {}
                if cache_path is not None:
                    cache, intact = Embedder._read_cache(cache_path)
                    if not intact:
                        # An append was interrupted (e.g. the job was
                        # killed), or a line is malformed. Rewrite the file
                        # from the entries that could be read, so new
                        # entries can be appended.
                        Logger().log(
                            f"Warning: '{model_name}': cache file '{cache_path}' could "
                            f"only be read in part; rewriting it with the {len(cache)} "
                            f"entries that could be read."
                        )
                        Embedder.save_cache(cache_path, cache)

                # Distinct keys not in the cache yet, in first-seen order.
                missing = [k for k in dict.fromkeys(keys) if k not in cache]
                if cache_path is not None:
                    Logger().log(
                        f"'{model_name}': {len(set(keys)) - len(missing)} distinct "
                        f"term(s) loaded from cache, {len(missing)} to encode."
                    )

                self._encode_into_cache(model_name, model, missing, cache, cache_path)

                # Assemble the output in the caller's order, including
                # duplicates, from the (now complete) cache.
                ret[model_name] = np.stack([cache[k] for k in keys])

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
                    f"Warning: '{model_name}': {len(vectors)} row(s) given, at least "
                    f"{self.config.n_components + 2} needed for dimensionality "
                    f"reduction; returning the full-dimensional embeddings."
                )
                ret[model_name] = vectors

        return ret

    @staticmethod
    def save_cache(path: str, cache: dict, decimals: int | None = None) -> None:
        """
        Write a complete embedding cache to a JSON file without ever leaving
        a half-written file behind.

        The data is first written to `<path>.tmp`, flushed to disk, and only
        then renamed to `path`. The rename replaces the old file in a single
        step, so if the process is killed mid-write, `path` still holds the
        previous complete version and only the .tmp file is incomplete (it
        is overwritten by the next save).

        The file is a JSON object mapping each term to its embedding (a list
        of numbers), written with one term per line so load_cache() can read
        large files line by line and append_cache() can add entries cheaply.

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
            # Make sure the data is on disk before the rename makes it visible.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @staticmethod
    def append_cache(path: str, entries: dict, decimals: int | None = None) -> None:
        """
        Add entries to a cache file written by save_cache(), without
        rewriting the entries already in it.

        The closing brace at the end of the file is replaced by the new
        entries followed by a new closing brace, so the file stays a valid
        JSON object and the cost of a checkpoint only depends on the number
        of new entries. If `path` does not exist yet, it is created with
        save_cache().

        Unlike save_cache(), an append is not done in a single step: if the
        process is killed during it, the file can end with an incomplete
        line and no closing brace. The entries written before are kept;
        load_cache() skips the incomplete line, and embed() rewrites such a
        file before appending to it again.

        Terms are not checked for duplicates; appending a term that is
        already in the file adds a second line for it, and the later one
        wins when the file is read.

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
            # The closing brace is the last non-whitespace character; the
            # end of the file is enough to find it and what precedes it.
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

            # Cut the file right after the last entry (or the opening brace
            # of an empty object) and write the new entries plus a new
            # closing brace in its place.
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

        Returns an empty dict if the file does not exist. Files in the
        one-term-per-line layout are parsed line by line, so a large cache
        never has to exist as one big Python object; any other valid JSON
        object of the same shape is read with json.load(). If an append was
        interrupted, the incomplete last line is skipped and all complete
        entries are returned.

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
        Encode `terms` with `model` in chunks of `progress_interval` and add
        each embedding to `cache`. After every chunk, log the progress and
        (if caching is enabled) append the chunk to the cache file, so the
        work done so far survives an interrupted run.

        With caching enabled, the vectors are rounded to `cache_decimals`
        before being added, so `cache` holds exactly what a later run reads
        back from the file.

        Vectors containing NaN or infinity (e.g. after a float16 overflow on
        a GPU) are still returned, but logged and not written to the cache
        file, which could not hold them.
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
            vectors = np.asarray(vectors, dtype=np.float32)
            if cache_path is not None and self.config.cache_decimals is not None:
                vectors = np.round(vectors, self.config.cache_decimals)
            entries = dict(zip(chunk, vectors))
            cache.update(entries)

            invalid = [t for t, v in entries.items() if not np.isfinite(v).all()]
            if invalid:
                shown = ", ".join(repr(t) for t in invalid[:5])
                Logger().log(
                    f"Warning: '{model_name}': NaN or infinite values for "
                    f"{len(invalid)} term(s) ({shown}{', ...' if len(invalid) > 5 else ''})"
                    f"{'; they are not cached' if cache_path is not None else ''}."
                )
                for term in invalid:
                    del entries[term]

            done = start + len(chunk)
            Logger().log(
                f"'{model_name}': encoded {done}/{total} term(s) ({100 * done / total:.1f}%)."
            )
            if cache_path is not None:
                Embedder.append_cache(cache_path, entries)

    def _cache_path(self, model_name: str) -> str | None:
        """
        Build the cache file path for one model, or return None if caching
        is disabled.

        The file name is the model ID (made safe for file systems) plus a
        SHA-256 hash over every setting that changes the output vectors:
        model ID, revision, pooling, max_seq_length, normalization and int8
        quantisation. Changing any of these starts a new file instead of
        mixing vectors computed in different ways.

        The backend itself is not part of the hash: torch and OpenVINO
        (without int8) produce the same vectors up to about 1e-6, below the
        cache's rounding, so they can share a file. Lowercasing and
        cache_decimals are not part of it either, since lowercasing only
        changes which keys are looked up and rounding is far below the
        differences that matter.
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
                # Only added when enabled, so files written before this
                # setting existed keep their names.
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

        model_kwargs = {}
        if self.config.num_threads is not None:
            # torch.set_num_threads() does not reach OpenVINO, which has its
            # own thread setting.
            model_kwargs["ov_config"] = {"INFERENCE_NUM_THREADS": str(self.config.num_threads)}
        if self.config.quantize_int8:
            # Compress the weights to 8-bit integers while converting the
            # model to OpenVINO (about twice as fast on the CPU).
            model_kwargs["load_in_8bit"] = True
        return {"backend": self.config.backend, "model_kwargs": model_kwargs}

    @staticmethod
    def _resolveConfig(config) -> tuple[EmbedderConfig, str | None, dict | None]:
        """
        Turn the `config` argument of __init__ into an EmbedderConfig.

        Returns the EmbedderConfig, the YAML file it came from (or None),
        and the "logger" section of that file or of a full configuration
        dict (or None), so __init__ can configure the logger from the same
        source.
        """
        if isinstance(config, EmbedderConfig):
            return config, None, None

        if isinstance(config, dict):
            # A full configuration (the parsed YAML file) has an "embedder"
            # key, which is not a valid embedder setting itself, so the two
            # kinds of dict cannot be confused.
            if configuration_section in config:
                section = config[configuration_section] or {}
                return EmbedderConfig.model_validate(section), None, config.get(logger_section)
            return EmbedderConfig.model_validate(config), None, None

        path = os.fspath(config)
        # An empty file parses to None, so fall back to an empty dict to get
        # the clearer "missing section" error below instead of a TypeError.
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if configuration_section not in data:
            raise ValueError(
                f"Configuration file '{path}' has no '{configuration_section}' section."
            )
        # A section that is present but empty (`embedder:` with nothing
        # below it) also parses to None; treat it as "use all defaults".
        return (
            EmbedderConfig.model_validate(data[configuration_section] or {}),
            path,
            data.get(logger_section),
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
