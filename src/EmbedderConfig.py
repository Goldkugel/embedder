"""
Pydantic configuration model for the Embedder class.

Defines and validates all settings read from the "embedder" section of
the YAML configuration file. Embedder.__init__ builds it from that file; it
can also be constructed directly, e.g. to validate settings in code.
"""

from typing   import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EmbedderConfig(BaseModel):
    """
    Pydantic model defining and validating the configuration schema for
    the Embedder class.

    Instances are typically created via `EmbedderConfig.model_validate(data)`,
    where `data` is a dict loaded from the "embedder" section of a YAML config
    file. Pydantic validates field types and value ranges, and enforces
    defaults for any fields not present in the supplied data.
    """

    # Reject any keys in the input data that aren't explicitly defined below.
    # This catches typos or outdated config keys early, instead of silently
    # ignoring them.
    model_config = ConfigDict(extra="forbid")

    # List of Hugging Face model IDs or local file paths for term embedding.
    # Each model is loaded separately and produces its own set of embeddings.
    # Blank entries (e.g. a `- ""` placeholder) are dropped during validation.
    model_id: list[str]            = []

    # Hugging Face revision (commit hash, branch or tag) to load, per model
    # ID. Pinning a commit hash makes runs reproducible even if the model is
    # updated on the Hub later. Models not listed load the latest revision;
    # local paths need no entry. Every key must also appear in model_id.
    revision: dict[str, str]       = {}

    # How the token vectors of a term are combined into one embedding,
    # applied to every configured model and replacing whatever pooling the
    # model would otherwise use:
    # "mean"  — average of all token vectors (sentence-transformers' own
    #           fallback for plain Hugging Face models)
    # "cls"   — the vector of the [CLS] token (e.g. what SapBERT was
    #           trained with)
    # "max", "mean_sqrt_len_tokens", "weightedmean", "lasttoken" — further
    #           modes supported by sentence-transformers.
    # Use the pooling a model was trained with; a mismatch still produces
    # vectors, but of lower quality.
    pooling: Literal[
        "mean", "cls", "max", "mean_sqrt_len_tokens", "weightedmean", "lasttoken"
    ]                              = "mean"

    # Compute target for model execution. Common values:
    # "cpu"  — standard CPU inference (slowest, always available)
    # "cuda" — NVIDIA GPU via CUDA (fastest for large batches)
    # "mps"  — Apple Silicon GPU via Metal Performance Shaders
    device: str                    = "cpu"

    # Inference engine used to run the models:
    # "torch"    — PyTorch (default).
    # "openvino" — Intel OpenVINO, which converts each model when it is
    #              loaded (a few seconds). Needs the "openvino" extra
    #              (pip install ".[openvino]"). Without quantize_int8 it gives
    #              the same vectors as torch and is not noticeably faster;
    #              its benefit is quantize_int8.
    # ONNX is not offered: its sentence-transformers extra currently cannot
    # be installed together with sentence-transformers 6.
    backend: Literal["torch", "openvino"] = "torch"

    # Store the model weights as 8-bit integers (OpenVINO only). Measured
    # with SapBERT on the CPU: about twice as fast as torch, with a cosine
    # similarity of at least 0.998 to the full-precision vectors. Changes
    # the vectors slightly, so it gets its own cache file.
    quantize_int8: bool            = False

    # Whether terms are lowercased before encoding and cache lookup:
    # "auto" — lowercase for models whose tokenizer lowercases anyway
    #          (uncased models such as SapBERT), where it changes nothing
    #          but avoids encoding and caching "Seizure" and "seizure"
    #          separately; keep the case for all other models.
    # True / False — always / never lowercase, for every model.
    # Runs of whitespace are always collapsed to single spaces.
    lowercase: bool | Literal["auto"] = "auto"

    # Folder for the embedding cache: one JSON file per model and settings,
    # mapping each term to its embedding. embed() only encodes terms that
    # are not in the file yet, which saves a lot of time on the CPU. None
    # disables caching. Pin the model's revision when caching, otherwise an
    # updated model would still be served old vectors.
    cache_folder: str | None       = None

    # Number of decimals the cached values are rounded to (with caching
    # enabled, embed() returns the rounded values too, so they always match
    # the cache). 6 decimals change a cosine similarity by less than 1e-6
    # and make the file about a quarter smaller than exact values; None
    # stores the exact float32 values.
    cache_decimals: int | None     = Field(default=6, ge=1)

    # Number of terms embed() encodes between two progress messages. After
    # each message the newly encoded terms are appended to the cache file
    # (if cache_folder is set), so an interrupted run resumes from there.
    progress_interval: int         = Field(default=10000, gt=0)

    # Maximum number of CPU threads the model may use. None keeps the
    # default (all cores); set a limit on shared machines. For PyTorch this
    # is a process-wide setting, so it also affects other PyTorch code
    # running in the same process; with the OpenVINO backend it is passed
    # to OpenVINO as well.
    num_threads: int | None        = Field(default=None, gt=0)

    # Number of terms processed in a single inference batch. Larger batches
    # are usually faster but need more memory (RAM on the CPU, VRAM on a
    # GPU); reduce it if you run out of memory.
    batch_size: int                = Field(default=128, gt=0)

    # Maximum token sequence length before input text is truncated. Inputs
    # longer than this are silently cut off — increase if your terms are long.
    max_seq_length: int            = Field(default=128, gt=0)

    # Whether to L2-normalise output vectors to unit length. Recommended when
    # computing cosine similarity downstream: for unit vectors, the plain dot
    # product already equals the cosine similarity.
    normalization: bool            = True

    # Whether to create one UMAP reducer per model, so Embedder.reduce() can
    # project embeddings down to n_components dimensions (e.g. for plots).
    # embed() always returns the full-dimensional vectors either way, since
    # those are the ones to use for similarity and matching. Reduction is
    # skipped for inputs with fewer than n_components + 2 rows (see reduce()).
    dimensionality_reduction: bool = False

    # Target number of output dimensions after reduction. Use 2 or 3 for
    # visualisation, or a higher value (e.g. 32–64) for downstream ML tasks
    # that benefit from reduced but still informative representations.
    n_components: int              = Field(default=2, ge=1)

    # Number of neighbouring points considered when constructing the UMAP
    # graph. Smaller values emphasise fine local structure; larger values
    # capture broader global structure at the cost of local detail. If an
    # input has fewer rows than this, UMAP lowers it to (rows - 1).
    n_neighbors: int               = Field(default=5, ge=2)

    # Minimum distance between points in the reduced space. Smaller values
    # produce tighter, more clustered layouts; larger values spread points
    # out more uniformly.
    min_dist: float                = Field(default=0.2, ge=0.0)

    # Distance metric used to compute proximity between embedding vectors
    # before reduction. "cosine" is recommended when embeddings are
    # L2-normalised; "euclidean" suits raw unnormalised vectors.
    metric: str                    = "cosine"

    # Random seed for UMAP, ensuring that repeated runs with the same input
    # produce the same reduced layout. Set to None to disable determinism
    # (this also lets UMAP run multi-threaded, which is faster).
    random_state: int | None       = 42

    @field_validator("model_id")
    @classmethod
    def _drop_blank_model_ids(cls, value: list[str]) -> list[str]:
        """
        Remove empty or whitespace-only model IDs, so a placeholder entry
        in the YAML file means "no model" instead of crashing at load time.
        """
        return [m.strip() for m in value if m.strip()]

    @model_validator(mode="after")
    def _check_revision_keys(self) -> "EmbedderConfig":
        """
        Reject revisions for models that are not configured, so a typo in a
        model ID cannot silently leave the real model unpinned.
        """
        unknown = sorted(set(self.revision) - set(self.model_id))
        if unknown:
            raise ValueError(f"revision given for model(s) not in model_id: {unknown}")
        return self

    @model_validator(mode="after")
    def _check_quantization_backend(self) -> "EmbedderConfig":
        """
        Reject quantize_int8 without the OpenVINO backend, instead of
        silently running the full-precision torch model.
        """
        if self.quantize_int8 and self.backend != "openvino":
            raise ValueError("quantize_int8 requires backend 'openvino'")
        return self
