"""
Pydantic configuration model for the Embedder class.

Defines and validates all settings read from the "embedder" section of
the YAML configuration file. Embedder.__init__ builds it from that file or
from a dict; it can also be constructed directly, e.g. to validate settings
in code.
"""

from typing     import Annotated, Any, Literal
from pydantic   import BaseModel, ConfigDict, Field, field_validator, model_validator
from umap.umap_ import dist as _umap_dist, pynn_named_distances as _pynn_distances

# Metric names UMAP accepts for dense input. UMAP itself only checks them when
# a reducer is fitted, after all terms have been embedded.
umap_metrics: frozenset = frozenset(_umap_dist.named_distances) | frozenset(_pynn_distances)

# Value types of the settings that can also be given per model.
PoolingMode = Literal["mean", "cls", "max", "mean_sqrt_len_tokens", "weightedmean", "lasttoken"]
SeqLength   = Annotated[int, Field(gt=0)]
Lowercase   = bool | Literal["auto"]

# Settings that take one value for every model or a mapping from model ID
# to value; see EmbedderConfig.model_setting().
per_model_settings: tuple = ("pooling", "max_seq_length", "lowercase")


class EmbedderConfig(BaseModel):
    """
    Configuration schema of the Embedder class, usually created with
    `EmbedderConfig.model_validate(data)` from the "embedder" section of a
    YAML config file. Missing fields get their defaults.
    """

    # Unknown keys (typos, outdated settings) are errors, not ignored.
    model_config = ConfigDict(extra="forbid")

    # Hugging Face model IDs or local paths; each gives its own embeddings.
    # Blank entries (e.g. a `- ""` placeholder) are dropped.
    model_id: list[str]            = []

    # Revision (commit hash, branch or tag) per model ID; a commit hash makes
    # runs reproducible. Unlisted models load the latest revision. Every key
    # must also appear in model_id.
    revision: dict[str, str]       = {}

    # pooling, max_seq_length and lowercase take either one value for every
    # model or a mapping from model ID to value, like revision. Models not in
    # such a mapping get the field's default, and every key must also appear
    # in model_id.

    # How token vectors are combined into one embedding, replacing every
    # model's own pooling. "mean" is sentence-transformers' fallback for plain
    # Hugging Face models, "cls" the [CLS] token (what SapBERT was trained
    # with). Use the pooling a model was trained with; a mismatch gives
    # vectors of lower quality.
    pooling: PoolingMode | dict[str, PoolingMode] = "mean"

    # Torch device, e.g. "cpu", "cuda" or "mps".
    device: str                    = "cpu"

    # Inference engine. "openvino" needs the "openvino" extra and converts each
    # model when loaded; without quantize_int8 it gives torch's vectors and is
    # not noticeably faster. ONNX is not offered: its sentence-transformers
    # extra cannot be installed together with sentence-transformers 6.
    backend: Literal["torch", "openvino"] = "torch"

    # 8-bit integer weights (OpenVINO only). With SapBERT on the CPU about
    # twice as fast as torch, cosine similarity >= 0.998 to full precision.
    # Gets its own cache file.
    quantize_int8: bool            = False

    # Lowercase terms before encoding and cache lookup. "auto" lowercases only
    # for models whose tokenizer does anyway (uncased, e.g. SapBERT), which
    # avoids caching "Seizure" and "seizure" twice. Whitespace is always
    # collapsed.
    lowercase: Lowercase | dict[str, Lowercase] = "auto"

    # Folder for the embedding cache (one JSON file per model and settings);
    # None disables it. Pin the revision when caching, or an updated model
    # would still be served old vectors.
    cache_folder: str | None       = None

    # Decimals of the cached values (embed() then returns rounded values too).
    # 6 changes a cosine similarity by < 1e-6 and saves about a quarter of the
    # file size; None stores exact float32 values.
    cache_decimals: int | None     = Field(default=6, ge=1)

    # Terms encoded between two progress messages and cache appends, so an
    # interrupted run resumes from the last one.
    progress_interval: int         = Field(default=10000, gt=0)

    # CPU thread limit for shared machines; None uses all cores. For PyTorch
    # this is process-wide, so it affects other PyTorch code as well.
    num_threads: int | None        = Field(default=None, gt=0)

    # Terms per inference batch; reduce it if you run out of memory.
    batch_size: int                = Field(default=128, gt=0)

    # Maximum tokens per term; longer inputs are silently truncated.
    max_seq_length: SeqLength | dict[str, SeqLength] = 128

    # L2-normalise the vectors, so a dot product equals the cosine similarity.
    normalization: bool            = True

    # Create one UMAP reducer per model for Embedder.reduce() (e.g. for
    # plots). embed() always returns the full-dimensional vectors.
    dimensionality_reduction: bool = False

    # Dimensions after reduction: 2 or 3 for plots, more (e.g. 32-64) for ML.
    n_components: int              = Field(default=2, ge=1)

    # UMAP neighbourhood size: small values keep local, large values global
    # structure. UMAP lowers it to (rows - 1) for smaller inputs.
    n_neighbors: int               = Field(default=5, ge=2)

    # Minimum distance in the reduced space; smaller values cluster tighter.
    min_dist: float                = Field(default=0.2, ge=0.0)

    # UMAP distance metric, one of umap_metrics.
    metric: str                    = "cosine"

    # UMAP seed for reproducible layouts; None lets UMAP run multi-threaded.
    random_state: int | None       = 42

    @field_validator("model_id")
    @classmethod
    def _drop_blank_model_ids(cls, value: list[str]) -> list[str]:
        """
        Remove empty or whitespace-only model IDs, so a placeholder entry
        in the YAML file means "no model" instead of crashing at load time,
        and reject duplicate IDs.
        """
        model_ids = [m.strip() for m in value if m.strip()]
        # embed() could only return one result for a repeated ID.
        duplicates = sorted({m for m in model_ids if model_ids.count(m) > 1})
        if duplicates:
            raise ValueError(f"model_id lists model(s) more than once: {duplicates}")
        return model_ids

    @field_validator("revision")
    @classmethod
    def _strip_revision_keys(cls, value: dict[str, str]) -> dict[str, str]:
        """
        Strip the model IDs and revisions like the entries of model_id, so
        the same spelling with stray spaces matches in both places.
        """
        return {m.strip(): r.strip() for m, r in value.items()}

    @field_validator(*per_model_settings)
    @classmethod
    def _strip_per_model_keys(cls, value: Any) -> Any:
        """
        Strip the model IDs of a per-model mapping like the entries of
        model_id; a single value is returned unchanged.
        """
        if isinstance(value, dict):
            return {m.strip(): v for m, v in value.items()}
        return value

    @field_validator("metric")
    @classmethod
    def _check_metric(cls, value: str) -> str:
        """
        Reject metrics UMAP does not know, which UMAP itself would only
        report in reduce(), after all terms have been embedded.
        """
        if value not in umap_metrics:
            raise ValueError(
                f"unknown UMAP metric {value!r}; use one of {sorted(umap_metrics)}"
            )
        return value

    @model_validator(mode="after")
    def _check_per_model_keys(self) -> "EmbedderConfig":
        """
        Reject revisions and per-model settings for models that are not
        configured, so a typo in a model ID cannot silently leave the real
        model unpinned or with the default setting.
        """
        for name in ("revision", *per_model_settings):
            value = getattr(self, name)
            if not isinstance(value, dict):
                continue
            unknown = sorted(set(value) - set(self.model_id))
            if unknown:
                raise ValueError(f"{name} given for model(s) not in model_id: {unknown}")
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

    def model_setting(self, name: str, model_id: str) -> Any:
        """
        The value of a per-model setting (pooling, max_seq_length or
        lowercase) for one model: the single configured value, the model's
        entry in the mapping, or the field's default for a model not in it.

        Raises
        ------
        ValueError
            If `name` is not one of the per-model settings.
        """
        if name not in per_model_settings:
            raise ValueError(f"{name!r} is not a per-model setting; use one of {list(per_model_settings)}")
        value = getattr(self, name)
        if not isinstance(value, dict):
            return value
        return value.get(model_id, type(self).model_fields[name].default)
