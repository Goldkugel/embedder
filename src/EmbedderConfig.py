"""
Pydantic configuration model for the Embedder class.

Defines and validates all settings read from the "embedder" section of
the YAML configuration file. Embedder.__init__ builds it from that file; it
can also be constructed directly, e.g. to validate settings in code.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    # Compute target for model execution. Common values:
    # "cpu"  — standard CPU inference (slowest, always available)
    # "cuda" — NVIDIA GPU via CUDA (fastest for large batches)
    # "mps"  — Apple Silicon GPU via Metal Performance Shaders
    device: str                    = "cpu"

    # Number of terms processed in a single inference batch. Larger batches
    # are faster on GPU but require more VRAM; reduce if you hit OOM errors.
    batch_size: int                = Field(default=128, gt=0)

    # Maximum token sequence length before input text is truncated. Inputs
    # longer than this are silently cut off — increase if your terms are long.
    max_seq_length: int            = Field(default=128, gt=0)

    # Whether to L2-normalise output vectors to unit length. Recommended when
    # computing cosine similarity downstream: for unit vectors, the plain dot
    # product already equals the cosine similarity.
    normalization: bool            = True

    # Master toggle for applying UMAP dimensionality reduction to the
    # embeddings after inference. Disable for downstream tasks that need
    # full-dimensional vectors (e.g. vector database indexing). Reduction is
    # skipped for inputs with fewer than n_components + 2 terms (see embed()).
    dimensionality_reduction: bool = False

    # Target number of output dimensions after reduction. Use 2 or 3 for
    # visualisation, or a higher value (e.g. 32–64) for downstream ML tasks
    # that benefit from reduced but still informative representations.
    n_components: int              = Field(default=2, ge=1)

    # Number of neighbouring points considered when constructing the UMAP
    # graph. Smaller values emphasise fine local structure; larger values
    # capture broader global structure at the cost of local detail. If an
    # input has fewer terms than this, UMAP lowers it to (terms - 1).
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
