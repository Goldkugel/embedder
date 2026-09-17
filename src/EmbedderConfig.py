"""
Pydantic configuration model for the Embedder class.

Defines and validates all settings read from the "embedder" section of
the YAML configuration file. Used exclusively by Embedder.__init__.
"""

from pydantic import BaseModel, ConfigDict


class EmbedderConfig(BaseModel):
    """
    Pydantic model defining and validating the configuration schema for
    the Embedder class.

    Instances are typically created via `EmbedderConfig.model_validate(data)`,
    where `data` is a dict loaded from the "embedder" section of a YAML config
    file. Pydantic validates field types and enforces defaults for any fields
    not present in the supplied data.
    """

    # Reject any keys in the input data that aren't explicitly defined below.
    # This catches typos or outdated config keys early, instead of silently
    # ignoring them.
    model_config = ConfigDict(extra="forbid")

    # List of Hugging Face model IDs or local file paths for term embedding.
    # Each model is loaded separately and produces its own set of embeddings.
    model_id: list[str]            = []

    # Compute target for model execution. Common values:
    # "cpu"  — standard CPU inference (slowest, always available)
    # "cuda" — NVIDIA GPU via CUDA (fastest for large batches)
    # "mps"  — Apple Silicon GPU via Metal Performance Shaders
    device: str                    = "cpu"

    # Number of terms processed in a single inference batch. Larger batches
    # are faster on GPU but require more VRAM; reduce if you hit OOM errors.
    batch_size: int                = 128

    # Maximum token sequence length before input text is truncated. Inputs
    # longer than this are silently cut off — increase if your terms are long.
    max_seq_length: int            = 128

    # Whether to L2-normalise output vectors to unit length. Recommended when
    # computing cosine similarity downstream, since it reduces dot product to
    # cosine distance without an extra normalisation step.
    normalization: bool            = True

    # Master toggle for applying UMAP dimensionality reduction to the
    # embeddings after inference. Disable for downstream tasks that need
    # full-dimensional vectors (e.g. vector database indexing).
    dimensionality_reduction: bool = False

    # Target number of output dimensions after reduction. Use 2 or 3 for
    # visualisation, or a higher value (e.g. 32–64) for downstream ML tasks
    # that benefit from reduced but still informative representations.
    n_components: int              = 2

    # Number of neighbouring points considered when constructing the UMAP
    # graph. Smaller values emphasise fine local structure; larger values
    # capture broader global structure at the cost of local detail.
    n_neighbors: int               = 5

    # Minimum distance between points in the reduced space. Smaller values
    # produce tighter, more clustered layouts; larger values spread points
    # out more uniformly.
    min_dist: float                = 0.2

    # Distance metric used to compute proximity between embedding vectors
    # before reduction. "cosine" is recommended when embeddings are
    # L2-normalised; "euclidean" suits raw unnormalised vectors.
    metric: str                    = "cosine"

    # Random seed for UMAP, ensuring that repeated runs with the same input
    # produce the same reduced layout. Set to None to disable determinism.
    random_state: int              = 42