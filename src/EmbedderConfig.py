import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from pydantic import BaseModel

class EmbedderConfig(BaseModel):
    """
    Unified configuration schema for model inference and dimensionality reduction settings.
    
    Attributes:
        model_id: List of Hugging Face transformer model identifiers or local paths.
        device: Hardware device target for inference (e.g., 'cpu', 'cuda', 'mps').
        batch_size: Number of term sequences processed in a single inference batch.
        max_seq_length: Maximum token length for input sequence truncation.
        normalization: Whether to L2-normalize output embedding vectors to unit length.
        dimensionality_reduction: Master toggle for applying manifold reduction (e.g., UMAP).
        n_components: Target dimension count for reduced vectors (e.g., 2 for 2D visualization).
        n_neighbors: Number of neighboring points considered for local vs. global structure.
        min_dist: Minimum distance between points in the reduced space (controls cluster tightness).
        metric: Distance metric used to calculate spatial relationships between embeddings.
        random_state: Seed value for deterministic and reproducible dimensional reductions.
    """

    # List of Hugging Face model IDs or local file paths for term embedding.
    model_id: list[str]                 = []
    
    # Compute target for model execution ('cpu', 'cuda', 'mps', or 'auto').
    device: str                         = "cpu"

    # Inference batch size passed to the embedding model.
    batch_size: int                     = 128

    # Maximum token sequence length before input text truncation.
    max_seq_length: int                 = 128

    # Toggle L2 normalization of output vectors (recommended for cosine similarity).
    normalization: bool                 = True

    # Enable or disable post-inference dimensionality reduction.
    dimensionality_reduction: bool      = False

    # Target number of output dimensions (typically 2 or 3 for visualization).
    n_components: int                   = 2

    # Neighborhood size parameter balancing local vs. global manifold structure.
    n_neighbors: int                    = 5

    # Controls how closely packed points are allowed to be in the reduced space.
    min_dist: float                     = 0.2

    # Distance metric for vector proximity calculations ('cosine', 'euclidean', etc.).
    metric: str                         = "cosine"

    # Random seed ensuring deterministic results across reduction runs.
    random_state: int                   = 42