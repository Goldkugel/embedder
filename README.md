# Term Embedding Engine

A flexible, configuration-driven Python library designed to encode textual terms into dense vector representations and optionally apply manifold dimensionality reduction (UMAP) across multiple Transformer models.

---

## Key Features

- **Pydantic Configuration Validation:** Enforces type safety and default parameters using `EmbedderConfig`.
- **Multi-Model Inference:** Supports simultaneous loading and encoding across multiple SentenceTransformer models.
- **Dimensionality Reduction:** Built-in UMAP support to project high-dimensional embeddings into 2D/3D spaces for visualization or downstream processing.
- **Configurable Hardware Acceleration:** Execution target configuration supporting `cpu`, `cuda`, `mps`, and `auto`.
- **Vector Normalization:** Built-in option to L2-normalize vector outputs for optimal cosine similarity operations.

---

## Directory Structure

```text
.
├── config/
│   └── config.yaml          # Default configuration file containing 'embedder' settings
├── src/
│   ├── EmbedderConfig.py    # Pydantic schema model
│   ├── Embedder.py          # Primary embedding engine class
│   └── Logger.py            # Custom logging utility
└── tests/
    └── EmbedderTest.py      # Pytest unit tests for Embedder initialization and execution

Installation

Install the required Python packages:
Bash

pip install sentence-transformers umap-learn pydantic pyyaml numpy pytest

Configuration (config.yaml)

Configuration files must include an embedder root dictionary containing the model runtime parameters:
YAML

embedder:
  # List of Hugging Face transformer model identifiers or local model paths
  model_id:
    - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  
  # Compute target: 'cpu', 'cuda', 'mps', or 'auto'
  device: "cpu"
  
  # Batch size for model.encode()
  batch_size: 128
  
  # Maximum token sequence length for truncation
  max_seq_length: 128
  
  # Toggle L2 normalization of output vectors
  normalization: true
  
  # Master toggle for UMAP dimensionality reduction
  dimensionality_reduction: true
  
  # UMAP Parameters
  n_components: 2       # Target dimensions (e.g., 2 for 2D visual plots)
  n_neighbors: 5        # Local vs. global balance
  min_dist: 0.2         # Cluster packing tightness in output space
  metric: "cosine"      # Spatial distance metric
  random_state: 42      # Seed for deterministic reduction runs

Usage
1. Generating Term Embeddings
Python

from Embedder import Embedder

# Initialize Embedder (defaults to "../config/config.yaml" if no path is provided)
embedder = Embedder("config/config.yaml")

# Define input terms
terms = [
    "Microcephaly",
    "Decreased skull size",
    "Abnormality of the kidney",
    "Renal dysplasia"
]

# Generate embeddings (returns a dictionary mapping model_id -> numpy ndarray)
embeddings_map = embedder.embed(terms)

for model_id, vectors in embeddings_map.items():
    print(f"Model ID: {model_id}")
    print(f"Embeddings Shape: {vectors.shape}")

2. Programmatic Configuration with EmbedderConfig
Python

from EmbedderConfig import EmbedderConfig

# Instantiate configuration directly
config = EmbedderConfig(
    model_id=["sentence-transformers/all-MiniLM-L6-v2"],
    device="cpu",
    batch_size=32,
    normalization=True,
    dimensionality_reduction=False
)

print(f"Models to load: {config.model_id}")
print(f"Batch size: {config.batch_size}")

Running Unit Tests

The test suite uses pytest and extensive mocking to test model loading, UMAP initialization, and batching without requiring actual model weight downloads or GPU hardware.

Execute all tests from the repository root:
Bash

pytest tests/EmbedderTest.py -v


<FollowUp label="Want me to write a CLI tool or script for batch processing CSV files?" query="Write a command-line interface script that accepts a CSV of terms, processes them with Embedder, and saves the resulting embeddings."/>
