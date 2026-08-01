Markdown

# Term Embedding Engine

A flexible, configuration-driven Python library designed to encode textual terms, clinical concepts, and domain terminologies into dense vector embeddings with built-in post-processing and dimensionality reduction.

---

## Features

- **Configuration-Driven Architecture:** Managed via `EmbeddingConfig` backed by Pydantic validation and YAML configuration files.
- **Multi-Model Inference:** Support for running inference across multiple Transformer models simultaneously (e.g., SapBERT, PubMedBERT, MiniLM).
- **Optional Dimensionality Reduction:** Native integration with UMAP for projecting high-dimensional term vectors to 2D/3D spaces for visualization and clustering.
- **Custom Hardware Support:** Configurable compute targets (`cpu`, `cuda`, `mps`, `auto`) and memory/speed parameters like `batch_size` and `max_seq_length`.
- **L2 Vector Normalization:** Built-in embedding normalization, optimizing output vectors directly for cosine distance metrics.

---

## Directory Structure

```text
.
├── config/
│   └── config.yaml           # YAML configuration file
├── src/
│   ├── EmbeddingConfig.py    # Pydantic schema for configuration validation
│   ├── Embedder.py           # Core embedding & reduction engine
│   └── Logger.py             # Internal logging utility
└── tests/
    ├── EmbeddingConfigTest.py# Unit tests for Pydantic configuration
    └── EmbedderTest.py       # Unit tests for Embedder inference & reduction

Installation

Ensure you have Python 3.9+ installed, then install the required dependencies:
Bash

pip install sentence-transformers umap-learn pydantic pyyaml numpy pytest

Configuration (config.yaml)

The library is configured via a YAML file under the embedder section key.
YAML

embedder:
  # List of Hugging Face transformer model identifiers or local directory paths
  model_id:
    - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  
  # Execution device: "cpu", "cuda", "mps", or "auto"
  device: "cpu"
  
  # Inference batch size
  batch_size: 128
  
  # Maximum sequence length (tokens) for input truncation
  max_seq_length: 128
  
  # L2-normalize output embeddings to unit length
  normalization: true
  
  # Master toggle for post-inference dimensionality reduction
  dimensionality_reduction: true
  
  # UMAP Manifold Parameters
  n_components: 2       # Target dimensions (e.g., 2 for 2D visual layout)
  n_neighbors: 5        # Local vs. global balance
  min_dist: 0.2         # Cluster packing tightness
  metric: "cosine"      # Distance metric
  random_state: 42      # Seed for reproducibility

Usage
1. Basic Term Embedding
Python

from Embedder import Embedder

# Initialize embedder using default config path ("../config/config.yaml")
embedder = Embedder(config="../config/config.yaml")

# Define terms to embed (e.g., HPO phenotype terms)
terms = [
    "Microcephaly",
    "Decreased skull size",
    "Abnormality of the kidney",
    "Renal dysplasia"
]

# Generate embeddings
embeddings_map = embedder.embed(terms)

# Access results by model name
for model_name, vectors in embeddings_map.items():
    print(f"Model: {model_name}")
    print(f"Output shape: {vectors.shape}")  # e.g., (4, 2) if dimensionality reduction is enabled

2. Inspecting Configuration Programmatically
Python

from EmbeddingConfig import EmbeddingConfig

# Direct dict validation
config_data = {
    "model_id": ["sentence-transformers/all-MiniLM-L6-v2"],
    "device": "cpu",
    "batch_size": 64,
    "max_seq_length": 128,
    "normalization": True,
    "dimensionality_reduction": False
}

config = EmbeddingConfig.model_validate(config_data)
print(f"Batch size: {config.batch_size}")

Testing

The repository includes a unit test suite built with pytest utilizing mocks for fast execution without needing external GPU resources or downloading large model weights.

Run all tests:
Bash

pytest -v

Run specific test modules:
Bash

pytest tests/EmbedderTest.py -v
pytest tests/EmbeddingConfigTest.py -v


<ElicitationsGroup message="Is there anything else you'd like to add to the library?">
  <Elicitation label="Add a CLI script for embedding files from terminal" query="Create a command-line interface (CLI) script in Python that takes an input CSV file of terms, uses Embedder, and saves the vectors to an output Parquet file."/>
  <Elicitation label="Create a GitHub Actions workflow for CI/CD" query="Write a GitHub Actions YAML workflow to automatically run Pytest on pushes and pull requests."/>
</ElicitationsGroup>