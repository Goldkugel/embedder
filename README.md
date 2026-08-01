Term Embedding Engine

A flexible, configuration-driven Python library designed to encode textual terms into dense vector representations and optionally apply manifold dimensionality reduction (UMAP) across multiple Transformer models.
Key Features

    Pydantic Configuration Validation: Enforces type safety and default parameters via EmbedderConfig.

    Multi-Model Inference: Supports simultaneous loading and encoding across multiple SentenceTransformer models.

    Dimensionality Reduction: Built-in UMAP support to project high-dimensional embeddings into 2D/3D spaces for visualization or downstream clustering.

    Configurable Hardware Acceleration: Execution target configuration supporting cpu, cuda, mps, and auto.

    Vector Normalization: Option to L2-normalize vector outputs for optimal cosine similarity operations.

Repository Structure

.
├── config/
│   └── config.yaml          # Default configuration file containing 'embedder' settings
└── src/
├── EmbedderConfig.py    # Pydantic configuration schema
├── Embedder.py          # Primary embedding engine class
└── EmbedderTest.py      # Pytest unit tests for Embedder initialization and execution
Requirements & Installation

This project depends on an external Logger package in addition to standard ML/NLP dependencies.

Install required dependencies:

pip install sentence-transformers umap-learn pydantic pyyaml numpy pytest
Configuration (config.yaml)

Configuration files must include an embedder section key containing runtime parameters:

embedder:
List of Hugging Face transformer model identifiers or local model paths

model_id:
- "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
Compute target: 'cpu', 'cuda', 'mps', or 'auto'

device: "cpu"
Batch size for model inference

batch_size: 128
Maximum token sequence length for input truncation

max_seq_length: 128
Toggle L2 normalization of output vectors

normalization: true
Master toggle for UMAP dimensionality reduction

dimensionality_reduction: true
UMAP Parameters

n_components: 2       # Target dimensions (e.g., 2 for 2D visual plots)
n_neighbors: 5        # Local vs. global balance
min_dist: 0.2         # Cluster packing tightness in output space
metric: "cosine"      # Spatial distance metric
random_state: 42      # Seed for deterministic reduction runs
Usage
1. Generating Term Embeddings

from Embedder import Embedder
Initialize Embedder (defaults to "../config/config.yaml" relative to src/)

embedder = Embedder("../config/config.yaml")
Define input terms

terms = [
"Microcephaly",
"Decreased skull size",
"Abnormality of the kidney",
"Renal dysplasia"
]
Generate embeddings (returns a dictionary mapping model_id -> numpy ndarray)

embeddings_map = embedder.embed(terms)

for model_id, vectors in embeddings_map.items():
print(f"Model ID: {model_id}")
print(f"Embeddings Shape: {vectors.shape}")
2. Programmatic Configuration with EmbedderConfig

from EmbedderConfig import EmbedderConfig
Instantiate configuration directly

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

The unit test suite lives directly in src/EmbedderTest.py and uses pytest with extensive mocking to test initialization, model loading, and batch encoding without requiring GPU resources or remote model downloads.

Run the test suite from the src/ directory or project root:

pytest src/EmbedderTest.py -v
