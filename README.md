# Embedder

A flexible, configuration-driven Python library for encoding textual terms into dense vector representations using one or more [Sentence Transformers](https://www.sbert.net/) models, with optional [UMAP](https://umap-learn.readthedocs.io/) dimensionality reduction.

## Features

- **Pydantic-validated configuration** — model selection, hardware target, and reduction parameters are all loaded and type-checked via `EmbedderConfig`.
- **Multi-model inference** — load and encode with several Sentence Transformers models simultaneously in a single call.
- **Optional dimensionality reduction** — project high-dimensional embeddings down to 2D/3D with UMAP, for visualization or downstream clustering.
- **Configurable hardware target** — run on `cpu`, `cuda`, or `mps` (`auto` is also accepted by the configuration schema, though device selection is passed straight through to Sentence Transformers without additional handling).
- **Optional L2 normalization** — normalize output vectors for cosine-similarity use cases.

## Repository Structure

```
.
├── config/
│   └── config.yaml          # Configuration file; must contain an "embedder" section
├── src/
│   ├── __init__.py
│   ├── EmbedderConfig.py    # Pydantic configuration schema
│   ├── Embedder.py          # Core embedding engine
│   └── EmbedderTest.py      # Pytest suite (mocks Sentence Transformers/UMAP, no GPU or downloads required)
├── LICENSE
└── pyproject.toml
```

## Requirements & Installation

This project depends on the [`logger`](https://github.com/Goldkugel/logger) package (installed directly from Git) plus the machine learning dependencies used by `Embedder.py`. Only the `logger` dependency is currently declared in `pyproject.toml`; the rest need to be installed separately:

```bash
pip install sentence-transformers umap-learn pydantic pyyaml numpy
pip install "logger @ git+https://github.com/Goldkugel/logger.git@v1.0.5"
```

For running the test suite, also install:

```bash
pip install pytest
```

## Configuration

`config.yaml` must contain an `embedder` section. Every field has a default in `EmbedderConfig`, so only the fields you want to override need to be present:

```yaml
embedder:
  model_id:
    - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  device: "cpu"
  batch_size: 64
  max_seq_length: 128
  normalization: false
  dimensionality_reduction: true
  n_components: 2
  n_neighbors: 5
  min_dist: 0.2
  metric: "cosine"
  random_state: 42
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model_id` | `list[str]` | `[]` | Hugging Face model identifiers or local paths to load. One model is loaded per entry. |
| `device` | `str` | `"cpu"` | Hardware target passed to Sentence Transformers (`cpu`, `cuda`, `mps`, or `auto`). |
| `batch_size` | `int` | `128` | Batch size used during inference. |
| `max_seq_length` | `int` | `128` | Maximum token sequence length; set on each loaded model after initialization. |
| `normalization` | `bool` | `True` | Whether to L2-normalize output embeddings. |
| `dimensionality_reduction` | `bool` | `False` | Whether to fit and apply a UMAP reducer after encoding. |
| `n_components` | `int` | `2` | Target dimensionality for UMAP output. |
| `n_neighbors` | `int` | `5` | UMAP neighborhood size (local vs. global structure). |
| `min_dist` | `float` | `0.2` | UMAP minimum distance between points in the reduced space. |
| `metric` | `str` | `"cosine"` | Distance metric used by UMAP. |
| `random_state` | `int` | `42` | Seed for deterministic UMAP output. |

If `dimensionality_reduction` is enabled, the UMAP reducer is only applied to a given model's output when the number of terms exceeds `n_components` — otherwise the raw embeddings are returned unchanged for that call.

## Usage

### Generating embeddings

```python
from Embedder import Embedder

# Defaults to "../config/config.yaml" relative to src/ if no path is given.
embedder = Embedder("../config/config.yaml")

terms = [
    "Microcephaly",
    "Decreased skull size",
    "Abnormality of the kidney",
    "Renal dysplasia",
]

# Returns a dict mapping model_id -> numpy.ndarray (one row per term, in order).
embeddings_map = embedder.embed(terms)

for model_id, vectors in embeddings_map.items():
    print(f"Model ID: {model_id}")
    print(f"Embeddings shape: {vectors.shape}")
```

### Building a configuration programmatically

```python
from EmbedderConfig import EmbedderConfig

config = EmbedderConfig(
    model_id=["sentence-transformers/all-MiniLM-L6-v2"],
    device="cpu",
    batch_size=32,
    normalization=True,
    dimensionality_reduction=False,
)

print(f"Models to load: {config.model_id}")
print(f"Batch size: {config.batch_size}")
```

## Running Unit Tests

`src/EmbedderTest.py` uses `pytest` with mocking (`Embedder.SentenceTransformer`, `Embedder.umap.UMAP`, `Embedder.Logger`) to test config loading, model/reducer initialization, and `embed()` — without downloading real models or requiring a GPU.

Run from the `src/` directory or the project root:

```bash
pytest src/EmbedderTest.py -v
```

## License

Distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [`LICENSE`](./LICENSE) for the full text.
