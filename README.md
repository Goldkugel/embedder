# Embedder

A flexible, configuration-driven Python library for encoding textual terms into dense vector representations using one or more [Sentence Transformers](https://www.sbert.net/) models, with optional [UMAP](https://umap-learn.readthedocs.io/) dimensionality reduction.

## Features

- **Pydantic-validated configuration** — model selection, hardware target, and reduction parameters are all loaded and type-checked via `EmbedderConfig`.
- **Multi-model inference** — load several Sentence Transformers models and encode the same terms with each of them in a single `embed()` call.
- **Optional dimensionality reduction** — project high-dimensional embeddings down to 2D/3D with UMAP, for visualization or downstream clustering.
- **Configurable hardware target** — run on `cpu`, `cuda`, or `mps` (the value is passed straight through to Sentence Transformers).
- **Optional L2 normalization** — normalize output vectors for cosine-similarity use cases.

## Repository Structure

```
.
├── config/
│   ├── config.yaml          # Shared pipeline configuration; must contain an "embedder" section
│   └── README.md            # Configuration reference
├── src/
│   ├── README.md            # Overview of the source files
│   ├── __init__.py
│   ├── EmbedderConfig.py    # Pydantic configuration schema
│   └── Embedder.py          # Core embedding engine
├── tests/
│   └── EmbedderTest.py      # Pytest suite (mocks Sentence Transformers/UMAP/Logger, no GPU or downloads required)
├── LICENSE
└── pyproject.toml
```

## Requirements & Installation

Requires Python 3.10 or newer. All dependencies, including the [`logger`](https://github.com/Goldkugel/logger) package (installed directly from Git), are declared in `pyproject.toml`:

```bash
pip install .
```

For running the test suite, install the `test` extra:

```bash
pip install ".[test]"
```

## Configuration

`config/config.yaml` is shared with the other packages of the pipeline; `Embedder` reads its `embedder` section, which is required. Every field has a default in `EmbedderConfig`, so only the fields you want to override need to be present:

```yaml
embedder:
  model_id:
    - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  device: "cpu"
  batch_size: 128
  max_seq_length: 128
  normalization: true
  dimensionality_reduction: true
  n_components: 2
  n_neighbors: 25
  min_dist: 0.2
  metric: "cosine"
  random_state: 253
```

See [`config/README.md`](./config/README.md) for the meaning, defaults and valid range of every field.

If `dimensionality_reduction` is enabled, each model gets its own UMAP reducer, which is **fitted afresh on every `embed()` call**. Reduced coordinates are therefore only comparable within the output of a single call; embed all terms you want to compare in one call. The reducer is only applied when there are at least `n_components + 2` terms (UMAP's spectral initialization fails on fewer points); with fewer terms, the full-dimensional embeddings are returned for that call and a message is logged.

The logger is a process-wide singleton (see the `logger` package). If it has not been configured yet when an `Embedder` is created, it is configured from the same YAML file as the embedder, so that file also needs a `logger` section.

## Usage

### Generating embeddings

```python
from embedder import Embedder

# Defaults to "./config/config.yaml" relative to the working directory.
embedder = Embedder("./config/config.yaml")

terms = [
    "Microcephaly",
    "Decreased skull size",
    "Abnormality of the kidney",
    "Renal dysplasia",
]

# Accepts a list, tuple, numpy array or pandas Series; a single string is
# treated as one term. None and NaN become "", any other value is converted
# with str().
# Returns a dict mapping model_id -> numpy.ndarray (one row per term, in order).
embeddings_map = embedder.embed(terms)

for model_id, vectors in embeddings_map.items():
    print(f"Model ID: {model_id}")
    print(f"Embeddings shape: {vectors.shape}")
```

### Validating a configuration in code

`Embedder` itself is always created from a YAML file. `EmbedderConfig` can also be built directly, for example to check settings before writing them to the file:

```python
from embedder import EmbedderConfig

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

`tests/EmbedderTest.py` uses `pytest` with mocking (`SentenceTransformer`, `umap.UMAP`, `Logger`) to test config validation, model/reducer initialization, and `embed()` — without downloading real models, requiring a GPU or writing log files. It imports the installed `embedder` package, so install it first (`pip install ".[test]"`), then run:

```bash
pytest tests/EmbedderTest.py -v
```

## License

Distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [`LICENSE`](./LICENSE) for the full text.
