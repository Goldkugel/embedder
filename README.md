# Embedder

A flexible, configuration-driven Python library for encoding textual terms into dense vector representations using one or more [Sentence Transformers](https://www.sbert.net/) models, with optional [UMAP](https://umap-learn.readthedocs.io/) dimensionality reduction.

## Features

- **Pydantic-validated configuration** — model selection, hardware target, and reduction parameters are all loaded and type-checked via `EmbedderConfig`.
- **Multi-model inference** — load several Sentence Transformers models and encode the same terms with each of them in a single `embed()` call.
- **Reproducible models** — pin each Hugging Face model to an exact revision (commit hash), so an update on the Hub cannot change your results.
- **Configurable pooling** — choose how token vectors are combined into one embedding (`mean`, `cls`, ...), e.g. `cls` to match how SapBERT was trained.
- **Flexible configuration** — create an `Embedder` from a YAML file, a dict or an `EmbedderConfig` object.
- **Embedding cache** — optionally keep a JSON file per model mapping each term to its embedding, so only terms not seen before are encoded. Values are rounded (6 decimals by default) to keep the files small, and whitespace variants (and, for uncased models such as SapBERT, case variants) of a term share one entry.
- **Progress and checkpoints** — long runs log their progress every `progress_interval` terms and append the new embeddings to the cache, so an interrupted run resumes where it stopped. Appending keeps checkpoints fast however large the cache grows, and a file cut off by a killed job is repaired automatically.
- **Optional dimensionality reduction** — `reduce()` projects embeddings down to 2D/3D with UMAP, for visualization or clustering. `embed()` always returns the full-dimensional vectors, which are the ones to use for similarity and matching.
- **CPU by default** — runs on `cpu`, with an optional limit on the number of CPU threads for shared machines; `cuda` or `mps` can be configured (the value is passed straight through to Sentence Transformers).
- **Faster CPU inference** — optionally run the models with OpenVINO and int8 weights, about twice as fast on the CPU for SapBERT.
- **Optional L2 normalization** — normalize output vectors for cosine-similarity use cases.

## Repository Structure

```
.
├── .github/workflows/
│   └── tests.yml            # CI: fast tests on Python 3.10/3.11/3.14, plus real-model smoke tests
├── config/
│   ├── config.yaml          # Configuration with "embedder" and "logger" sections
│   └── README.md            # Configuration reference
├── src/
│   ├── README.md            # Overview of the source files
│   ├── __init__.py
│   ├── EmbedderConfig.py    # Pydantic configuration schema
│   └── Embedder.py          # Core embedding engine
├── tests/
│   └── EmbedderTest.py      # Pytest suite (mocked by default; slow real-model tests)
├── LICENSE
└── pyproject.toml
```

## Requirements & Installation

Requires Python 3.10 or newer. All dependencies, including the [`logger`](https://github.com/Goldkugel/logger) package (installed directly from Git), are declared in `pyproject.toml` with version bounds.

Since the embedder runs on the CPU by default, install the CPU-only build of PyTorch first. Otherwise pip installs the default build, which bundles several GB of CUDA libraries that are never used:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the package:

```bash
pip install .
```

For the OpenVINO backend (`backend: "openvino"`), install the `openvino` extra. It also needs `torchvision`; install that from the CPU index as well, otherwise pip installs a CUDA build that does not match the CPU-only PyTorch:

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install ".[openvino]"
```

For running the test suite, install the `test` extra:

```bash
pip install ".[test]"
```

## Configuration

`config/config.yaml` contains an `embedder` section (required) and a `logger` section for the [`logger`](https://github.com/Goldkugel/logger) package. Every field has a default in `EmbedderConfig`, so only the fields you want to override need to be present:

```yaml
embedder:
  model_id:
    - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  revision:
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"
  pooling: "mean"
  cache_folder: "./data/cache/embeddings/"
  cache_decimals: 6
  progress_interval: 10000
  num_threads: null
  device: "cpu"
  backend: "torch"
  quantize_int8: False
  lowercase: "auto"
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

If `dimensionality_reduction` is enabled, each model gets its own UMAP reducer, which is **fitted afresh on every `reduce()` call**. Reduced coordinates are therefore only comparable within the output of a single call; reduce all terms you want to compare in one call. The reducer is only applied when there are at least `n_components + 2` rows (UMAP's spectral initialization fails on fewer points); with fewer rows, the full-dimensional embeddings are returned for that call and a warning is logged.

The logger is a process-wide singleton (see the `logger` package). If it has not been configured yet when an `Embedder` is created, it is configured from the same YAML file as the embedder, so that file also needs a `logger` section; without it, `Embedder` raises a `ValueError` before loading any model.

## Usage

### Creating an Embedder

```python
from embedder import Embedder, EmbedderConfig

# From a YAML file (default: "./config/config.yaml" relative to the
# working directory); str or pathlib.Path.
embedder = Embedder("./config/config.yaml")

# From a dict: either the whole configuration (with "embedder" and
# optionally "logger" keys) or just the embedder settings.
embedder = Embedder({"model_id": ["cambridgeltl/SapBERT-from-PubMedBERT-fulltext"]})

# From an EmbedderConfig.
embedder = Embedder(EmbedderConfig(model_id=["cambridgeltl/SapBERT-from-PubMedBERT-fulltext"]))
```

If the logger singleton has not been configured yet, it is configured from the same source: the YAML file, or the `logger` key of a full configuration dict. Otherwise (embedder settings only, or an `EmbedderConfig`) configure it first with `Logger("path/to/config.yaml")`, or it falls back to `./config/config.yaml`.

### Generating embeddings

```python
from embedder import Embedder

embedder = Embedder("./config/config.yaml")

terms = [
    "Microcephaly",
    "Decreased skull size",
    "Abnormality of the kidney",
    "Renal dysplasia",
]

# Accepts a list, tuple, numpy array or pandas Series; a single string is
# treated as one term. Missing values (None, NaN, pd.NA, NaT) become "", any
# other value is converted with str().
# Returns a dict mapping model_id -> numpy.ndarray (one row per term, in order)
# with the model's full dimension (768 for SapBERT).
embeddings_map = embedder.embed(terms)

for model_id, vectors in embeddings_map.items():
    print(f"Model ID: {model_id}")
    print(f"Embeddings shape: {vectors.shape}")
```

### Reducing embeddings for plots

Requires `dimensionality_reduction: true` in the configuration; otherwise `reduce()` raises a `RuntimeError`.

```python
# Same keys as embeddings_map; each array has n_components columns.
reduced_map = embedder.reduce(embeddings_map)
```

### Validating a configuration in code

`EmbedderConfig` can be built on its own, for example to check settings before writing them to the file or passing them to `Embedder`:

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

The tests import the installed `embedder` package, so install it first (`pip install ".[test]"`). Run them from the repository root: pytest collects every `tests/*Test.py` file (configured in `pyproject.toml`), and one test checks the shipped `config/config.yaml`.

By default, only the fast tests run. They mock `SentenceTransformer`, `umap.UMAP` and `Logger` to test config validation, model/reducer initialization, `embed()`, the cache files, progress checkpoints and `reduce()`, without downloading models or writing log files:

```bash
pytest -v
```

Four slow smoke tests run a small real model (`all-MiniLM-L6-v2`, about 90 MB, downloaded on first use) with real UMAP on the CPU, to catch API changes in the libraries, check that the pooling setting takes effect, check the pooling warning and the `max_seq_length` limit against the real model, and compare the OpenVINO backend with torch (skipped unless the `openvino` extra is installed):

```bash
pytest -v -m slow
```

Both run automatically on GitHub Actions on every push and pull request; the fast tests also report line and branch coverage (`--cov=embedder --cov-branch`) in the job log and on the run's summary page.

## License

Distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [`LICENSE`](./LICENSE) for the full text.
