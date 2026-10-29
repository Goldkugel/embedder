# Source

This directory contains the `embedder` package. See the [top-level README](../README.md) for installation and usage.

| File | Description |
|---|---|
| [`__init__.py`](./__init__.py) | Package entry point; exports `Embedder` and `EmbedderConfig`. |
| [`Embedder.py`](./Embedder.py) | The `Embedder` class: loads the configuration, loads one `SentenceTransformer` per configured model ID at its pinned revision (plus one UMAP reducer per model when dimensionality reduction is enabled). Sets the configured pooling on each model and, optionally, a CPU thread limit. Provides `embed(terms)`, which returns a dict mapping each model ID to a numpy array with one full-dimensional row per term (encoding only terms missing from the JSON cache, with progress messages and checkpoints), `reduce(embeddings)`, which projects such a dict with UMAP, and `save_cache()` / `load_cache()`, which write and read cache files safely. |
| [`EmbedderConfig.py`](./EmbedderConfig.py) | The `EmbedderConfig` pydantic model that validates the `embedder` section of the YAML configuration file. See [`config/README.md`](../config/README.md) for the field reference. |

The tests live in [`tests/`](../tests/) and are not part of the installed package.

## API at a glance

```python
from embedder import Embedder

embedder = Embedder(config="config/config.yaml")

full = embedder.embed(["Microcephaly", "Decreased skull size", "Renal dysplasia", "Seizure"])
# -> {"cambridgeltl/SapBERT-from-PubMedBERT-fulltext": array of shape (4, 768)}

reduced = embedder.reduce(full)   # requires dimensionality_reduction: True
# -> {"cambridgeltl/SapBERT-from-PubMedBERT-fulltext": array of shape (4, 2)}
```

`reduce()` only reduces inputs with at least `n_components + 2` rows. Each reducer is fitted afresh on every `reduce()` call, so reduced coordinates are only comparable within a single call.
