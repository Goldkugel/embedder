# Source

This directory contains the `embedder` package. See the [top-level README](../README.md) for installation and usage.

| File | Description |
|---|---|
| [`__init__.py`](./__init__.py) | Package entry point; exports `Embedder` and `EmbedderConfig`. |
| [`Embedder.py`](./Embedder.py) | The `Embedder` class: loads the configuration, loads one `SentenceTransformer` per configured model ID (plus one UMAP reducer per model when dimensionality reduction is enabled), and provides `embed(terms)`, which returns a dict mapping each model ID to a numpy array with one row per term. |
| [`EmbedderConfig.py`](./EmbedderConfig.py) | The `EmbedderConfig` pydantic model that validates the `embedder` section of the YAML configuration file. See [`config/README.md`](../config/README.md) for the field reference. |

The tests live in [`tests/`](../tests/) and are not part of the installed package.

## API at a glance

```python
from embedder import Embedder

embedder = Embedder(config="config/config.yaml")

result = embedder.embed(["Microcephaly", "Decreased skull size"])
# -> {"cambridgeltl/SapBERT-from-PubMedBERT-fulltext": array of shape (2, 768)}
```

With `dimensionality_reduction` enabled and at least `n_components + 2` terms, each array has `n_components` columns instead of the model's native dimension. Each reducer is fitted afresh on every `embed()` call, so reduced coordinates are only comparable within a single call.
