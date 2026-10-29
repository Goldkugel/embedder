# Embedder Configuration

[`config.yaml`](./config.yaml) is a shared configuration file with sections for several packages of the pipeline (`adapter`, `logger`, `normalizer`, `llm`, `embedder`). `Embedder` reads only two of them:

- **`embedder`** — validated against the [`EmbedderConfig`](../src/EmbedderConfig.py) Pydantic model and documented below. Required: a file without it raises a `ValueError`.
- **`logger`** — only read if no other part of the program has configured the [`logger`](https://github.com/Goldkugel/logger) singleton yet. See the `logger` package's README for its fields.

All other sections are ignored by `Embedder`.

---

## `embedder` section

```yaml
embedder:
  model_id:
    - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  device: "cpu"
  batch_size: 128
  max_seq_length: 128
  normalization: True
  dimensionality_reduction: True
  n_components: 2
  n_neighbors: 25
  min_dist: 0.2
  metric: "cosine"
  random_state: 253
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model_id` | list of strings | `[]` | Hugging Face model IDs or local paths. One `SentenceTransformer` is loaded per entry, and `embed()` returns one result per entry. Blank entries are ignored, so `- ""` means "no model". |
| `device` | string | `"cpu"` | Hardware target passed to Sentence Transformers: `"cpu"`, `"cuda"` (NVIDIA GPU) or `"mps"` (Apple Silicon). |
| `batch_size` | integer | `128` | Number of terms encoded per forward pass. Larger is faster on a GPU but needs more memory. Must be > 0. |
| `max_seq_length` | integer | `128` | Maximum number of tokens per term; longer terms are truncated. Set on each model after loading. Must be > 0. |
| `normalization` | boolean | `True` | L2-normalize output vectors to unit length, so a dot product equals the cosine similarity. |
| `dimensionality_reduction` | boolean | `False` | Apply UMAP to each model's output after encoding. |
| `n_components` | integer | `2` | Number of output dimensions after UMAP. Use 2 or 3 for plots. Must be ≥ 1. |
| `n_neighbors` | integer | `5` | UMAP neighborhood size: smaller values keep local structure, larger values global structure. Must be ≥ 2. UMAP lowers it automatically when an input has fewer terms. |
| `min_dist` | float | `0.2` | UMAP minimum distance between points in the output; smaller values give tighter clusters. Must be ≥ 0. |
| `metric` | string | `"cosine"` | Distance metric UMAP uses on the input vectors. `"cosine"` suits normalized embeddings, `"euclidean"` raw ones. |
| `random_state` | integer or `null` | `42` | UMAP seed for reproducible output. `null` turns determinism off and lets UMAP use several threads. |

All fields are optional; omitted fields fall back to the defaults above. Unknown keys (e.g. typos) are rejected with a `pydantic.ValidationError` when `Embedder` is created.

---

## Notes

- **Paths** — the config path passed to `Embedder` (default `./config/config.yaml`) and the logger's `folder` are resolved relative to the working directory, not to this file.
- **Dimensionality reduction** — each model has its own UMAP reducer, fitted afresh on every `embed()` call, so reduced coordinates are only comparable within one call. Inputs with fewer than `n_components + 2` terms are returned unreduced (UMAP's initialization fails on fewer points), and a message is logged.
