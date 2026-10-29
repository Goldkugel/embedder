# Embedder Configuration

[`config.yaml`](./config.yaml) contains the two sections `Embedder` reads:

- **`embedder`** — validated against the [`EmbedderConfig`](../src/EmbedderConfig.py) Pydantic model and documented below. Required: a file without it raises a `ValueError`.
- **`logger`** — only read if no other part of the program has configured the [`logger`](https://github.com/Goldkugel/logger) singleton yet. The file sets all of its fields for `logger` v1.1.0, including `encoding`; see the `logger` package's README for their meaning.

When `embedder` is used inside a larger pipeline, its section can also live in a shared file next to other packages' sections; `Embedder` ignores all sections except these two.

---

## `embedder` section

```yaml
embedder:
  model_id:
    - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  revision:
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"
  pooling: "mean"
  cache_folder: "./data/cache/embeddings/"
  progress_interval: 10000
  num_threads: null
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
| `revision` | mapping of model ID → string | `{}` | Hugging Face revision (commit hash, branch or tag) to load per model. A commit hash makes runs reproducible even if the model is updated on the Hub. Models not listed load the latest revision; local paths need no entry. Every key must also appear in `model_id`, so a typo cannot leave a model unpinned. |
| `pooling` | string | `"mean"` | How the token vectors of a term are combined into one embedding: `"mean"`, `"cls"`, `"max"`, `"mean_sqrt_len_tokens"`, `"weightedmean"` or `"lasttoken"`. Applies to every configured model and replaces the pooling the model would otherwise use. `"mean"` matches what sentence-transformers does by default for plain Hugging Face models such as SapBERT; SapBERT itself was trained with `"cls"`. |
| `cache_folder` | string or `null` | `null` | Folder for the embedding cache: one JSON file per model and settings, mapping each term to its embedding. `embed()` only encodes terms not yet in the file. `null` disables caching. |
| `progress_interval` | integer | `10000` | Number of terms encoded between two progress messages. After each message the embeddings so far are saved to the cache (if enabled). Must be > 0. |
| `num_threads` | integer or `null` | `null` | Maximum number of CPU threads PyTorch may use. `null` keeps PyTorch's default (all cores). Process-wide: also affects other PyTorch code in the same process. Must be > 0. |
| `device` | string | `"cpu"` | Hardware target passed to Sentence Transformers: `"cpu"`, `"cuda"` (NVIDIA GPU) or `"mps"` (Apple Silicon). |
| `batch_size` | integer | `128` | Number of terms encoded per forward pass. Larger is faster but needs more memory. Must be > 0. |
| `max_seq_length` | integer | `128` | Maximum number of tokens per term; longer terms are truncated. Set on each model after loading. Must be > 0. |
| `normalization` | boolean | `True` | L2-normalize output vectors to unit length, so a dot product equals the cosine similarity. |
| `dimensionality_reduction` | boolean | `False` | Create one UMAP reducer per model, so `Embedder.reduce()` can project embeddings to `n_components` dimensions. `embed()` always returns the full-dimensional vectors either way. |
| `n_components` | integer | `2` | Number of output dimensions after UMAP. Use 2 or 3 for plots. Must be ≥ 1. |
| `n_neighbors` | integer | `5` | UMAP neighborhood size: smaller values keep local structure, larger values global structure. Must be ≥ 2. UMAP lowers it to (rows − 1) when an input to `reduce()` has fewer rows. |
| `min_dist` | float | `0.2` | UMAP minimum distance between points in the output; smaller values give tighter clusters. Must be ≥ 0. |
| `metric` | string | `"cosine"` | Distance metric UMAP uses on the input vectors. `"cosine"` suits normalized embeddings, `"euclidean"` raw ones. |
| `random_state` | integer or `null` | `42` | UMAP seed for reproducible output. `null` turns determinism off and lets UMAP use several threads. |

All fields are optional; omitted fields fall back to the defaults above. Unknown keys (e.g. typos) are rejected with a `pydantic.ValidationError` when `Embedder` is created.

---

## Notes

- **Paths** — the config path passed to `Embedder` (default `./config/config.yaml`), `cache_folder` and the logger's `folder` are resolved relative to the working directory, not to this file.
- **Dimensionality reduction** — each model has its own UMAP reducer, fitted afresh on every `reduce()` call, so reduced coordinates are only comparable within one call. Inputs with fewer than `n_components + 2` rows are returned unreduced (UMAP's initialization fails on fewer points), and a message is logged.
- **Cache files** — one JSON file per model, named `<model ID>-<hash>.json`, where the hash covers the model ID, its revision, `pooling`, `max_seq_length` and `normalization`; changing any of these starts a new file instead of mixing vectors computed in different ways. The file maps each term to its embedding, one term per line. Pin the model's `revision` when caching: without a pin, an updated model on the Hub would still be served the old cached vectors. The cache is never cleaned up automatically; delete the folder to clear it. `data/cache/` is excluded from Git.
- **Safe saves** — `Embedder.save_cache()` writes to `<file>.tmp` first and then renames it over the cache file. If a run is killed during a save, the cache file still holds the previous complete version; a leftover `.tmp` file is incomplete and is overwritten by the next save.
- **Checkpoints** — each checkpoint rewrites the whole cache file, so the time per save grows with the cache. With very large vocabularies, a larger `progress_interval` means fewer saves (but more lost work if a run is interrupted).
- **Finding a revision** — the current commit hash of a model is shown on its Hugging Face page under "Files and versions", or returned as `sha` by `https://huggingface.co/api/models/<model_id>`.
