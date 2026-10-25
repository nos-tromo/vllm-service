# Topics fit service — design

## Purpose

A stateless CPU service that fits a topic model to vectors the caller already
has. A client posts a sample of embedding vectors and the matching texts. It
gets back one topic label per input and a keyword list per topic.

The service stores nothing. The client keeps the topic table, computes topic
centres, and assigns further points itself.

The fit is [BERTopic](https://maartengr.github.io/BERTopic/) (UMAP → HDBSCAN →
c-TF-IDF) with no embedding model: the vectors arrive with the request, so
nothing is embedded or downloaded at runtime.

## Contract

### `POST /topics/fit`

The body is `multipart/form-data`. Every part is sent as a file part:

| Field | Filename | Content |
|---|---|---|
| `meta` | `meta.json` | `{"n", "dim", "seed", "min_topic_size", "nr_topics", "top_n_words", "min_df", "stop_words"}` |
| `texts` | `texts.json` | A JSON array of `n` strings (the c-TF-IDF input) |
| `vectors` | `vectors.f32` | `n × dim` little-endian float32, row-major |

`nr_topics` is `0` (no reduction) or at least `2`; BERTopic counts the outlier class as one of the topics.

`200` response:

```json
{
  "labels": [0, -1, 2, ...],
  "outliers": 123,
  "topics": [{"id": 0, "size": 410, "keywords": [["word", 0.031], ...]}, ...],
  "versions": {"bertopic": "...", "umap-learn": "...", "hdbscan": "...", "scikit-learn": "...", "numpy": "...", "numba": "..."},
  "runtime": {"machine": "...", "python": "...", "threads": {"OPENBLAS_NUM_THREADS": "1", ...}},
  "timings": {"fit_s": 0.0}
}
```

- `labels[i]` is the topic of input `i`; `-1` means outlier.
- `topics` lists the ids `>= 0` in ascending id order.
- A topic's `keywords` are best first: the score rounded to 6 decimals, highest first, and words with equal scores in alphabetical order.

Errors:

| Status | When |
|---|---|
| `400` | A malformed `meta.json` (including a `dim` above 8192 or a stop word over 64 characters), or a multipart body that names no boundary |
| `413` | `n > TOPICS_MAX_POINTS`, `n·dim > TOPICS_MAX_POINTS·1024`, a `meta.json` over 1 MiB, or a `texts` part over `min(n·16384 + 65536, 536870912)` bytes. Each is decided from `meta.json` and the parts' sizes alone, before a large part is read into memory |
| `422` | A missing part (including a body that is not multipart), a `texts` part that is not a JSON array of strings, `len(texts) != n`, a vector byte count other than `n·dim·4`, non-finite values, or fewer points than the minimum, `max(16, 2·min_topic_size)` |
| `503` + `Retry-After` | A fit is already running (one at a time; requests are not queued) |
| `500` | The fit itself failed |

Exact-duplicate vectors are clustered once; copies inherit their representative's label. The fit runs on the distinct vectors in first-occurrence order, each paired with its first occurrence's text. The texts of the copies are not used, for keywords or for `min_df`, while a topic's `size` counts every input row (`sum(size) + outliers == n`). Inputs with fewer distinct vectors than that minimum get a `200`, not a `422`: every label is `-1` and `topics` is empty, and UMAP and HDBSCAN do not run on them.

`TOPICS_MAX_POINTS` defaults to `150000`. Because `n·dim` is capped at `TOPICS_MAX_POINTS·1024`, a wider vector means fewer points, not a larger peak. The limit does not bound memory on its own: a fit's peak grows with the number of points, their dimensions and their text length, and the share of distinct vectors moves it too. Measured on the real server with synthetic data (1024 dimensions, texts of about 2,000 characters, `min_topic_size` 100, 10% exact duplicates), one fit peaked at 2.3 GB with 50k points and 3.9 GB with 100k, about 5.4 GB at 150k extrapolated linearly. Both topics services therefore set `mem_limit: ${TOPICS_MEM_LIMIT:-7g}`, that peak plus 25% rounded up to a whole GiB. A request that outgrows it kills only the topics container, which restarts.

### `GET /health`

Returns `{"status": "ok", "versions": {...}}`. It passes only after the startup
self-check has fitted a synthetic dataset twice and found the results
identical.

## Determinism

The same input bytes, settings, image and CPU class give the same labels and
keywords.

- **Input order:** the client supplies it. The service never sorts or samples.
- **UMAP:** `n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed, n_jobs=1`. A fixed seed makes UMAP single-threaded, which `n_jobs=1` states explicitly.
- **hdbscan:** `metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=1`, with `min_cluster_size = min_topic_size` and `min_samples = min(min_topic_size, 100)`. hdbscan's own default is `min_samples = min_cluster_size`, and its core-distance query holds `min_samples + 1` neighbours of every distinct vector at 16 bytes each, so the cap bounds it at about `n·101·16` bytes. Up to `min_topic_size` 100 the clustering is identical to the default.
- **c-TF-IDF:** `ClassTfidfTransformer(reduce_frequent_words=True)`. A per-text `CountVectorizer(stop_words, min_df)` fixes the vocabulary before the fit, and BERTopic receives it fixed. So `min_df` counts texts, not topics, and the vocabulary cannot come out empty after clustering.
- **Keywords:** the service ranks each topic's row of the fitted c-TF-IDF matrix itself: by the score rounded to 6 decimals, then alphabetically, then cut to `top_n_words`. BERTopic's own lists leave the order of equal scores, and so which tied words survive the cut, to numpy's sort, which differs between platforms.
- **Text:** BERTopic runs with `language=None`, so texts are not stripped to ASCII. When fewer than two words survive `stop_words` and `min_df`, every text is replaced by a two-word sentinel. The clustering is unchanged, and every topic's keywords are empty.
- **Threads:** `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` are all set to 1. A multi-threaded BLAS reduction inside UMAP's spectral initialisation would otherwise make results depend on the host's core count.
- **No randomness in sampling:** BERTopic's representative-document sampling uses its own fixed seed, and the service returns no representative documents.

## Pinning

The service has its own locked project, `docker/topics/{pyproject.toml,uv.lock}`. It is separate from the root project, whose lock resolves the eval environments, so neither world's pins move the other's.

- The image installs the exported lock with `--no-deps --require-hashes`.
- The export uses `--locked`, so a lock that has drifted from `docker/topics/pyproject.toml` fails the build instead of installing stale pins.
- `exclude-dependencies` keeps `sentence-transformers` (and with it torch) and `plotly` out. BERTopic imports neither when no embedding model is configured.
- BERTopic imports `yaml` and `packaging` at module level without declaring them; they normally arrive through sentence-transformers. The project therefore lists `pyyaml` and `packaging` itself.
- `pandas<3` keeps BERTopic 0.17.4 on its tested range; `numpy<2.6` is numba's bound.
- hdbscan has no linux/aarch64 wheel, so it is built in a builder stage, and only the virtualenv is copied into the runtime stage.

## Deployment

Both shapes use one CPU image, as VAD does: `vllm-service-topics` in the full
stack, `vllm-service-topics-cpu` in the `topics-only` shape.

- **Full stack:** the router passes `/topics/fit` through, with `auth: true` and no `headers` block. LiteLLM reads every file part into memory and re-encodes the multipart body with a new boundary; the part names, filenames and content types survive. The global 4 h pass-through timeout covers a fit. The router waits only for the service to *start*, not for it to be healthy, so a failed self-check never keeps inference down.
- **`topics-only`:** alias `topics-only` on `inference-net`, port 8000 (8008 on the host in dev), no auth.
- **Writable paths:** `/tmp` is the disk-backed `media-tmp` volume, because multipart parts spool there. `NUMBA_CACHE_DIR` is `/run/numba`, a tmpfs private to the container (64 MB, uid and gid 10001, mode 0700): numba unpickles its cache's index files, so the cache must not sit on `media-tmp`, which `asr`, `diarize` and `vad` write in the full stack. The service has no model weights and mounts no Hugging Face cache.
- **Privacy:** texts and vectors are never logged; logs carry counts and timings only.
