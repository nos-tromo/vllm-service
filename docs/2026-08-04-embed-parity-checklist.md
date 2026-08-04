# embed-only ↔ vLLM parity checklist (issue #75)

How to verify that the CPU `embed-only` server (`src/embed_server.py`)
matches the full-stack CUDA vLLM `embed` backend
(`BgeM3EmbeddingModel`), and what was found when this was first run
(2026-08-04, vLLM 0.20.1, `BAAI/bge-m3`).

## The four comparisons

Same synthetic inputs against both backends (short ASCII, a long
multilingual passage, and a mixed-length batch that exercises per-row
padding), then diff:

1. **Token count / ids** — `POST /tokenize` per text. Ids must match
   exactly. **Result: identical on every case.**
2. **Per-token sparse weights** — `POST /pooling` with
   `task: token_classify`, element-wise within tolerance. **Result:
   match within 3.3e-3 abs worst case** (fp16 under vLLM's
   `dtype=auto` vs float32 on CPU; a handful of near-zero values flip
   across the ReLU boundary on long inputs — all below tolerance).
3. **Special-token positions** — **the one real divergence found.**
   vLLM wraps its `token_classify` pooler in `BOSEOSFilter`
   (`vllm/model_executor/models/roberta.py`), dropping the first
   position iff its id is the BOS id and the last iff its id is the
   EOS id, so every sparse row has `len(tokenize) - 2` entries. The
   CPU server used to return all positions — and the boundary
   positions do **not** ReLU to zero (measured 0.11–0.24 on `<s>`),
   so the consumer's drop-non-positive filter could not remove them.
   FlagEmbedding's explicit special-token exclusion is not dead code.
   The server now applies the same conditional strip
   (`encode_token_weights`), verified element-wise against vLLM.
4. **Dense vector** — `POST /v1/embeddings`. Cosine between backends
   ≥ 0.999998 on every case, **and** both sides unit-length (cosine
   alone would hide a missing L2 step). CLS pooling confirmed —
   element-wise agreement within 2.2e-4 rules out mean pooling.

Rankings of the issue's residual risks against these results: #1
(special tokens) was real and is fixed; #2 (pooling method), #3
(normalisation), and #5 (post-ReLU transform) are cleared — vLLM does
CLS + L2 and no extra sparse transform; #4 (dtype) is the only
remaining source of drift and stays within the tolerances below.

## Fixtures and test

- Goldens: `eval/fixtures/embed_parity/vllm_golden.json` — captured
  from the CUDA backend, all inputs fully synthetic.
- Test: `eval/tests/test_embed_parity.py` — asserts a *live* server
  against the goldens. Skips when the fixture is absent or when no
  target is configured.

Run against a local embed-only container:

```bash
make up-dev-embed-only
EMBED_PARITY_BASE_URL=http://localhost:${EMBED_HOST_PORT:-8007} \
  uv run --group eval pytest eval/tests/test_embed_parity.py
```

Self-check against the vLLM backend itself (from a host on
`inference-net`, or inside the container):

```bash
EMBED_PARITY_BASE_URL=http://embed:8000 \
EMBED_PARITY_API_KEY=$OPENAI_API_KEY \
  uv run --group eval pytest eval/tests/test_embed_parity.py
```

Tolerances (empirical worst case ×~1.5–3 headroom): sparse abs 5e-3,
dense cosine ≥ 0.9999, dense abs 1e-3, unit-norm ±1e-3.

## Re-capturing goldens

Re-capture whenever the vLLM base image, `EMBED_MODEL`, or
`EMBED_HF_OVERRIDES` changes: run the three routes against the CUDA
backend with the fixture's own `texts` (or fresh synthetic ones) and
rewrite `vllm_golden.json`, keeping the `meta` block's version fields
current. Keep every input synthetic — no production or testing data,
per the repo confidentiality rule.
