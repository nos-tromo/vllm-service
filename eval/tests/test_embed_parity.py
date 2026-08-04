"""CPU-vs-vLLM parity test for the embed-only server (issue #75).

Asserts a *live* embed backend against golden fixtures captured from the
full-stack CUDA vLLM ``BgeM3EmbeddingModel`` backend, covering the four
comparisons issue #75 enumerates: token ids, per-token sparse weights,
special-token arity (vLLM's ``BOSEOSFilter`` drops BOS/EOS positions),
and dense vectors (CLS pooling + L2, compared by cosine AND unit norm —
cosine alone would hide a missing normalisation).

The target is any server speaking the three routes — normally the CPU
``embed-only`` container, but pointing it at the vLLM backend itself is a
valid self-check. Configure via environment:

- ``EMBED_PARITY_BASE_URL`` — e.g. ``http://localhost:8007``. Unset:
  the module is skipped (there is no server to test).
- ``EMBED_PARITY_API_KEY`` — optional bearer token (needed only when
  targeting the vLLM backend directly).

Also skipped when the fixture file is absent, per the issue's
deliverable spec. Tolerances absorb the fp16(vLLM)-vs-float32(CPU)
dtype gap: measured worst-case sparse drift was 3.3e-3 abs on a
531-token multilingual text, dense cosine >= 0.999998.
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "embed_parity" / "vllm_golden.json"
BASE_URL = os.environ.get("EMBED_PARITY_BASE_URL", "").rstrip("/")

SPARSE_ABS_TOLERANCE = 5e-3
DENSE_MIN_COSINE = 0.9999
DENSE_ABS_TOLERANCE = 1e-3
NORM_TOLERANCE = 1e-3

if not BASE_URL:
    pytest.skip(
        "set EMBED_PARITY_BASE_URL to a live embed backend (e.g. http://localhost:8007) to run the parity suite",
        allow_module_level=True,
    )
if not FIXTURE_PATH.is_file():
    pytest.skip(f"parity fixture absent: {FIXTURE_PATH}", allow_module_level=True)

import httpx  # noqa: E402

FIXTURE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text())
CASES: list[dict[str, Any]] = FIXTURE["cases"]
CASE_IDS = [case["name"] for case in CASES]


@pytest.fixture(scope="module")
def client() -> Iterator[httpx.Client]:
    """One connection-reusing client for the whole module."""
    key = os.environ.get("EMBED_PARITY_API_KEY", "")
    headers: dict[str, str] = {"Authorization": f"Bearer {key}"} if key else {}
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=600.0) as live:
        yield live


def _rows_by_index(items: list[dict[str, Any]], key: str) -> list[list[float]]:
    """Order response rows by their declared index, not arrival order."""
    return [item[key] for item in sorted(items, key=lambda item: item["index"])]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_tokenize_ids_match_golden(client: httpx.Client, case: dict[str, Any]) -> None:
    """Comparison 1: identical token ids (and therefore counts) per text."""
    for text, golden_ids in zip(case["texts"], case["tokenize"], strict=True):
        response = client.post("/tokenize", json={"prompt": text})
        response.raise_for_status()
        assert response.json()["tokens"] == golden_ids


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_sparse_weights_match_golden(client: httpx.Client, case: dict[str, Any]) -> None:
    """Comparisons 2+3: element-wise sparse parity at BOS/EOS-stripped arity.

    The length assertion is the special-token comparison: golden rows
    carry len(tokenize) - 2 entries because vLLM drops the BOS and EOS
    positions server-side, and those positions do NOT relu to zero — a
    server that keeps them fails here on arity before values.
    """
    response = client.post("/pooling", json={"task": "token_classify", "input": case["texts"]})
    response.raise_for_status()
    rows = _rows_by_index(response.json()["data"], "data")

    for row, golden_row, golden_ids in zip(rows, case["sparse"], case["tokenize"], strict=True):
        assert len(row) == len(golden_ids) - 2
        assert len(row) == len(golden_row)
        assert all(abs(a - b) <= SPARSE_ABS_TOLERANCE for a, b in zip(row, golden_row, strict=True))


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_dense_vectors_match_golden(client: httpx.Client, case: dict[str, Any]) -> None:
    """Comparison 4: dense cosine ~1.0 AND unit length on both sides.

    Unit length is asserted separately because cosine normalises
    internally and would mask a dropped L2 step; element-wise tolerance
    additionally pins the pooling method (mean pooling instead of CLS
    still produces plausible unit vectors, but not these ones).
    """
    response = client.post("/v1/embeddings", json={"model": FIXTURE["meta"]["model"], "input": case["texts"]})
    response.raise_for_status()
    vectors = _rows_by_index(response.json()["data"], "embedding")

    for vector, golden_vector in zip(vectors, case["dense"], strict=True):
        assert len(vector) == len(golden_vector)
        norm = sum(component * component for component in vector) ** 0.5
        assert abs(norm - 1.0) <= NORM_TOLERANCE
        cosine = sum(a * b for a, b in zip(vector, golden_vector, strict=True))
        assert cosine >= DENSE_MIN_COSINE
        assert all(abs(a - b) <= DENSE_ABS_TOLERANCE for a, b in zip(vector, golden_vector, strict=True))
