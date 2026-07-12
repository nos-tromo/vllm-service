"""Build the pyannote diarization pipeline, optionally overriding hyperparameters.

Single home for pipeline construction, shared by the server and the eval
harness so both run byte-identical pipelines. pyannote (and the ``diarize_compat``
shim it needs) are imported lazily inside ``build_pipeline`` so importing this
module — e.g. for the pure ``_resolve_param_overrides`` unit tests — never pulls
in torch.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch-free at runtime; only for type-checkers
    from pyannote.audio import Pipeline

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


def _resolve_param_overrides(
    defaults: dict[str, Any],
    *,
    clustering_threshold: float | None,
    segmentation_min_duration_off: float | None,
    fa: float | None = None,
    fb: float | None = None,
) -> dict[str, Any] | None:
    """Merge requested hyperparameter overrides onto the pipeline's defaults.

    ``Fa``/``Fb`` are community-1's PLDA-clustering weights and its only effective
    speaker-granularity knob — lowering ``Fb`` yields more speakers (``threshold``
    is inert for its clustering). They live under ``clustering`` alongside
    ``threshold``; 3.1's centroid clustering has no such keys, so requesting them
    against a 3.1 pipeline raises (below).

    Args:
        defaults: The pipeline's instantiated parameter tree.
        clustering_threshold: New clustering threshold, or None to leave it.
        segmentation_min_duration_off: New segmentation value, or None.
        fa: New clustering ``Fa`` weight, or None to leave it.
        fb: New clustering ``Fb`` weight, or None to leave it.

    Returns:
        A new merged parameter dict, or None when no override was requested (the
        caller then skips ``instantiate`` entirely, preserving defaults exactly).

    Raises:
        ValueError: If an override targets a key absent from ``defaults``.
    """
    if clustering_threshold is None and segmentation_min_duration_off is None and fa is None and fb is None:
        return None
    merged = copy.deepcopy(defaults)
    if clustering_threshold is not None:
        if "threshold" not in merged.get("clustering", {}):
            raise ValueError(f"loaded pipeline has no clustering.threshold to override; keys={sorted(merged)}")
        merged["clustering"]["threshold"] = clustering_threshold
    if fa is not None:
        if "Fa" not in merged.get("clustering", {}):
            raise ValueError(f"loaded pipeline has no clustering.Fa to override; keys={sorted(merged)}")
        merged["clustering"]["Fa"] = fa
    if fb is not None:
        if "Fb" not in merged.get("clustering", {}):
            raise ValueError(f"loaded pipeline has no clustering.Fb to override; keys={sorted(merged)}")
        merged["clustering"]["Fb"] = fb
    if segmentation_min_duration_off is not None:
        if "min_duration_off" not in merged.get("segmentation", {}):
            raise ValueError(f"loaded pipeline has no segmentation.min_duration_off to override; keys={sorted(merged)}")
        merged["segmentation"]["min_duration_off"] = segmentation_min_duration_off
    return merged


def build_pipeline(
    *,
    model_id: str | None = None,
    device: str | None = None,
    clustering_threshold: float | None = None,
    segmentation_min_duration_off: float | None = None,
    fa: float | None = None,
    fb: float | None = None,
) -> Pipeline:
    """Load the diarization pipeline, applying overrides only when given.

    With no overrides this is byte-for-byte the server's historical construction
    (stock model from ``DIARIZE_MODEL``, no ``instantiate`` call).

    Args:
        model_id: pyannote pipeline id; None → ``DIARIZE_MODEL`` env or 3.1.
        device: Torch device; None → ``DIARIZE_DEVICE`` env or ``cuda``.
        clustering_threshold: Clustering-threshold override, or None.
        segmentation_min_duration_off: Segmentation override, or None.
        fa: Clustering ``Fa`` (PLDA) override, or None.
        fb: Clustering ``Fb`` (PLDA) override, or None.

    Returns:
        The instantiated pyannote ``Pipeline`` moved to ``device``.

    Raises:
        RuntimeError: If ``from_pretrained`` returns None (gated-repo misconfig).
    """
    # Shim-before-pyannote order matters (see diarize_compat's module docstring);
    # fenced so ruff cannot hoist the pyannote import above the shim import.
    # isort: off
    import diarize_compat  # noqa: F401 — restore torchaudio shims BEFORE importing pyannote
    import torch
    from pyannote.audio import Pipeline
    # isort: on

    resolved_model = model_id or os.environ.get("DIARIZE_MODEL", DEFAULT_MODEL)
    resolved_device = device or os.environ.get("DIARIZE_DEVICE", "cuda")
    token = os.environ.get("HF_TOKEN") or None
    try:
        # pyannote.audio 4.x renamed the auth kwarg to `token`; 3.x uses `use_auth_token`.
        pipeline = Pipeline.from_pretrained(resolved_model, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(resolved_model, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            f"Pipeline.from_pretrained({resolved_model!r}) returned None — gated-repo access "
            "missing? Accept the conditions for the model and its segmentation dependency "
            "on the Hugging Face Hub, then run once with HF_HUB_OFFLINE=0, "
            "TRANSFORMERS_OFFLINE=0, and HF_TOKEN set."
        )
    overrides = _resolve_param_overrides(
        pipeline.parameters(instantiated=True),
        clustering_threshold=clustering_threshold,
        segmentation_min_duration_off=segmentation_min_duration_off,
        fa=fa,
        fb=fb,
    )
    if overrides is not None:
        pipeline.instantiate(overrides)
    pipeline.to(torch.device(resolved_device))
    return pipeline
