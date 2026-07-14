"""Value object describing one diarization configuration under evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DiarizeConfig:
    """One diarization configuration: model, device, hyperparameters, bounds.

    Attributes:
        label: Human-readable name, used as the report row key.
        model_id: pyannote pipeline id; None → the server's env/3.1 default.
        device: Torch device string; None → the server's env/cuda default.
        clustering_threshold: Clustering-threshold override; None → pretrained default.
        segmentation_min_duration_off: Segmentation override; None → pretrained default.
        fa: Clustering ``Fa`` (PLDA) override; None → pretrained default. community-1's
            speaker-granularity knob (with ``fb``); inapplicable to 3.1.
        fb: Clustering ``Fb`` (PLDA) override; None → pretrained default. Lower → more
            speakers.
        num_speakers: Exact speaker count, if forced.
        min_speakers: Lower bound on the speaker count.
        max_speakers: Upper bound on the speaker count.
    """

    label: str
    model_id: str | None = None
    device: str | None = None
    clustering_threshold: float | None = None
    segmentation_min_duration_off: float | None = None
    fa: float | None = None
    fb: float | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    @property
    def pipeline_kwargs(self) -> dict[str, int]:
        """Return only the set speaker-bound kwargs for the pipeline call.

        Returns:
            Mapping of ``num_speakers``/``min_speakers``/``max_speakers`` to
            their values, omitting any that are None.
        """
        pairs = (
            ("num_speakers", self.num_speakers),
            ("min_speakers", self.min_speakers),
            ("max_speakers", self.max_speakers),
        )
        return {name: value for name, value in pairs if value is not None}

    def as_dict(self) -> dict[str, Any]:
        """Return a flat dict of every field for report serialization.

        Returns:
            All dataclass fields as a plain dictionary.
        """
        return asdict(self)
