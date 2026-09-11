"""Shared constants and models for archive ingestion."""

from dataclasses import dataclass
from typing import Optional

TIERS = ("Tier 1", "Tier 2", "Tier 3", "Tier 4")
DECISIONS = ("accept", "reject", "defer")


@dataclass(frozen=True)
class SeriesKey:
    """Stable identity for a discovered series."""

    subject_folder: str
    study_instance_uid: str
    series_instance_uid: str

    @property
    def value(self) -> str:
        return "|".join(
            (self.subject_folder, self.study_instance_uid, self.series_instance_uid)
        )


@dataclass(frozen=True)
class ReviewDecision:
    """A human decision associated with one stable series key."""

    series_key: str
    decision: str
    reviewer: str = ""
    review_timestamp: str = ""
    comment: str = ""


@dataclass(frozen=True)
class ManifestRecord:
    """Selected series provenance used for sourcedata staging."""

    subject_id: str
    session_id: Optional[str]
    series_uid: str
    source_directory: str
    series_number: str = ""
    series_description: str = ""
    study_description: str = ""
    study_instance_uid: str = ""
