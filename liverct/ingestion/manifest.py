"""Build the authoritative selected-series manifest."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def build_manifest(scored_inventory: Path, review_path: Path, output_path: Path, config=None):
    """Merge tiering and human decisions into a validated manifest."""
    import pandas as pd

    if config is None:
        from .config import IngestionConfig
        config = IngestionConfig()
    inventory = pd.read_csv(scored_inventory, sep="\t", dtype=str).fillna("")
    review = pd.read_csv(review_path, sep="\t", dtype=str).fillna("")
    required = {"series_key", "decision"}
    if not required.issubset(review.columns):
        raise ValueError("review.tsv must contain series_key and decision columns")
    invalid_decisions = set(review["decision"]) - {"", "accept", "reject", "defer"}
    if invalid_decisions:
        raise ValueError("Invalid review decisions: {}".format(sorted(invalid_decisions)))
    inventory["series_key"] = inventory.apply(_series_key, axis=1)
    unknown = set(review["series_key"]) - set(inventory["series_key"])
    if unknown:
        raise ValueError("review.tsv contains unknown series_key values: {}".format(sorted(unknown)))
    decisions = review.drop_duplicates("series_key").set_index("series_key")["decision"].to_dict()
    selected = []
    for _, row in inventory.iterrows():
        tier = row.get("tier", "")
        decision = "accept" if tier == "Tier 1" else "reject" if tier == "Tier 4" else decisions.get(row["series_key"], "")
        if tier in ("Tier 2", "Tier 3") and not decision:
            raise ValueError("Missing review decision for {}".format(row["series_key"]))
        if tier in ("Tier 2", "Tier 3") and decision not in ("accept", "reject"):
            raise ValueError("Invalid or incomplete decision for {}: {}".format(row["series_key"], decision))
        if decision != "accept":
            continue
        session_id = _session_id(row, config)
        selected.append({
            "subject_id": _subject_id(row), "session_id": session_id,
            "series_uid": row.get("series_instance_uid", ""),
            "study_instance_uid": row.get("study_instance_uid", ""),
            "series_number": row.get("series_number", ""),
            "series_description": row.get("series_description", ""),
            "study_description": row.get("study_description", ""),
            "source_directory": row.get("source_directory", ""),
            "representative_file": row.get("representative_file", ""),
            "tier": tier, "selection_status": "accepted", "selection_reason": row.get("tier_reason", ""),
            "reviewer": _review_value(review, row["series_key"], "reviewer"),
            "review_timestamp": _review_value(review, row["series_key"], "review_timestamp"),
            "rule_version": row.get("rule_version", ""),
        })
    output = pd.DataFrame(selected)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep="\t", index=False)
    return output


def _series_key(row) -> str:
    return "|".join((str(row.get("subject_folder", "")), str(row.get("study_instance_uid", "")), str(row.get("series_instance_uid", ""))))


def _subject_id(row) -> str:
    value = str(row.get("subject_folder", ""))
    return value[4:] if value.startswith("sub-") else value


def _session_id(row, config) -> str:
    if config.archive.get("session_from") == "study_date":
        date = str(row.get("study_date", ""))
        return date if len(date) == 8 else ""
    return ""


def _review_value(review, key: str, column: str) -> str:
    matches = review[review["series_key"] == key]
    return str(matches.iloc[0].get(column, "")) if not matches.empty else ""
