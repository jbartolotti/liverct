"""Build the authoritative selected-series manifest."""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_manifest(scored_inventory: Path, review_path: Path, output_path: Path, config=None, include_secondary: bool = False):
    """Merge study-level review decisions into a validated manifest."""
    import pandas as pd

    if config is None:
        from .config import IngestionConfig
        config = IngestionConfig()
    inventory = pd.read_csv(scored_inventory, sep="\t", dtype=str).fillna("")
    review = pd.read_csv(review_path, sep="\t", dtype=str).fillna("")
    if "is_data" in review.columns:
        review = review[review["is_data"].astype(str) == "1"]
    review = review[review.get("series_key", "") != ""]
    logger.info(
        "Building manifest: inventory=%s (%d rows), review=%s (%d rows)",
        scored_inventory, len(inventory), review_path, len(review),
    )
    if "series_key" not in review.columns:
        raise ValueError("review.tsv must contain a series_key column")
    decision_column = "reviewer_decision" if "reviewer_decision" in review.columns else "decision"
    if decision_column not in review.columns:
        raise ValueError("review.tsv must contain reviewer_decision or decision")
    if decision_column == "decision":
        review["reviewer_decision"] = review["decision"].map({"accept": "PRIMARY", "reject": "REJECT", "defer": ""}).fillna("")
        decision_column = "reviewer_decision"
    review[decision_column] = review[decision_column].str.upper()
    invalid_decisions = set(review[decision_column]) - {"", "PRIMARY", "SECONDARY", "REJECT"}
    if invalid_decisions:
        raise ValueError("Invalid reviewer decisions: {}".format(sorted(invalid_decisions)))
    inventory["series_key"] = inventory.apply(_series_key, axis=1)
    unknown = set(review["series_key"]) - set(inventory["series_key"])
    if unknown:
        raise ValueError("review.tsv contains unknown series_key values: {}".format(sorted(unknown)))
    decisions = review.drop_duplicates("series_key").set_index("series_key")[decision_column].to_dict()
    selected = []
    tier_counts = inventory["tier"].value_counts().to_dict()
    decision_counts = {"PRIMARY": 0, "SECONDARY": 0, "REJECT": 0}
    selected_primary = {}
    for _, row in inventory.iterrows():
        tier = row.get("tier", "")
        recommendation = str(row.get("recommendation", "" )).upper()
        decision = decisions.get(row["series_key"], recommendation if recommendation else ("PRIMARY" if tier == "Tier 1" else "REJECT" if tier == "Tier 4" else ""))
        if not decision:
            logger.error("Missing reviewer decision: series_key=%s", row["series_key"])
            raise ValueError("Missing review decision for {}".format(row["series_key"]))
        if decision not in decision_counts:
            raise ValueError("Invalid or incomplete decision for {}: {}".format(row["series_key"], decision))
        decision_counts[decision] += 1
        if decision == "REJECT" or (decision == "SECONDARY" and not include_secondary):
            continue
        study_key = str(row.get("scan_group_key", _scan_group_key(row)))
        if decision == "PRIMARY" and study_key in selected_primary:
            raise ValueError("Multiple PRIMARY series selected for study {}".format(study_key))
        if decision == "PRIMARY":
            selected_primary[study_key] = row["series_key"]
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
            "tier": tier, "recommendation": recommendation, "selection_status": decision.lower(), "selection_reason": row.get("tier_reason", ""),
            "reviewer": _review_value(review, row["series_key"], "reviewer"),
            "review_timestamp": _review_value(review, row["series_key"], "review_timestamp"),
            "rule_version": row.get("rule_version", ""),
        })
    output = pd.DataFrame(selected)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep="\t", index=False)
    logger.info(
        "Manifest complete: Tier 1=%d Tier 2=%d Tier 3=%d Tier 4=%d primary=%d secondary=%d rejected=%d output=%s",
        tier_counts.get("Tier 1", 0), tier_counts.get("Tier 2", 0),
        tier_counts.get("Tier 3", 0), tier_counts.get("Tier 4", 0),
        decision_counts["PRIMARY"], decision_counts["SECONDARY"], decision_counts["REJECT"], output_path,
    )
    return output


def _series_key(row) -> str:
    return "|".join((str(row.get("subject_folder", "")), str(row.get("study_instance_uid", "")), str(row.get("series_instance_uid", ""))))


def _study_group_key(row) -> str:
    return "|".join((str(row.get("subject_folder", "")), str(row.get("study_instance_uid", "") or row.get("study_date", ""))))


def _scan_group_key(row) -> str:
    return "|".join((str(row.get("subject_folder", "")), str(row.get("study_date", ""))))


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
