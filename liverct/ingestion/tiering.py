"""Configurable, explainable CT series tiering."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def score_inventory(input_path: Path, output_path: Optional[Path] = None, config=None):
    """Assign legacy tiers plus study-level primary/secondary recommendations."""
    import pandas as pd

    if config is None:
        from .config import IngestionConfig
        config = IngestionConfig()
    frame = pd.read_csv(input_path, sep="\t", dtype=str).fillna("")
    logger.info("Loading inventory for scoring: %s (%d rows)", input_path, len(frame))
    rules = config.tiering
    abdomen_terms = [str(term).upper() for term in rules.get("abdomen_terms", [])]
    reject_terms = [str(term).upper() for term in rules.get("reject_terms", [])]
    abdomen_pattern = "|".join(abdomen_terms)
    reject_pattern = "|".join(reject_terms)
    logger.info(
        "Scoring rules: required_modality=%s min_z_extent_mm=%s min_num_slices=%s",
        rules.get("required_modality", "CT"),
        rules.get("min_z_extent_mm", 200.0),
        rules.get("min_num_slices", 50),
    )

    modality = frame.get("modality", "").str.upper()
    image_type = frame.get("image_type", "").str.upper()
    series = frame.get("series_description", "").str.upper()
    study = frame.get("study_description", "").str.upper()
    text = series + " " + study
    z_extent = pd.to_numeric(frame.get("z_extent_mm", ""), errors="coerce")
    num_slices = pd.to_numeric(frame.get("num_slices", ""), errors="coerce")

    frame["pass_modality"] = (modality == str(rules.get("required_modality", "CT")).upper()).astype(int)
    frame["pass_original"] = image_type.str.contains("ORIGINAL", regex=False).astype(int)
    frame["pass_primary"] = image_type.str.contains("PRIMARY", regex=False).astype(int)
    frame["pass_axial"] = image_type.str.contains("AXIAL", regex=False).astype(int)
    frame["pass_torso_description"] = text.str.contains(abdomen_pattern, regex=True, na=False).astype(int) if abdomen_pattern else 0
    frame["reject_description"] = text.str.contains(reject_pattern, regex=True, na=False).astype(int) if reject_pattern else 0
    frame["pass_z_extent"] = (z_extent >= float(rules.get("min_z_extent_mm", 200.0))).fillna(False).astype(int)
    frame["pass_num_slices"] = (num_slices >= int(rules.get("min_num_slices", 50))).fillna(False).astype(int)
    frame["reject_short_series"] = num_slices.isin([1, 2]).astype(int)

    tiers = []
    reasons = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        if values["reject_short_series"] == 1:
            tier, reason = "Tier 4", "series contains only 1 or 2 slices"
        elif values["pass_modality"] == 0 or values["reject_description"] == 1:
            tier, reason = "Tier 4", "non-CT modality or explicit reject description"
        elif all(values[name] == 1 for name in ("pass_original", "pass_primary", "pass_axial", "pass_torso_description", "pass_z_extent", "pass_num_slices")):
            tier, reason = "Tier 1", "original primary axial torso series with sufficient coverage"
        elif values["pass_torso_description"] and values["pass_z_extent"]:
            tier, reason = "Tier 2", "likely torso series; requires visual review"
        else:
            tier, reason = "Tier 3", "CT series does not meet definite keep criteria; requires review"
        tiers.append(tier)
        reasons.append(reason)
    frame["tier"] = tiers
    frame["tier_reason"] = reasons
    frame["study_group_key"] = frame.apply(_study_group_key, axis=1)
    frame["candidate_score"] = frame.apply(_candidate_score, axis=1)
    frame["recommendation"] = _recommendations(frame)
    frame["rule_version"] = str(config.values.get("config_version", "1"))
    tier_counts = frame["tier"].value_counts().to_dict()
    logger.info(
        "Scoring complete: Tier 1=%d Tier 2=%d Tier 3=%d Tier 4=%d",
        tier_counts.get("Tier 1", 0), tier_counts.get("Tier 2", 0),
        tier_counts.get("Tier 3", 0), tier_counts.get("Tier 4", 0),
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, sep="\t", index=False)
        logger.info("Wrote scored inventory: %s", output_path)
    return frame


def _study_group_key(row) -> str:
    subject = str(row.get("subject_folder", ""))
    study_uid = str(row.get("study_instance_uid", ""))
    study_date = str(row.get("study_date", ""))
    return "|".join((subject, study_uid or study_date))


def _candidate_score(row) -> int:
    """Rank likely primary acquisitions without changing the tier rules."""
    if str(row.get("tier", "")) == "Tier 4":
        return -1000
    score = 0
    score += int(row.get("pass_modality", 0)) * 100
    score += int(row.get("pass_torso_description", 0)) * 40
    score += int(row.get("pass_original", 0)) * 20
    score += int(row.get("pass_primary", 0)) * 15
    score += int(row.get("pass_axial", 0)) * 15
    score += int(row.get("pass_z_extent", 0)) * 10
    score += int(row.get("pass_num_slices", 0)) * 10
    return score


def _recommendations(frame):
    recommendations = ["REJECT"] * len(frame)
    eligible = frame[
        (frame["tier"] != "Tier 4")
        & (frame["pass_modality"] == 1)
        & (frame["reject_description"] == 0)
    ]
    for _, study_rows in eligible.groupby("study_group_key", sort=False):
        primary_index = study_rows["candidate_score"].idxmax()
        for index in study_rows.index:
            recommendations[frame.index.get_loc(index)] = "PRIMARY" if index == primary_index else "SECONDARY"
    return recommendations
