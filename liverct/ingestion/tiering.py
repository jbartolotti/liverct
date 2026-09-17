"""Configurable, explainable CT series tiering."""

import logging
from pathlib import Path
import re
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
    strong_abdomen_terms = [str(term).upper() for term in rules.get("strong_abdomen_terms", abdomen_terms)]
    supporting_abdomen_terms = [str(term).upper() for term in rules.get("supporting_abdomen_terms", [])]
    anatomy_exclude_terms = [str(term).upper() for term in rules.get("anatomy_exclude_terms", [])]
    derived_exclude_terms = [str(term).upper() for term in rules.get("derived_exclude_terms", reject_terms)]
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
    body_part = frame.get("body_part_examined", pd.Series("", index=frame.index)).str.upper()
    text = series + " " + study + " " + body_part
    z_extent = pd.to_numeric(frame.get("z_extent_mm", ""), errors="coerce")
    num_slices = pd.to_numeric(frame.get("num_slices", ""), errors="coerce")

    frame["pass_modality"] = (modality == str(rules.get("required_modality", "CT")).upper()).astype(int)
    frame["pass_original"] = image_type.str.contains("ORIGINAL", regex=False).astype(int)
    frame["pass_primary"] = image_type.str.contains("PRIMARY", regex=False).astype(int)
    frame["pass_axial"] = image_type.str.contains("AXIAL", regex=False).astype(int)
    frame["pass_torso_description"] = text.map(lambda value: int(_contains_any(value, abdomen_terms)))
    frame["strong_abdomen_evidence"] = text.map(lambda value: int(_contains_any(value, strong_abdomen_terms)))
    frame["supporting_abdomen_evidence"] = text.map(lambda value: int(_contains_any(value, supporting_abdomen_terms)))
    frame["reject_anatomy"] = text.map(lambda value: int(_contains_any(value, anatomy_exclude_terms)))
    frame["reject_derived"] = (image_type + " " + text).map(lambda value: int(_contains_any(value, derived_exclude_terms)))
    frame["reject_description"] = frame["reject_derived"]
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
    frame["scan_group_key"] = frame.apply(_scan_group_key, axis=1)
    frame["candidate_score"] = frame.apply(_candidate_score, axis=1)
    frame["automatic_candidate"] = frame.apply(_automatic_candidate, axis=1).astype(int)
    frame["recommendation"], frame["candidate_status"], frame["candidate_rank"], frame["is_auto_primary"], frame["candidate_reason"] = _recommendations(frame, rules)
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


def _scan_group_key(row) -> str:
    return "|".join((str(row.get("subject_folder", "")), str(row.get("study_date", ""))))


def _candidate_score(row) -> int:
    """Rank strict automatic candidates with an explainable score."""
    if not _automatic_candidate(row):
        return -1
    score = 0
    score += int(row.get("pass_original", 0)) * 30
    score += int(row.get("pass_primary", 0)) * 25
    score += int(row.get("pass_axial", 0)) * 25
    score += int(row.get("strong_abdomen_evidence", 0)) * 30
    score += int(row.get("supporting_abdomen_evidence", 0)) * 10
    score += int(row.get("pass_z_extent", 0)) * 15
    try:
        z_extent = float(row.get("z_extent_mm", ""))
    except (TypeError, ValueError):
        z_extent = 0
    if z_extent >= 300:
        score += 10
    try:
        num_slices = float(row.get("num_slices", ""))
    except (TypeError, ValueError):
        num_slices = 0
    if num_slices >= 100:
        score += 15
    try:
        thickness = float(row.get("slice_thickness", ""))
    except (TypeError, ValueError):
        thickness = 0
    if 0 < thickness <= 3:
        score += 10
    return score


def _automatic_candidate(row) -> bool:
    return all(int(row.get(name, 0)) == 1 for name in (
        "pass_modality", "pass_original", "pass_primary", "pass_axial",
        "strong_abdomen_evidence", "pass_z_extent", "pass_num_slices",
    )) and int(row.get("reject_anatomy", 0)) == 0 and int(row.get("reject_derived", 0)) == 0 and int(row.get("reject_short_series", 0)) == 0


def _recommendations(frame, rules):
    import pandas as pd

    recommendations = ["REJECT"] * len(frame)
    statuses = ["NO_CANDIDATE"] * len(frame)
    ranks = [""] * len(frame)
    is_primary = [0] * len(frame)
    reasons = ["no candidate passed automatic eligibility gates"] * len(frame)
    min_score = int(rules.get("auto_min_score", 110))
    min_margin = int(rules.get("auto_min_margin", 10))
    for _, scan_rows in frame.groupby("scan_group_key", sort=False):
        eligible = scan_rows[scan_rows["automatic_candidate"] == 1].copy()
        eligible["_series_number_sort"] = pd.to_numeric(eligible.get("series_number", pd.Series("", index=eligible.index)), errors="coerce")
        eligible["_series_uid_sort"] = eligible.get("series_instance_uid", pd.Series("", index=eligible.index)).astype(str)
        eligible = eligible.sort_values(
            ["candidate_score", "z_extent_mm", "num_slices", "_series_number_sort", "_series_uid_sort"],
            ascending=[False, False, False, True, True], na_position="last", kind="mergesort")
        if eligible.empty:
            continue
        best = eligible.iloc[0]
        second_score = int(eligible.iloc[1]["candidate_score"]) if len(eligible) > 1 else -1
        score = int(best["candidate_score"])
        status = "AUTO_PRIMARY" if score >= min_score and (len(eligible) == 1 or score - second_score >= min_margin) else "REVIEW_REQUIRED"
        reason = "best strict abdominal axial candidate"
        if status == "REVIEW_REQUIRED":
            reason = "candidate score or separation from runner-up is below automatic threshold"
        for index in scan_rows.index:
            statuses[frame.index.get_loc(index)] = status
        for rank, (index, row) in enumerate(eligible.iterrows(), start=1):
            position = frame.index.get_loc(index)
            ranks[position] = str(rank)
            statuses[position] = status
            reasons[position] = reason
            recommendations[position] = "PRIMARY" if index == best.name else "SECONDARY"
            if status == "AUTO_PRIMARY" and index == best.name:
                is_primary[position] = 1
    return recommendations, statuses, ranks, is_primary, reasons


def _contains_any(value, terms) -> bool:
    text = str(value).upper()
    return any(re.search(r"(?:^|[^A-Z0-9]){}(?:$|[^A-Z0-9])".format(re.escape(term)), text) for term in terms)
