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
    nuclear_penalty_terms = [str(term).upper() for term in rules.get("nuclear_penalty_terms", [])]
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
    frame["nuclear_penalty_flag"] = text.map(lambda value: int(_contains_any(value, nuclear_penalty_terms)))
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
    feature_scores = frame.apply(lambda row: _feature_scores(row, rules), axis=1, result_type="expand")
    for column in feature_scores.columns:
        frame[column] = feature_scores[column]
    frame["automatic_candidate"] = frame.apply(lambda row: int(_automatic_candidate(row, rules)), axis=1)
    frame["candidate_score"] = frame.apply(_candidate_score, axis=1)
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
    """Return the explainable geometry/reconstruction score for a candidate."""
    if int(row.get("automatic_candidate", 0)) != 1:
        return -1
    score_columns = (
        "coverage_score", "fov_score", "kernel_score", "phase_score",
        "thickness_score", "original_score", "primary_score", "axial_score",
        "abdomen_score", "organ_focus_penalty", "nuclear_penalty",
    )
    return sum(int(row.get(column, 0)) for column in score_columns)


def _automatic_candidate(row, rules=None) -> bool:
    """Eligibility excludes unusable acquisitions but does not require naming conventions."""
    min_z_extent = float((rules or {}).get("min_z_extent_mm", 200.0))
    min_num_slices = int((rules or {}).get("min_num_slices", 50))
    try:
        z_extent = float(row.get("z_extent_mm", ""))
    except (TypeError, ValueError):
        z_extent = 0
    try:
        num_slices = float(row.get("num_slices", ""))
    except (TypeError, ValueError):
        num_slices = 0
    return (
        int(row.get("pass_modality", 0)) == 1
        and z_extent >= min_z_extent
        and num_slices >= min_num_slices
        and int(row.get("reject_anatomy", 0)) == 0
        and int(row.get("reject_derived", 0)) == 0
        and int(row.get("reject_short_series", 0)) == 0
    )


def _feature_scores(row, rules):
    text = " ".join(str(row.get(column, "")) for column in ("series_description", "study_description", "body_part_examined")).upper()
    kernel_type = _classify_kernel(row, rules)
    phase_type = _classify_phase(row, rules)
    coverage_score = _score_z_coverage(row.get("z_extent_mm", ""), rules)
    fov_score = _score_reconstruction_diameter(row.get("reconstruction_diameter", ""), rules)
    kernel_score = {"soft": 25, "bone": -50, "neutral": 0}[kernel_type]
    phase_score = {"venous": 30, "noncontrast": 20, "pre": 15, "arterial": 5, "delayed": -30, "unknown": 0}[phase_type]
    thickness = _number(row.get("slice_thickness", ""))
    thickness_score = (
        20 if thickness is not None and 3 <= thickness <= 5 else
        10 if thickness is not None and 2 <= thickness < 3 else
        0 if thickness is not None and 1 <= thickness < 2 else
        -10 if thickness is not None and thickness < 1 else 0
    )
    image_type = str(row.get("image_type", "")).upper()
    original_score = 20 if _contains_any(image_type, ("ORIGINAL",)) else 0
    primary_score = 15 if _contains_any(image_type, ("PRIMARY",)) else 0
    axial_score = 10 if _contains_any(image_type, ("AXIAL",)) else 0
    abdomen_score = 10 if _contains_any(text, rules.get("strong_abdomen_terms", [])) else 5 if _contains_any(text, rules.get("supporting_abdomen_terms", [])) else 0
    organ_focus_penalty = -int(rules.get("organ_focus_penalty", 15)) if _has_organ_focus(text, rules) else 0
    nuclear_penalty = -int(rules.get("nuclear_penalty", 40)) if _contains_any(text, rules.get("nuclear_penalty_terms", [])) else 0
    anatomy_class = _classify_anatomy(row, rules)
    return {
        "kernel_class": kernel_type, "phase_type": phase_type, "anatomy_class": anatomy_class,
        "is_soft_kernel": int(kernel_type == "soft"),
        "is_bone_kernel": int(kernel_type == "bone"), "is_large_fov": int(fov_score > 0),
        "organ_focus_penalty": organ_focus_penalty, "coverage_score": coverage_score,
        "fov_score": fov_score, "kernel_score": kernel_score, "phase_score": phase_score,
        "thickness_score": thickness_score, "original_score": original_score,
        "primary_score": primary_score, "axial_score": axial_score, "abdomen_score": abdomen_score,
        "nuclear_penalty": nuclear_penalty,
    }


def _classify_kernel(row, rules=None):
    value = " ".join(str(row.get(column, "")) for column in ("convolution_kernel", "series_description")).upper()
    rules = rules or {}
    if _contains_any(value, rules.get("bone_kernel_terms", ("BONE", "LUNG", "SHARP", "EDGE", "DETAIL", "B70", "B80"))):
        return "bone"
    if _contains_any(value, rules.get("soft_kernel_terms", ("STD", "STANDARD", "SOFT", "BODY", "B30", "B31", "B35", "B40"))):
        return "soft"
    return "neutral"


def _classify_phase(row, rules=None):
    value = " ".join(str(row.get(column, "")) for column in ("series_description", "study_description")).upper()
    phase_terms = (rules or {}).get("phase_terms", {})
    if _contains_any(value, phase_terms.get("venous", ("PORTAL VENOUS", "PORTALVENOUS", "VENOUS"))):
        return "venous"
    if _contains_any(value, phase_terms.get("delayed", ("DELAYED", "DELAY"))):
        return "delayed"
    if _contains_any(value, phase_terms.get("noncontrast", ("NONCONTRAST", "NON-CONTRAST"))):
        return "noncontrast"
    if _contains_any(value, phase_terms.get("pre", ("PRE", "PRE-CONTRAST", "PRECONTRAST"))):
        return "pre"
    if _contains_any(value, phase_terms.get("arterial", ("ARTERIAL", "ART"))):
        return "arterial"
    return "unknown"


def _score_z_coverage(value, rules=None):
    try:
        extent = float(value)
    except (TypeError, ValueError):
        return 0
    thresholds = (rules or {}).get("coverage_score_thresholds", {500: 60, 400: 40, 300: 20, 200: 10})
    return max((int(score) for threshold, score in thresholds.items() if extent >= float(threshold)), default=0)


def _score_reconstruction_diameter(value, rules=None):
    try:
        diameter = float(value)
    except (TypeError, ValueError):
        return 0
    thresholds = (rules or {}).get("fov_score_thresholds", {350: 40, 300: 20})
    score = max((int(score) for threshold, score in thresholds.items() if diameter >= float(threshold)), default=0)
    if diameter < float((rules or {}).get("small_fov_threshold", 250)):
        score -= int((rules or {}).get("small_fov_penalty", 20))
    return score


def _has_organ_focus(text, rules):
    return _contains_any(text, rules.get("organ_focus_terms", ("LIVER", "HEPATIC", "RENAL", "KIDNEY", "PANCREAS", "ADRENAL", "AORTA", "CTA")))


def _classify_anatomy(row, rules=None):
    text = " ".join(str(row.get(column, "")) for column in ("series_description", "study_description", "body_part_examined")).upper()
    if _contains_any(text, (rules or {}).get("anatomy_exclude_terms", ())):
        return "non_torso"
    if _contains_any(text, (rules or {}).get("strong_abdomen_terms", ())):
        return "torso"
    if _contains_any(text, (rules or {}).get("supporting_abdomen_terms", ())):
        return "torso"
    return "unknown"


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _same_acquisition(best, other, rules):
    """Identify reconstruction variants of the same scan for automatic tie-breaking."""
    if str(best.get("study_instance_uid", "")) != str(other.get("study_instance_uid", "")):
        return False
    best_extent = _number(best.get("z_extent_mm", ""))
    other_extent = _number(other.get("z_extent_mm", ""))
    if best_extent is None or other_extent is None:
        return False
    tolerance = float(rules.get("duplicate_z_tolerance_mm", 10.0))
    return abs(best_extent - other_extent) <= tolerance


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
        eligible["_reconstruction_diameter_sort"] = pd.to_numeric(
            eligible.get("reconstruction_diameter", pd.Series("", index=eligible.index)), errors="coerce")
        eligible = eligible.sort_values(
            ["candidate_score", "z_extent_mm", "num_slices", "_reconstruction_diameter_sort", "_series_number_sort", "_series_uid_sort"],
            ascending=[False, False, False, False, True, True], na_position="last", kind="mergesort")
        if eligible.empty:
            continue
        best = eligible.iloc[0]
        second_score = int(eligible.iloc[1]["candidate_score"]) if len(eligible) > 1 else -1
        score = int(best["candidate_score"])
        single_candidate = len(eligible) == 1
        clear_margin = score - second_score >= min_margin
        duplicate_reconstructions = len(eligible) > 1 and all(
            _same_acquisition(best, row, rules) for _, row in eligible.iloc[1:].iterrows())
        status = "AUTO_PRIMARY" if single_candidate or duplicate_reconstructions or (score >= min_score and clear_margin) else "REVIEW_REQUIRED"
        reason = "only strict abdominal axial candidate" if single_candidate else "best strict abdominal axial candidate"
        if duplicate_reconstructions:
            reason = "best standard reconstruction among duplicate acquisition series"
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
    return any(re.search(r"(?:^|[^A-Z0-9]){}(?:$|[^A-Z0-9])".format(re.escape(str(term).upper())), text) for term in terms)
