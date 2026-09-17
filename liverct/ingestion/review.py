"""Study-centric image-based review reports."""

from datetime import datetime, timezone
from html import escape
import logging
from pathlib import Path
import re
from typing import Optional

logger = logging.getLogger(__name__)


def generate_review_reports(scored_inventory: Path, output_dir: Path, config=None) -> Path:
    """Generate one hierarchical HTML report per subject and preserve review.tsv."""
    import pandas as pd

    if config is None:
        from .config import IngestionConfig
        config = IngestionConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "review_assets"
    assets.mkdir(exist_ok=True)
    frame = pd.read_csv(scored_inventory, sep="\t", dtype=str).fillna("")
    frame = _ensure_review_columns(frame)
    frame = _sort_review_rows(frame)
    logger.info("Loading scored inventory for review: %s (%d rows)", scored_inventory, len(frame))
    decision_path = output_dir / "review.tsv"
    _write_review_template(frame, decision_path)
    detailed_review = bool(config.review.get("detailed_review", False))
    subjects = frame.groupby("subject_folder", sort=True, dropna=False)
    for subject, subject_rows in subjects:
        study_sections = []
        thumbnail_count = 0
        for study_key, study_rows in subject_rows.groupby("scan_group_key", sort=False, dropna=False):
            study_rows = _sort_review_rows(study_rows)
            first = study_rows.iloc[0]
            series_rows = []
            for _, row in study_rows.iterrows():
                recommendation = row.get("recommendation", "REJECT")
                should_render = detailed_review or recommendation in ("PRIMARY", "SECONDARY")
                thumbnail = _make_thumbnail(row, assets, config) if should_render else ""
                if thumbnail:
                    thumbnail_count += 1
                image_html = "<img src='{}' alt='Representative slice' height='260'>".format(escape(thumbnail)) if thumbnail else ""
                series_rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    escape(recommendation), escape(str(row.get("series_number", ""))),
                    escape(str(row.get("series_description", ""))), escape(str(row.get("image_type", ""))),
                    escape(str(row.get("num_slices", ""))), escape(str(row.get("z_extent_mm", ""))),
                    escape(str(row.get("slice_thickness", ""))), image_html))
            study_uids = ", ".join(sorted(set(str(value) for value in study_rows["study_instance_uid"] if value)))
            scan_status = ", ".join(sorted(set(str(value) for value in study_rows["candidate_status"] if value)))
            study_sections.append("<section><h2>Study Date: {}</h2><p><strong>Study Description:</strong> {}</p><p><strong>Study Instance UID(s):</strong> {}</p><p><strong>Automatic status:</strong> {}</p><table><thead><tr><th>Recommendation</th><th>Series #</th><th>Description</th><th>Image Type</th><th>Num Slices</th><th>Z Extent (mm)</th><th>Slice Thickness</th><th>Montage</th></tr></thead><tbody>{}</tbody></table></section>".format(
                escape(_display_date(first.get("study_date", ""))), escape(str(first.get("study_description", ""))),
                escape(study_uids or str(study_key)), escape(scan_status), "".join(series_rows)))
        patient_ids = sorted(set(str(value) for value in subject_rows.get("patient_id", [] ) if value))
        report = "<!doctype html><meta charset='utf-8'><title>Subject {0}</title><h1>Subject {0}</h1><dl><dt>Patient ID</dt><dd>{1}</dd><dt>Total Studies</dt><dd>{2}</dd><dt>Study Dates</dt><dd>{3}</dd></dl>{4}".format(
            escape(str(subject)), escape(", ".join(patient_ids)), len(study_sections),
            escape(", ".join(sorted(set(_display_date(value) for value in subject_rows["study_date"] if value)))),
            "\n".join(study_sections) or "<p>No studies.</p>")
        report_subject = str(subject) if str(subject).startswith("sub-") else "sub-{}".format(subject)
        report_path = output_dir / "{}.html".format(_safe_filename(report_subject))
        report_path.write_text(report, encoding="utf-8")
        logger.info("Wrote %s: %d studies, %d series, %d montages", report_path, len(study_sections), len(subject_rows), thumbnail_count)
    logger.info("Review report generation complete: decision file=%s", decision_path)
    return decision_path


def _write_review_template(frame, decision_path: Path) -> None:
    """Write hierarchical review rows while preserving reviewer edits."""
    import pandas as pd

    candidate_columns = [
        "index", "is_data", "series_key", "subject_id", "patient_id", "study_date", "study_description", "study_instance_uid",
        "series_instance_uid", "series_number", "series_description", "image_type",
        "num_slices", "z_extent_mm", "slice_thickness", "recommendation", "candidate_score",
        "candidate_status", "candidate_rank", "is_auto_primary", "candidate_reason",
    ]
    review_columns = ["reviewer_decision", "notes"]
    candidates = frame.copy()
    candidates["is_data"] = "1"
    candidates["series_key"] = candidates.apply(_series_key, axis=1)
    candidates["subject_id"] = candidates["subject_folder"].str.replace(r"^sub-", "", regex=True)
    for column in candidate_columns + review_columns:
        if column not in candidates:
            if column == "reviewer_decision":
                candidates[column] = candidates.get("recommendation", "")
            else:
                candidates[column] = ""

    if decision_path.exists():
        existing = pd.read_csv(decision_path, sep="\t", dtype=str).fillna("")
        if "series_key" not in existing.columns:
            if {"subject_id", "study_instance_uid", "series_instance_uid"}.issubset(existing.columns):
                existing["series_key"] = existing.apply(lambda row: "|".join((str(row["subject_id"]), str(row["study_instance_uid"]), str(row["series_instance_uid"]))), axis=1)
            else:
                raise ValueError("Existing review.tsv must contain series_key or the new identifier columns")
        if "is_data" in existing.columns:
            existing = existing[existing["is_data"].astype(str) != "0"]
        existing = existing[existing["series_key"] != ""].drop_duplicates("series_key").set_index("series_key")
        candidates = candidates.set_index("series_key")
        if "decision" in existing.columns and "reviewer_decision" not in existing.columns:
            existing["reviewer_decision"] = existing["decision"].map({"accept": "PRIMARY", "reject": "REJECT", "defer": ""}).fillna("")
        if "comment" in existing.columns and "notes" not in existing.columns:
            existing["notes"] = existing["comment"]
        for column in review_columns:
            if column in existing.columns:
                values = existing[column].reindex(candidates.index)
                candidates[column] = values.where(values != "", candidates[column]).fillna(candidates[column])
        logger.info(
            "Preserved existing review decisions: %d of %d current series",
            len(candidates.index.intersection(existing.index)), len(candidates),
        )
        candidates = candidates.reset_index()
    else:
        logger.info("Creating study-level review template for %d series", len(candidates))

    candidates = _sort_review_rows(candidates)
    output_rows = []
    previous_date = None
    previous_subject = None
    for _, row in candidates.iterrows():
        subject = str(row.get("subject_id", ""))
        study_date = str(row.get("study_date", ""))
        if output_rows and (subject != previous_subject or study_date != previous_date):
            output_rows.append({column: "" for column in candidate_columns + review_columns})
        output_rows.append(row.to_dict())
        previous_subject = subject
        previous_date = study_date

    output = pd.DataFrame(output_rows, columns=candidate_columns + review_columns)
    output["index"] = range(1, len(output) + 1)
    output["is_data"] = output["series_key"].ne("").astype(int)
    output.to_csv(decision_path, sep="\t", index=False, lineterminator="\n")
    logger.info("Wrote review decision template: %s (%d data rows, %d total rows)", decision_path, len(candidates), len(output))


def _sort_review_rows(frame):
    """Sort review artifacts by subject, date, and numeric series number."""
    import pandas as pd

    sorted_frame = frame.copy()
    sorted_frame["_sort_subject"] = sorted_frame.get("subject_id", sorted_frame.get("subject_folder", "")).astype(str)
    sorted_frame["_sort_date"] = sorted_frame.get("study_date", "").astype(str)
    sorted_frame["_sort_series"] = pd.to_numeric(sorted_frame.get("series_number", ""), errors="coerce")
    sorted_frame = sorted_frame.sort_values(
        ["_sort_subject", "_sort_date", "_sort_series", "series_number", "series_instance_uid"],
        ascending=[True, True, True, True, True],
        na_position="last",
        kind="mergesort",
    )
    return sorted_frame.drop(columns=["_sort_subject", "_sort_date", "_sort_series"])


def _ensure_review_columns(frame):
    """Derive new review fields when reading an older scored inventory."""
    if "study_group_key" not in frame:
        frame["study_group_key"] = frame.apply(
            lambda row: "|".join((str(row.get("subject_folder", "")), str(row.get("study_instance_uid", "") or row.get("study_date", "")))),
            axis=1,
        )
    if "scan_group_key" not in frame:
        frame["scan_group_key"] = frame.apply(
            lambda row: "|".join((str(row.get("subject_folder", "")), str(row.get("study_date", "")))),
            axis=1,
        )
    if "recommendation" not in frame:
        frame["recommendation"] = "REJECT"
        for study_key, indexes in frame.groupby("scan_group_key", sort=False).groups.items():
            candidates = frame.loc[indexes]
            eligible = candidates[candidates.get("tier", "") != "Tier 4"]
            if not eligible.empty:
                frame.loc[eligible.index, "recommendation"] = "SECONDARY"
                frame.loc[eligible.index[0], "recommendation"] = "PRIMARY"
    if "candidate_status" not in frame:
        frame["candidate_status"] = "REVIEW_REQUIRED"
    if "candidate_rank" not in frame:
        frame["candidate_rank"] = ""
    if "is_auto_primary" not in frame:
        frame["is_auto_primary"] = 0
    if "candidate_reason" not in frame:
        frame["candidate_reason"] = ""
    return frame


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value or "subject"


def _display_date(value) -> str:
    value = str(value)
    if len(value) == 8 and value.isdigit():
        return "{}-{}-{}".format(value[:4], value[4:6], value[6:])
    return value


def _series_key(row) -> str:
    return "|".join((str(row.get("subject_folder", "")), str(row.get("study_instance_uid", "")), str(row.get("series_instance_uid", ""))))


def _make_thumbnail(row, assets: Path, config) -> str:
    try:
        import numpy as np
        import pydicom
        from PIL import Image, ImageDraw

        assets.mkdir(parents=True, exist_ok=True)
        source_dirs = [Path(item) for item in str(row.get("source_directory", "")).split(";") if item]
        series_uid = str(row.get("series_instance_uid", ""))
        # Keep only paths during discovery. Holding decoded DICOM datasets for
        # every candidate slice can retain substantial metadata and pixel state.
        slices = []
        for source_dir in source_dirs:
            for file_path in source_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    dataset = pydicom.dcmread(str(file_path), force=True)
                    if str(dataset.get("SeriesInstanceUID", "")) != series_uid or not hasattr(dataset, "pixel_array"):
                        continue
                    z_position = float(dataset.ImagePositionPatient[2])
                except Exception:
                    continue
                slices.append((z_position, file_path))
        if not slices:
            return ""
        slices.sort(key=lambda item: item[0])
        count = max(1, int(config.review.get("thumbnail_count", 6)))
        selected = [slices[index] for index in np.linspace(0, len(slices) - 1, min(count, len(slices))).astype(int)]
        display_height = 240
        max_width = max(1, int(config.review.get("max_montage_width", 4096)))
        estimated_width = 0
        for _, file_path in selected:
            dataset = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
            rows = int(dataset.get("Rows", 1))
            columns = int(dataset.get("Columns", 1))
            estimated_width += max(1, round(columns * display_height / rows))
        if estimated_width > max_width:
            display_height = max(1, int(display_height * max_width / estimated_width))
            logger.warning(
                "Scaled review montage before rendering: series_uid=%s estimated_width=%d height=%d",
                series_uid, estimated_width, display_height,
            )
        images = []
        for z_position, file_path in selected:
            dataset = pydicom.dcmread(str(file_path), force=True)
            pixels = _display_pixels(dataset)
            image = Image.fromarray(pixels).convert("RGB")
            # Keep a consistent image height while preserving each slice's aspect ratio.
            display_width = max(1, round(image.width * display_height / image.height))
            image = image.resize((display_width, display_height))
            canvas = Image.new("RGB", (display_width, 260), "white")
            canvas.paste(image, (0, 0))
            ImageDraw.Draw(canvas).text((8, 242), "z={:.1f}".format(z_position), fill="black")
            images.append(canvas)
        montage_height = display_height + 20
        montage = Image.new("RGB", (sum(image.width for image in images), montage_height), "white")
        x_offset = 0
        for image in images:
            montage.paste(image, (x_offset, 0))
            x_offset += image.width
        name = "{}.png".format(str(row.get("series_instance_uid", "unknown")).replace(".", "_"))
        montage.save(assets / name)
        return "review_assets/{}".format(name)
    except Exception:
        logger.exception("Failed to generate review montage for series_uid=%s", row.get("series_instance_uid", ""))
        return ""


def _display_pixels(dataset):
    """Convert a DICOM slice into an anatomy-focused 8-bit review image."""
    import numpy as np

    stored_pixels = dataset.pixel_array.astype(np.float32)

    # CT vendors store detector values that must be converted to HU before display.
    slope = _dicom_number(dataset, "RescaleSlope", default=1.0)
    intercept = _dicom_number(dataset, "RescaleIntercept", default=0.0)
    hu_pixels = stored_pixels * slope + intercept

    center = _dicom_number(dataset, "WindowCenter")
    width = _dicom_number(dataset, "WindowWidth")
    if center is not None and width is not None and width > 0:
        lower = center - width / 2.0
        upper = center + width / 2.0
        # Apply the DICOM display window when it is usable for this slice.
        clipped = np.clip(hu_pixels, lower, upper)
        if np.mean(clipped == lower) < 0.98 and np.mean(clipped == upper) < 0.98:
            return ((clipped - lower) / width * 255.0).clip(0, 255).astype("uint8")

    # Missing or unhelpful DICOM windows use robust percentiles instead of min/max,
    # preventing a few extreme voxels from washing out the anatomy.
    finite_pixels = hu_pixels[np.isfinite(hu_pixels)]
    if finite_pixels.size == 0:
        return np.zeros(hu_pixels.shape, dtype="uint8")
    lower, upper = np.percentile(finite_pixels, [1, 99])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.min(finite_pixels))
        upper = float(np.max(finite_pixels))
    if upper <= lower:
        return np.zeros(hu_pixels.shape, dtype="uint8")
    return ((np.clip(hu_pixels, lower, upper) - lower) / (upper - lower) * 255.0).astype("uint8")


def _dicom_number(dataset, keyword, default=None):
    """Read a scalar DICOM numeric value, including the first MultiValue item."""
    try:
        value = dataset.get(keyword, None)
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return default
