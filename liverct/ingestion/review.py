"""Study-centric image-based review reports."""

from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from html import escape
import logging
from pathlib import Path
import re
import time
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
    montage_rows = []
    for _, patient_rows in frame.groupby("patient_id", sort=True, dropna=False):
        for _, study_rows in patient_rows.groupby("scan_group_key", sort=False, dropna=False):
            for _, row in _human_review_rows(_sort_review_rows(study_rows)).iterrows():
                recommendation = row.get("recommendation", "REJECT")
                if row.get("series_key", "") and (detailed_review or recommendation in ("PRIMARY", "SECONDARY")):
                    montage_rows.append(row.to_dict())
    montage_paths = _generate_montages(montage_rows, assets, config)
    patients = frame.groupby("patient_id", sort=True, dropna=False)
    for patient_id, patient_rows in patients:
        study_sections = []
        thumbnail_count = 0
        for study_key, study_rows in patient_rows.groupby("scan_group_key", sort=False, dropna=False):
            study_rows = _sort_review_rows(study_rows)
            first = study_rows.iloc[0]
            review_rows = _human_review_rows(study_rows)
            series_rows = []
            for _, row in review_rows.iterrows():
                recommendation = row.get("recommendation", "REJECT")
                if not row.get("series_key", ""):
                    continue
                thumbnail = montage_paths.get(_series_key(row), "")
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
            if not series_rows:
                series_rows.append("<tr><td colspan='8'>{}</td></tr>".format(escape(str(review_rows.iloc[0].get("candidate_reason", "No candidate sequence available")))))
            study_sections.append("<section><h2>Study Date: {}</h2><p><strong>Study Description:</strong> {}</p><p><strong>Study Instance UID(s):</strong> {}</p><p><strong>Automatic status:</strong> {}</p><table><thead><tr><th>Recommendation</th><th>Series #</th><th>Description</th><th>Image Type</th><th>Num Slices</th><th>Z Extent (mm)</th><th>Slice Thickness</th><th>Montage</th></tr></thead><tbody>{}</tbody></table></section>".format(
                escape(_display_date(first.get("study_date", ""))), escape(str(first.get("study_description", ""))),
                escape(study_uids or str(study_key)), escape(scan_status), "".join(series_rows)))
        patient_label = str(patient_id) or "unknown"
        report = "<!doctype html><meta charset='utf-8'><title>Subject {0}</title><h1>Subject {0}</h1><dl><dt>Patient ID</dt><dd>{0}</dd><dt>Total Studies</dt><dd>{1}</dd><dt>Study Dates</dt><dd>{2}</dd></dl>{3}".format(
            escape(patient_label), len(study_sections),
            escape(", ".join(sorted(set(_display_date(value) for value in patient_rows["study_date"] if value)))),
            "\n".join(study_sections) or "<p>No studies.</p>")
        report_subject = patient_label if patient_label.startswith("sub-") else "sub-{}".format(patient_label)
        report_path = output_dir / "{}.html".format(_safe_filename(report_subject))
        report_path.write_text(report, encoding="utf-8")
        logger.info("Wrote %s: %d studies, %d series, %d montages", report_path, len(study_sections), len(patient_rows), thumbnail_count)
    logger.info("Review report generation complete: decision file=%s", decision_path)
    return decision_path


def _write_review_template(frame, decision_path: Path) -> None:
    """Write hierarchical review rows while preserving reviewer edits."""
    import pandas as pd

    candidate_columns = [
        "index", "is_data", "review_row_type", "series_key", "subject_id", "patient_id", "study_date", "study_description", "study_instance_uid",
        "series_instance_uid", "series_number", "series_description", "image_type",
        "num_slices", "z_extent_mm", "slice_thickness", "recommendation", "candidate_score",
        "candidate_status", "candidate_rank", "is_auto_primary", "candidate_reason",
    ]
    review_columns = ["reviewer_decision", "notes"]
    candidates = pd.concat([_human_review_rows(group) for _, group in frame.groupby("scan_group_key", sort=False, dropna=False)], ignore_index=True)
    candidates = candidates[candidates["candidate_status"] == "REVIEW_REQUIRED"].copy()
    candidates["is_data"] = "1"
    candidates["is_data"] = candidates["series_key"].ne("").astype(int)
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
            separator = {column: "" for column in candidate_columns + review_columns}
            separator["review_row_type"] = "SEPARATOR"
            output_rows.append(separator)
        output_rows.append(row.to_dict())
        previous_subject = subject
        previous_date = study_date

    output = pd.DataFrame(output_rows, columns=candidate_columns + review_columns)
    output["index"] = range(1, len(output) + 1)
    output["is_data"] = output["series_key"].ne("").astype(int)
    output.to_csv(decision_path, sep="\t", index=False, lineterminator="\n")
    logger.info("Wrote review decision template: %s (%d data rows, %d total rows)", decision_path, len(candidates), len(output))


def _human_review_rows(scan_rows):
    """Return the compact rows intended for human review for one scan date."""
    import pandas as pd

    scan_rows = scan_rows.copy()
    status = str(scan_rows.iloc[0].get("candidate_status", "NO_CANDIDATE")) if not scan_rows.empty else "NO_CANDIDATE"
    if status == "NO_CANDIDATE":
        row = {column: "" for column in scan_rows.columns}
        first = scan_rows.iloc[0] if not scan_rows.empty else row
        row.update({
            "review_row_type": "STATUS", "series_key": "", "subject_folder": first.get("subject_folder", ""),
            "subject_id": str(first.get("subject_folder", "")).replace("sub-", "", 1),
            "patient_id": first.get("patient_id", ""), "study_date": first.get("study_date", ""),
            "study_description": first.get("study_description", ""), "study_instance_uid": "",
            "candidate_status": "NO_CANDIDATE", "candidate_reason": "No eligible abdominal CT candidate sequence for this subject/date",
            "recommendation": "REJECT", "reviewer_decision": "REJECT",
        })
        return pd.DataFrame([row])

    automatic_candidate = scan_rows.get("automatic_candidate", pd.Series(0, index=scan_rows.index)).astype(str)
    eligible = scan_rows[automatic_candidate == "1"].copy()
    if status == "AUTO_PRIMARY":
        auto_primary = eligible.get("is_auto_primary", pd.Series(0, index=eligible.index)).astype(str)
        eligible = eligible[auto_primary == "1"]
    eligible = _sort_review_rows(eligible)
    eligible["review_row_type"] = "SERIES"
    eligible["series_key"] = eligible.apply(_series_key, axis=1)
    eligible["subject_id"] = eligible["subject_folder"].str.replace(r"^sub-", "", regex=True)
    eligible["reviewer_decision"] = eligible["recommendation"]
    return eligible


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
    if "patient_id" not in frame:
        frame["patient_id"] = ""
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
    if "automatic_candidate" not in frame:
        frame["automatic_candidate"] = (frame["recommendation"] != "REJECT").astype(int)
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


def _generate_montages(rows, assets: Path, config):
    """Build required montages from one metadata index, reusing cached PNGs."""
    started = time.perf_counter()
    assets.mkdir(parents=True, exist_ok=True)
    pending = []
    results = {}
    for row in rows:
        key = _series_key(row)
        relative_path, montage_file = _montage_paths(row, assets)
        if montage_file.exists():
            results[key] = relative_path
            logger.info("Reusing existing montage: %s", montage_file)
            continue
        pending.append((key, row, relative_path, montage_file))

    if pending:
        index = _build_slice_index(pending)
        review_values = _review_values(config)
        workers = max(1, int(review_values.get("montage_workers", 1)))
        tasks = []
        for key, row, relative_path, montage_file in pending:
            tasks.append((row, assets, review_values, index.get(key, [])))
        logger.info("Generating %d review montages with %d worker(s)", len(tasks), workers)
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                generated = list(executor.map(_make_thumbnail_worker, tasks))
        else:
            generated = [_make_thumbnail_worker(task) for task in tasks]
        for (key, _, relative_path, _), generated_path in zip(pending, generated):
            if generated_path:
                results[key] = generated_path
            else:
                logger.warning("Montage generation failed or found no readable slices: series_key=%s", key)
    logger.info("Montage build completed in %.2f seconds (workers=%d, requested=%d, reused=%d)", time.perf_counter() - started, max(1, int(_review_values(config).get("montage_workers", 1))), len(rows), len(rows) - len(pending))
    return results


def _make_thumbnail_worker(task):
    row, assets, review_values, slices = task
    return _make_thumbnail(row, Path(assets), review_values, slices=slices)


def _build_slice_index(pending):
    """Index series slice metadata once per source directory without pixels."""
    import pydicom

    source_dirs = sorted({str(source_dir) for _, row, _, _ in pending for source_dir in _source_dirs(row)})
    index = {}
    for source_dir in source_dirs:
        for file_path in Path(source_dir).rglob("*"):
            if not file_path.is_file():
                continue
            try:
                dataset = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
                series_uid = str(dataset.get("SeriesInstanceUID", ""))
                if not series_uid:
                    continue
                z_position = _slice_position(dataset)
                if z_position is None:
                    continue
                rows = int(dataset.get("Rows", 1))
                columns = int(dataset.get("Columns", 1))
            except Exception:
                continue
            index.setdefault(series_uid, []).append((z_position, str(file_path), rows, columns))
    for slices in index.values():
        slices.sort(key=lambda item: (item[0], item[1]))
    return {key: [item for item in index.get(str(row.get("series_instance_uid", "")), []) if Path(item[1]).parent in _source_dirs(row)] for key, row, _, _ in pending}


def _source_dirs(row):
    return [Path(item) for item in str(row.get("source_directory", "")).split(";") if item]


def _slice_position(dataset):
    try:
        return float(dataset.ImagePositionPatient[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        try:
            return float(dataset.get("InstanceNumber", ""))
        except (TypeError, ValueError):
            return None


def _montage_paths(row, assets):
    name = "{}.png".format(str(row.get("series_instance_uid", "unknown")).replace(".", "_"))
    montage_file = assets / name
    return "review_assets/{}".format(name), montage_file


def _review_values(config):
    return dict(config.review) if hasattr(config, "review") else dict(config)


def _make_thumbnail(row, assets: Path, config, slices=None) -> str:
    try:
        import numpy as np
        import pydicom
        from PIL import Image, ImageDraw

        assets.mkdir(parents=True, exist_ok=True)
        series_uid = str(row.get("series_instance_uid", ""))
        if slices is None:
            slices = _build_slice_index([(series_uid, row, "", "")]).get(series_uid, [])
        if not slices:
            return ""
        count = max(1, int(_review_values(config).get("thumbnail_count", 6)))
        selected = [slices[index] for index in np.linspace(0, len(slices) - 1, min(count, len(slices))).astype(int)]
        display_height = 240
        max_width = max(1, int(_review_values(config).get("max_montage_width", 4096)))
        estimated_width = sum(max(1, round(columns * display_height / rows)) for _, _, rows, columns in selected)
        if estimated_width > max_width:
            display_height = max(1, int(display_height * max_width / estimated_width))
            logger.warning(
                "Scaled review montage before rendering: series_uid=%s estimated_width=%d height=%d",
                series_uid, estimated_width, display_height,
            )
        images = []
        for z_position, file_path, _, _ in selected:
            dataset = pydicom.dcmread(str(file_path), force=True)
            values = _review_values(config)
            pixels = _display_pixels(dataset, values.get("window_min"), values.get("window_max"))
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
        _, montage_file = _montage_paths(row, assets)
        montage.save(montage_file)
        logger.info("Generated montage: %s", montage_file)
        return _montage_paths(row, assets)[0]
    except Exception:
        logger.exception("Failed to generate review montage for series_uid=%s", row.get("series_instance_uid", ""))
        return ""


def _display_pixels(dataset, window_min=None, window_max=None):
    """Convert a DICOM slice into an anatomy-focused 8-bit review image."""
    import numpy as np

    stored_pixels = dataset.pixel_array.astype(np.float32)

    # CT vendors store detector values that must be converted to HU before display.
    slope = _dicom_number(dataset, "RescaleSlope", default=1.0)
    intercept = _dicom_number(dataset, "RescaleIntercept", default=0.0)
    hu_pixels = stored_pixels * slope + intercept

    if window_min is not None and window_max is not None:
        try:
            lower = float(window_min)
            upper = float(window_max)
        except (TypeError, ValueError):
            lower, upper = None, None
        if lower is not None and upper is not None and upper > lower:
            return ((np.clip(hu_pixels, lower, upper) - lower) / (upper - lower) * 255.0).clip(0, 255).astype("uint8")

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
