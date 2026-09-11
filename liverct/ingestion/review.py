"""Static image-based review reports for ambiguous series."""

from datetime import datetime, timezone
from html import escape
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_review_reports(scored_inventory: Path, output_dir: Path, config=None) -> Path:
    """Generate Tier 2/Tier 3 HTML reports and preserve review.tsv."""
    import pandas as pd

    if config is None:
        from .config import IngestionConfig
        config = IngestionConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "review_assets"
    assets.mkdir(exist_ok=True)
    frame = pd.read_csv(scored_inventory, sep="\t", dtype=str).fillna("")
    logger.info("Loading scored inventory for review: %s (%d rows)", scored_inventory, len(frame))
    decision_path = output_dir / "review.tsv"
    _write_review_template(frame, decision_path)

    for tier in ("Tier 2", "Tier 3"):
        rows = frame[frame["tier"] == tier]
        logger.info("Generating %s report: %d series", tier, len(rows))
        html_rows = []
        thumbnail_count = 0
        for _, row in rows.iterrows():
            key = _series_key(row)
            thumbnail = _make_thumbnail(row, assets, config)
            if thumbnail:
                thumbnail_count += 1
            image_html = "<img src='{}' alt='Representative slice' height='260'>".format(escape(thumbnail)) if thumbnail else "<p>No thumbnail available</p>"
            html_rows.append("<article><h2>{}</h2>{}<dl>{}</dl></article>".format(
                escape(key), image_html, "".join("<dt>{}</dt><dd>{}</dd>".format(escape(str(k)), escape(str(row.get(k, "")))) for k in ("study_description", "series_description", "modality", "num_slices", "z_extent_mm", "source_directory", "tier_reason"))))
        report = "<!doctype html><meta charset='utf-8'><title>{}</title><h1>{}</h1>{}".format(tier, tier, "\n".join(html_rows) or "<p>No series in this tier.</p>")
        report_path = output_dir / ("review_tier2.html" if tier == "Tier 2" else "review_tier3.html")
        report_path.write_text(report, encoding="utf-8")
        logger.info("Wrote %s: %d rows, %d montages", report_path, len(rows), thumbnail_count)
    logger.info("Review report generation complete: decision file=%s", decision_path)
    return decision_path


def _write_review_template(frame, decision_path: Path) -> None:
    """Write candidate identifiers while preserving existing review fields."""
    import pandas as pd

    candidate_columns = [
        "series_key", "series_instance_uid", "subject_folder", "study_instance_uid",
        "study_date", "study_description", "series_number", "series_description",
        "modality", "image_type", "num_slices", "z_extent_mm", "source_directory",
        "tier", "tier_reason",
    ]
    review_columns = ["decision", "reviewer", "review_timestamp", "comment"]
    candidates = frame[frame["tier"].isin(("Tier 2", "Tier 3"))].copy()
    candidates["series_key"] = candidates.apply(_series_key, axis=1)
    for column in candidate_columns + review_columns:
        if column not in candidates:
            candidates[column] = ""

    if decision_path.exists():
        existing = pd.read_csv(decision_path, sep="\t", dtype=str).fillna("")
        if "series_key" not in existing.columns:
            raise ValueError("Existing review.tsv must contain a series_key column")
        existing = existing.drop_duplicates("series_key").set_index("series_key")
        candidates = candidates.set_index("series_key")
        for column in review_columns:
            if column in existing.columns:
                candidates[column] = existing[column].reindex(candidates.index).fillna("")
        logger.info(
            "Preserved existing review decisions: %d of %d current candidates",
            len(candidates.index.intersection(existing.index)), len(candidates),
        )
        candidates = candidates.reset_index()
    else:
        logger.info("Creating review decision template for %d candidates", len(candidates))

    candidates[candidate_columns + review_columns].to_csv(decision_path, sep="\t", index=False)
    logger.info("Wrote review decision template: %s (%d rows)", decision_path, len(candidates))


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
