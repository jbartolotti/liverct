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
    if not decision_path.exists():
        pd.DataFrame(columns=["series_key", "series_instance_uid", "decision", "reviewer", "review_timestamp", "comment"]).to_csv(decision_path, sep="\t", index=False)
        logger.info("Created review decision template: %s", decision_path)
    else:
        logger.info("Preserving existing review decisions: %s", decision_path)

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
            image_html = "<img src='{}' alt='Representative slice' width='240'>".format(escape(thumbnail)) if thumbnail else "<p>No thumbnail available</p>"
            html_rows.append("<article><h2>{}</h2>{}<dl>{}</dl></article>".format(
                escape(key), image_html, "".join("<dt>{}</dt><dd>{}</dd>".format(escape(str(k)), escape(str(row.get(k, "")))) for k in ("study_description", "series_description", "modality", "num_slices", "z_extent_mm", "source_directory", "tier_reason"))))
        report = "<!doctype html><meta charset='utf-8'><title>{}</title><h1>{}</h1>{}".format(tier, tier, "\n".join(html_rows) or "<p>No series in this tier.</p>")
        report_path = output_dir / ("review_tier2.html" if tier == "Tier 2" else "review_tier3.html")
        report_path.write_text(report, encoding="utf-8")
        logger.info("Wrote %s: %d rows, %d montages", report_path, len(rows), thumbnail_count)
    logger.info("Review report generation complete: decision file=%s", decision_path)
    return decision_path


def _series_key(row) -> str:
    return "|".join((str(row.get("subject_folder", "")), str(row.get("study_instance_uid", "")), str(row.get("series_instance_uid", ""))))


def _make_thumbnail(row, assets: Path, config) -> str:
    try:
        import numpy as np
        import pydicom
        from PIL import Image, ImageDraw

        source_dirs = [Path(item) for item in str(row.get("source_directory", "")).split(";") if item]
        series_uid = str(row.get("series_instance_uid", ""))
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
                slices.append((z_position, dataset.pixel_array.astype(float)))
        if not slices:
            return ""
        slices.sort(key=lambda item: item[0])
        count = max(1, int(config.review.get("thumbnail_count", 6)))
        selected = [slices[index] for index in np.linspace(0, len(slices) - 1, min(count, len(slices))).astype(int)]
        low = float(config.review.get("window_min", -200))
        high = float(config.review.get("window_max", 300))
        images = []
        for z_position, pixels in selected:
            pixels = np.clip(pixels, low, high)
            pixels = ((pixels - low) / (high - low) * 255).astype("uint8")
            image = Image.fromarray(pixels).convert("RGB")
            image.thumbnail((240, 240))
            canvas = Image.new("RGB", (240, 260), "white")
            canvas.paste(image, ((240 - image.width) // 2, 0))
            ImageDraw.Draw(canvas).text((8, 242), "z={:.1f}".format(z_position), fill="black")
            images.append(canvas)
        montage = Image.new("RGB", (240 * len(images), 260), "white")
        for index, image in enumerate(images):
            montage.paste(image, (index * 240, 0))
        name = "{}.png".format(str(row.get("series_instance_uid", "unknown")).replace(".", "_"))
        montage.save(assets / name)
        return "review_assets/{}".format(name)
    except Exception:
        return ""
