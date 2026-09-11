"""Archive-agnostic DICOM series inventory."""

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def inventory_archive(
    root_dir: Path,
    output_path: Optional[Path] = None,
    test_mode: bool = False,
):
    """Discover readable DICOM series recursively and return one row per series.

    When ``test_mode`` is true, only the first top-level directory below the
    archive root is searched. This is intended for a quick archive smoke test.
    """
    import pandas as pd
    import pydicom

    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Archive directory not found: {}".format(root))

    top_level_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if test_mode and top_level_dirs:
        search_roots = [top_level_dirs[0]]
        logger.warning(
            "TEST MODE: scanning only top-level directory %s; %d top-level directories remain unscanned",
            search_roots[0].name,
            max(0, len(top_level_dirs) - 1),
        )
    else:
        search_roots = top_level_dirs or [root]
        logger.info(
            "Inventory search root: %s (%d top-level directories)",
            root,
            len(top_level_dirs),
        )

    grouped: Dict[str, Dict[str, Any]] = {}
    positions: Dict[str, List[float]] = defaultdict(list)
    instances: Dict[str, List[int]] = defaultdict(list)
    source_dirs: Dict[str, set] = defaultdict(set)

    file_count = 0
    readable_count = 0
    unreadable_count = 0
    missing_uid_count = 0
    logged_subjects = set()
    for search_root in search_roots:
        logger.info("Scanning archive directory: %s", search_root)
        search_files = sorted(path for path in search_root.rglob("*") if path.is_file())
        logger.info("Found %d files below %s", len(search_files), search_root.name)
        for file_path in search_files:
            file_count += 1
            subject_folder = _subject_folder(root, file_path)
            if subject_folder not in logged_subjects:
                logger.info("Processing subject/top-level folder: %s", subject_folder)
                logged_subjects.add(subject_folder)
            try:
                dataset = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
            except Exception:
                unreadable_count += 1
                continue
            readable_count += 1
            series_uid = _text(dataset, "SeriesInstanceUID")
            if not series_uid:
                missing_uid_count += 1
                continue
            study_uid = _text(dataset, "StudyInstanceUID")
            key = "|".join((subject_folder, study_uid, series_uid))
            if key not in grouped:
                grouped[key] = _series_metadata(dataset, root, file_path, subject_folder, study_uid, series_uid)
                logger.info(
                    "Discovered series: subject=%s study=%s series=%s description=%s",
                    subject_folder,
                    study_uid or "<missing>",
                    series_uid,
                    _text(dataset, "SeriesDescription") or "<missing>",
                )
            source_dirs[key].add(str(file_path.parent))
            grouped[key]["num_files"] += 1
            instance = _number(dataset, "InstanceNumber")
            if instance is not None:
                instances[key].append(instance)
            position = _image_z(dataset)
            if position is not None:
                positions[key].append(position)

    rows = []
    for key, row in grouped.items():
        instance_values = instances[key]
        z_values = positions[key]
        row["num_slices"] = len(set(z_values)) if z_values else (max(instance_values) - min(instance_values) + 1 if instance_values else "")
        row["z_extent_mm"] = round(max(z_values) - min(z_values), 1) if len(z_values) > 1 else ""
        row["source_directory"] = ";".join(sorted(source_dirs[key]))
        rows.append(row)

    columns = list(_series_metadata(None, root, Path("."), "", "", "").keys()) + ["num_slices", "z_extent_mm"]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["subject_folder", "study_date", "series_number", "series_instance_uid"], na_position="last")
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, sep="\t", index=False)
        logger.info("Wrote inventory: %s", output_path)
    logger.info(
        "Inventory complete: files=%d readable_dicom=%d unreadable=%d missing_series_uid=%d subjects=%d series=%d",
        file_count,
        readable_count,
        unreadable_count,
        missing_uid_count,
        len(logged_subjects),
        len(frame),
    )
    return frame.reset_index(drop=True)


def _series_metadata(dataset, root: Path, file_path: Path, subject_folder: str, study_uid: str, series_uid: str) -> Dict[str, Any]:
    if dataset is None:
        return {name: "" for name in [
            "subject_folder", "patient_id", "patient_name", "study_instance_uid", "series_instance_uid",
            "study_date", "study_description", "series_number", "series_description", "modality",
            "sop_class_uid", "sop_class_name", "image_type", "body_part_examined", "acquisition_date",
            "acquisition_time", "manufacturer", "manufacturer_model_name", "slice_thickness",
            "pixel_spacing", "kvp", "convolution_kernel", "reconstruction_diameter", "patient_position",
            "num_files", "source_directory", "representative_file",
        ]}
    sop_class = _text(dataset, "SOPClassUID")
    try:
        sop_name = dataset.SOPClassUID.name
    except Exception:
        sop_name = ""
    return {
        "subject_folder": subject_folder,
        "patient_id": _text(dataset, "PatientID"),
        "patient_name": _text(dataset, "PatientName"),
        "study_instance_uid": study_uid,
        "series_instance_uid": series_uid,
        "study_date": _text(dataset, "StudyDate"),
        "study_description": _text(dataset, "StudyDescription"),
        "series_number": _text(dataset, "SeriesNumber"),
        "series_description": _text(dataset, "SeriesDescription"),
        "modality": _text(dataset, "Modality"),
        "sop_class_uid": sop_class,
        "sop_class_name": sop_name,
        "image_type": _multi_text(dataset, "ImageType"),
        "body_part_examined": _text(dataset, "BodyPartExamined"),
        "acquisition_date": _text(dataset, "AcquisitionDate"),
        "acquisition_time": _text(dataset, "AcquisitionTime"),
        "manufacturer": _text(dataset, "Manufacturer"),
        "manufacturer_model_name": _text(dataset, "ManufacturerModelName"),
        "slice_thickness": _text(dataset, "SliceThickness"),
        "pixel_spacing": _multi_text(dataset, "PixelSpacing"),
        "kvp": _text(dataset, "KVP"),
        "convolution_kernel": _text(dataset, "ConvolutionKernel"),
        "reconstruction_diameter": _text(dataset, "ReconstructionDiameter"),
        "patient_position": _text(dataset, "PatientPosition"),
        "num_files": 0,
        "source_directory": str(file_path.parent),
        "representative_file": str(file_path),
    }


def _subject_folder(root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(root)
    return relative.parts[0] if relative.parts else ""


def _text(dataset, keyword: str) -> str:
    if dataset is None:
        return ""
    try:
        value = dataset.get(keyword, "")
        return "" if value is None else str(value)
    except Exception:
        return ""


def _multi_text(dataset, keyword: str) -> str:
    if dataset is None:
        return ""
    try:
        value = dataset.get(keyword, "")
        if isinstance(value, (list, tuple)):
            return "\\".join(str(item) for item in value)
        return str(value) if value is not None else ""
    except Exception:
        return ""


def _number(dataset, keyword: str) -> Optional[int]:
    try:
        return int(getattr(dataset, keyword))
    except (AttributeError, TypeError, ValueError):
        return None


def _image_z(dataset) -> Optional[float]:
    try:
        return float(dataset.ImagePositionPatient[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
