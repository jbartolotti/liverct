"""Materialize selected manifest rows as BIDS sourcedata."""

import hashlib
import shutil
from pathlib import Path
from typing import Optional


def stage_sourcedata(manifest_path: Path, archive_root: Path, bids_root: Path, mode: str = "copy") -> Path:
    """Copy, hardlink, or symlink selected DICOM files into sourcedata."""
    import pandas as pd
    import pydicom

    if mode not in ("copy", "hardlink", "symlink"):
        raise ValueError("mode must be copy, hardlink, or symlink")
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    archive_root = Path(archive_root).resolve()
    sourcedata = Path(bids_root) / "sourcedata"
    for _, row in manifest.iterrows():
        subject = str(row["subject_id"])
        session = str(row.get("session_id", ""))
        series_uid = str(row["series_uid"])
        source_text = str(row["source_directory"])
        source_dirs = []
        for item in source_text.split(";"):
            if not item:
                continue
            source_dir = Path(item)
            if not source_dir.is_absolute():
                source_dir = archive_root / source_dir
            source_dirs.append(source_dir)
        destination = sourcedata / "sub-{}".format(subject)
        if session:
            destination /= "ses-{}".format(session.replace("ses-", ""))
        destination /= "series-{}".format(_short_uid(series_uid))
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        for source_dir in source_dirs:
            if not source_dir.is_dir():
                raise FileNotFoundError("Source directory not found: {}".format(source_dir))
            for path in source_dir.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
                except Exception:
                    continue
                if str(dataset.get("SeriesInstanceUID", "")) == series_uid:
                    files.append(path)
        for source_file in sorted(set(files)):
            target = destination / source_file.name
            if target.exists():
                continue
            if mode == "copy":
                shutil.copy2(source_file, target)
            elif mode == "hardlink":
                target.hardlink_to(source_file)
            else:
                target.symlink_to(source_file)
    return sourcedata


def _short_uid(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
