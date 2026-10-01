"""
BIDS conversion utilities for CT imaging data.

Uses dcm2bids4ct for converting DICOM CT data to BIDS format following
the proposed CT extension: https://bids.neuroimaging.io/extensions/beps/bep_024.html
"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Any, List

logger = logging.getLogger(__name__)


class CTBIDSConverter:
    """Convert DICOM CT data to BIDS format using dcm2bids4ct."""

    def __init__(
        self,
        dcm2bids4ct_path: Optional[str] = None,
        gdcmconv_path: str = "gdcmconv",
    ):
        """
        Initialize the converter.

        Parameters
        ----------
        dcm2bids4ct_path : str, optional
            Path to dcm2bids4ct executable. If None, assumes it's in PATH.
        gdcmconv_path : str
            Path to the GDCM decompression executable.
        """
        self.dcm2bids4ct_path = dcm2bids4ct_path or "dcm2bids4ct"
        self.gdcmconv_path = gdcmconv_path

    def convert(
        self,
        dicom_dir: str,
        bids_root: str,
        subject_id: Optional[str] = None,
        session_id: Optional[str] = None,
        config_file: Optional[str] = None,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Convert DICOM CT data to BIDS format.

        Parameters
        ----------
        dicom_dir : str
            Path to directory containing DICOM files.
        bids_root : str
            Path to root BIDS directory (will be created if it doesn't exist).
        subject_id : str, optional
            Subject identifier (e.g., "001", "sub-001"). If None, attempts to
            derive from DICOM headers (PatientID/PatientName/AccessionNumber).
        session_id : str, optional
            Session identifier (e.g., "01", "ses-01").
        config_file : str, optional
            Path to custom dcm2bids configuration file.
        overwrite : bool
            If False, skip conversion when matching NIfTI and JSON outputs
            already exist. Defaults to True for backward compatibility.
        **kwargs : dict
            Additional arguments to pass to dcm2bids4ct.

        Returns
        -------
        bool
            True if conversion succeeded, False otherwise.

        Raises
        ------
        FileNotFoundError
            If DICOM directory does not exist.
        """
        dicom_path = Path(dicom_dir)
        if not dicom_path.exists():
            raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

        bids_path = Path(bids_root)
        bids_path.mkdir(parents=True, exist_ok=True)

        # Normalize/derive subject identifier
        subject_label = self._normalize_subject_label(subject_id)
        if not subject_label:
            subject_label = self._derive_subject_label(dicom_path)
            if subject_label:
                logger.info(f"Derived subject label from DICOM: {subject_label}")

        if not subject_label:
            raise ValueError("Unable to determine subject label from DICOM headers.")

        subject_dir = f"sub-{subject_label}"

        session_label = None
        if session_id:
            session_label = session_id.replace("ses-", "")

        # Build BIDS output directory: bids_root/sub-<id>/[ses-<id>/]ct
        if session_label:
            output_dir = bids_path / subject_dir / f"ses-{session_label}" / "ct"
            name_prefix = f"sub-{subject_label}_ses-{session_label}_ct_%s"
        else:
            output_dir = bids_path / subject_dir / "ct"
            name_prefix = f"sub-{subject_label}_ct_%s"

        output_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite and self._has_complete_output(output_dir, name_prefix):
            logger.info("Skipping existing conversion: %s", dicom_path)
            return True

        # Build command for dcm2bids4ct (wrapper around dcm2niix)
        # Usage: dcm2bids4ct <input_dir> [dcm2niix_args...]
        cmd = [
            self.dcm2bids4ct_path,
            str(dicom_path),
            "-o",
            str(output_dir),
            "-f",
            name_prefix,
        ]

        # Optional configuration file is currently not used by dcm2bids4ct
        # Retained for future compatibility
        if config_file:
            logger.warning("config_file is provided but dcm2bids4ct does not accept -c; ignoring.")

        extra_args = []
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    extra_args.append(f"--{key}")
            else:
                extra_args.extend([f"--{key}", str(value)])
        cmd.extend(extra_args)

        # Run conversion, then retry with GDCM only when the normal conversion
        # fails.
        logger.info("Running dcm2bids4ct: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("DICOM to BIDS conversion completed successfully")
            if result.stdout:
                logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Initial conversion failed, attempting GDCM decompression fallback: %s",
                e.stderr,
            )
            return self._convert_with_gdcm_fallback(
                dicom_path,
                output_dir,
                name_prefix,
                original_error=e,
                extra_args=extra_args,
            )
        except FileNotFoundError:
            logger.error(
                f"dcm2bids4ct not found. Please install it or provide path to executable."
            )
            return False

    @staticmethod
    def _has_complete_output(output_dir: Path, name_prefix: str) -> bool:
        """Return whether a matching NIfTI has its JSON sidecar."""
        output_prefix = name_prefix.replace("%s", "")
        for nifti_path in output_dir.glob(f"{output_prefix}*.nii.gz"):
            nifti_stem = nifti_path.name[:-len(".nii.gz")]
            if (output_dir / f"{nifti_stem}.json").is_file():
                return True
        return False

    def _convert_with_gdcm_fallback(
        self,
        dicom_path: Path,
        output_dir: Path,
        name_prefix: str,
        original_error: subprocess.CalledProcessError,
        extra_args: List[str],
    ) -> bool:
        """Decompress a failed series with GDCM and retry conversion."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                decompressed_dir = Path(temp_dir)
                source_files = [path for path in dicom_path.rglob("*") if path.is_file()]
                decompressed_count = 0

                for source_file in source_files:
                    relative_path = source_file.relative_to(dicom_path)
                    output_file = decompressed_dir / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [
                            self.gdcmconv_path,
                            "--raw",
                            str(source_file),
                            str(output_file),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    if output_file.is_file():
                        decompressed_count += 1

                if decompressed_count == 0:
                    raise RuntimeError("GDCM decompression produced zero DICOM files.")

                logger.info(
                    "Decompressed %d DICOM files with GDCM", decompressed_count
                )
                retry_cmd = [
                    self.dcm2bids4ct_path,
                    str(decompressed_dir),
                    "-o",
                    str(output_dir),
                    "-f",
                    name_prefix,
                ]
                retry_cmd.extend(extra_args)
                logger.info("Retrying dcm2bids4ct with GDCM output")
                retry_result = subprocess.run(
                    retry_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if retry_result.stdout:
                    logger.debug(retry_result.stdout)
                logger.info("GDCM fallback conversion succeeded")
                return True
        except FileNotFoundError as e:
            if e.filename == self.gdcmconv_path:
                raise RuntimeError(
                    "gdcmconv not found. Install GDCM to enable conversion fallback."
                ) from e
            logger.error("GDCM fallback conversion failed: %s", e)
            raise original_error from e
        except subprocess.CalledProcessError as e:
            logger.error("GDCM fallback conversion failed: %s", e.stderr)
            raise original_error from e

    @staticmethod
    def _normalize_subject_label(subject_id: Optional[str]) -> Optional[str]:
        if not subject_id:
            return None
        label = str(subject_id).replace("sub-", "").strip()
        if not label:
            return None
        return CTBIDSConverter._sanitize_bids_label(label)

    @staticmethod
    def _sanitize_bids_label(value: str) -> str:
        # BIDS labels should be alphanumeric, with optional dashes
        value = value.strip()
        value = re.sub(r"\s+", "", value)
        value = re.sub(r"[^a-zA-Z0-9-]", "", value)
        return value

    @staticmethod
    def _derive_subject_label(dicom_path: Path) -> Optional[str]:
        """Derive subject label from the first readable DICOM file."""
        try:
            import pydicom
        except Exception:
            logger.error("pydicom is required to derive subject label from DICOM headers.")
            return None

        for file_path in dicom_path.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                ds = pydicom.dcmread(
                    str(file_path),
                    stop_before_pixels=True,
                    force=True,
                )
            except Exception:
                continue

            for tag in ("PatientID", "PatientName", "AccessionNumber", "StudyID"):
                value = getattr(ds, tag, None)
                if value:
                    label = CTBIDSConverter._sanitize_bids_label(str(value))
                    if label:
                        return label
        return None


def convert_dicom_directory_to_bids(
    raw_data_dir: Optional[Path] = None,
    bids_root: Optional[Path] = None,
    config_file: Optional[Path] = None,
    dcm2bids4ct_path: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """
    Convert all DICOM series in a sourcedata directory to BIDS format.

    The expected input layout is ``sub-*/[ses-*/]<series-directory>/``.
    Series directories are identified by their contents rather than by a
    required directory name.

    Parameters
    ----------
    raw_data_dir : Path, optional
        Path to the sourcedata directory containing ``sub-*`` directories. If
        omitted, ``bids_root / "sourcedata"`` is used.
    bids_root : Path
        Path where BIDS dataset will be created
    config_file : Path, optional
        Path to custom dcm2bids configuration file
    dcm2bids4ct_path : str, optional
        Path to dcm2bids4ct executable (if not in PATH)
    overwrite : bool
        If True, re-convert all subject/session inputs even if they already
        exist in BIDS. If False (default), skip existing subject/session CT
        outputs.

    Returns
    -------
    dict
        Summary with keys: 'successful', 'failed', 'skipped'
    """
    logger = logging.getLogger(__name__)

    if bids_root is None:
        raise ValueError("bids_root is required")

    bids_path = Path(bids_root)
    source_path = Path(raw_data_dir) if raw_data_dir is not None else bids_path / "sourcedata"
    if not source_path.is_dir():
        raise FileNotFoundError(f"DICOM source directory not found: {source_path}")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Starting DICOM to BIDS conversion")
    logger.info(f"Raw data directory: {source_path}")
    logger.info(f"BIDS root: {bids_path}")
    if not overwrite:
        logger.info("Skipping already-converted subjects (overwrite=False)")
    else:
        logger.info("Re-converting all subjects (overwrite=True)")

    converter = CTBIDSConverter(dcm2bids4ct_path=dcm2bids4ct_path)

    subject_folders = [
        d
        for d in source_path.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ]
    logger.info(f"Found {len(subject_folders)} subject folders")

    results = {"successful": 0, "failed": 0, "skipped": 0}

    for subject_folder in sorted(subject_folders):
        subject_label = CTBIDSConverter._normalize_subject_label(subject_folder.name)
        if not subject_label:
            logger.warning(f"Invalid subject directory name: {subject_folder.name}, skipping...")
            results["skipped"] += 1
            continue

        session_folders = sorted(
            d
            for d in subject_folder.iterdir()
            if d.is_dir() and d.name.startswith("ses-")
        )

        if session_folders:
            conversion_groups = [
                (session_folder.name, session_folder)
                for session_folder in session_folders
            ]
        else:
            conversion_groups = [(None, subject_folder)]

        for session_id, parent_folder in conversion_groups:
            logger.info(
                "Processing subject %s, session %s",
                subject_folder.name,
                session_id or "(no session)",
            )
            series_folders = _find_dicom_series_directories(parent_folder)
            if not series_folders:
                if session_id is None:
                    logger.warning(
                        f"No readable DICOM series found in {subject_folder.name}, skipping..."
                    )
                else:
                    logger.warning(
                        f"No readable DICOM series found in {parent_folder.name}, skipping..."
                    )
                results["skipped"] += 1
                continue

            session_label = session_id.replace("ses-", "") if session_id else None
            if session_label:
                output_dir = (
                    bids_path
                    / f"sub-{subject_label}"
                    / f"ses-{session_label}"
                    / "ct"
                )
            else:
                output_dir = bids_path / f"sub-{subject_label}" / "ct"

            for series_folder in series_folders:
                entity = f"{subject_folder.name}"
                if session_id:
                    entity += f"/{session_id}"
                logger.info(f"Converting: {entity}/{series_folder.name}")

                success = converter.convert(
                    dicom_dir=str(series_folder),
                    bids_root=str(bids_path),
                    subject_id=subject_label,
                    session_id=session_label,
                    config_file=str(config_file) if config_file else None,
                    overwrite=overwrite,
                )

                if success:
                    results["successful"] += 1
                    logger.info(f"Successfully converted {series_folder.name}")
                else:
                    results["failed"] += 1
                    logger.error(f"Failed to convert {series_folder.name}")

    logger.info(f"\n{'='*60}")
    logger.info(
        f"Conversion complete: {results['successful']} successful, "
        f"{results['failed']} failed, {results['skipped']} skipped"
    )
    logger.info(f"BIDS dataset location: {bids_root}")
    logger.info(f"{'='*60}")

    return results


def _find_dicom_series_directories(parent_dir: Path) -> List[Path]:
    """Find immediate child directories containing at least one readable DICOM."""
    return sorted(
        child
        for child in parent_dir.iterdir()
        if child.is_dir()
        and not child.name.startswith("ses-")
        and _contains_readable_dicom(child)
    )


def _contains_readable_dicom(directory: Path) -> bool:
    """Return whether a directory contains at least one readable DICOM file."""
    try:
        import pydicom
    except Exception:
        logger.error("pydicom is required to discover DICOM series directories.")
        return False

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            pydicom.dcmread(str(file_path), stop_before_pixels=True)
            return True
        except Exception:
            continue
    return False


def _try_derive_subject_label(dicom_path: Path) -> Optional[str]:
    """
    Attempt to derive subject label from DICOM headers.
    
    Returns None if derivation fails (so we proceed with conversion anyway).
    """
    try:
        import pydicom
    except Exception:
        return None

    try:
        for file_path in dicom_path.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                # Try PatientID first
                if hasattr(dcm, "PatientID") and dcm.PatientID:
                    label = str(dcm.PatientID).strip()
                    label = CTBIDSConverter._sanitize_bids_label(label)
                    if label:
                        return label
                # Try PatientName
                if hasattr(dcm, "PatientName") and dcm.PatientName:
                    label = str(dcm.PatientName).strip()
                    label = CTBIDSConverter._sanitize_bids_label(label)
                    if label:
                        return label
                # Try AccessionNumber
                if hasattr(dcm, "AccessionNumber") and dcm.AccessionNumber:
                    label = str(dcm.AccessionNumber).strip()
                    label = CTBIDSConverter._sanitize_bids_label(label)
                    if label:
                        return label
                break
            except Exception:
                continue
    except Exception:
        pass
    
    return None
