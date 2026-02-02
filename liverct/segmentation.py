"""
CT segmentation utilities for BIDS datasets.

Handles series selection, HU unit validation, and TotalSegmentator integration.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import nibabel as nib
except ImportError:
    nib = None


class CTSegmentationPipeline:
    """Pipeline for segmenting CT scans in BIDS format."""

    def __init__(self):
        """Initialize the segmentation pipeline."""
        pass

    @staticmethod
    def find_ct_series(
        bids_root: Path,
        subject_label: str,
        session_label: Optional[str] = None,
        series_description_pattern: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Find CT series in BIDS directory matching criteria.

        Parameters
        ----------
        bids_root : Path
            Root BIDS directory
        subject_label : str
            Subject label (with or without "sub-" prefix)
        session_label : str, optional
            Session label (with or without "ses-" prefix)
        series_description_pattern : str, optional
            Regex pattern or partial string to match series description.
            Examples: "ABD/PEL.*2.5mm", "STND DLIR"

        Returns
        -------
        Path or None
            Path to the best matching CT image directory, or None if not found.
        """
        subject_label = subject_label.replace("sub-", "")
        session_label = session_label.replace("ses-", "") if session_label else None

        # Build subject directory path
        subject_dir = bids_root / f"sub-{subject_label}"
        if not subject_dir.exists():
            logger.warning(f"Subject directory not found: {subject_dir}")
            return None

        # Build CT directory path
        if session_label:
            ct_base = subject_dir / f"ses-{session_label}" / "ct"
        else:
            ct_base = subject_dir / "ct"

        if not ct_base.exists():
            logger.warning(f"CT directory not found: {ct_base}")
            return None

        # Find NIfTI files with metadata (both .nii and .nii.gz)
        nifti_files = list(ct_base.glob("*.nii.gz")) + list(ct_base.glob("*.nii"))
        if not nifti_files:
            logger.warning(f"No CT images found in {ct_base}")
            return None

        # Filter by series description if pattern provided
        if series_description_pattern:
            matching_files = []
            for nii_file in nifti_files:
                json_file = nii_file.with_suffix("").with_suffix(".json")
                if not json_file.exists():
                    continue

                try:
                    with open(json_file) as f:
                        metadata = json.load(f)
                    series_desc = metadata.get("SeriesDescription", "")

                    # Try regex match first, then substring match
                    if re.search(
                        series_description_pattern, series_desc, re.IGNORECASE
                    ):
                        matching_files.append((nii_file, series_desc))
                except Exception as e:
                    logger.warning(f"Failed to read {json_file}: {e}")
                    continue

            if matching_files:
                # Return the first match (or could implement priority logic)
                best_file, series_desc = matching_files[0]
                logger.info(
                    f"Found matching CT series: {best_file.name} "
                    f"(Description: {series_desc})"
                )
                return best_file
            else:
                logger.warning(
                    f"No CT series matching pattern '{series_description_pattern}' "
                    f"found in {ct_base}"
                )
                return None
        else:
            # Return first available if no pattern specified
            logger.info(f"Using first available CT series: {nifti_files[0].name}")
            return nifti_files[0]

    @staticmethod
    def is_in_hounsfield_units(
        nifti_file: Path, json_metadata: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """
        Check if CT image is in Hounsfield units (HU).

        Hounsfield units typically range from -1024 (air) to ~3000+ (bone).
        Checks both metadata and actual voxel values.

        Parameters
        ----------
        nifti_file : Path
            Path to NIfTI CT image
        json_metadata : dict, optional
            Parsed JSON metadata. If None, will attempt to load from .json sidecar.

        Returns
        -------
        is_hu : bool
            True if likely in HU, False otherwise
        reason : str
            Explanation of determination
        """
        if json_metadata is None:
            json_file = nifti_file.with_suffix("").with_suffix(".json")
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        json_metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read JSON metadata: {e}")
                    json_metadata = {}
            else:
                json_metadata = {}

        # Check metadata for rescaling indicators
        rescale_slope = json_metadata.get("RescaleSlope", 1.0)
        rescale_intercept = json_metadata.get("RescaleIntercept", -1024.0)

        if rescale_slope == 1.0 and rescale_intercept == -1024.0:
            return True, "Metadata indicates standard HU conversion (slope=1, intercept=-1024)"

        # Check voxel value ranges
        if nib is None:
            return (
                True,
                "Assuming HU (cannot verify without nibabel, but standard BIDS conversion uses HU)",
            )

        try:
            img = nib.load(nifti_file)
            data = img.get_fdata()

            min_val = float(np.min(data))
            max_val = float(np.max(data))
            mean_val = float(np.mean(data))

            # Typical HU ranges
            # Air: ~-1000, Water: 0, Fat: -100 to -50, Soft tissue: 20-40, Bone: 200+
            if -1100 <= min_val <= -900 and 100 <= max_val <= 4000:
                return (
                    True,
                    f"Voxel range suggests HU (min={min_val:.0f}, max={max_val:.0f}, mean={mean_val:.0f})",
                )

            # If min is near -1024, also likely HU
            if min_val <= -1000:
                return (
                    True,
                    f"Voxel minimum near -1024 suggests HU (min={min_val:.0f}, max={max_val:.0f})",
                )

            # If in 0-4095 range, might be 12-bit unsigned (not HU)
            if min_val >= 0 and max_val <= 4095:
                return (
                    False,
                    f"Voxel range suggests raw 12-bit values, not HU (min={min_val:.0f}, max={max_val:.0f})",
                )

            return (
                True,
                f"Voxel range is ambiguous but assuming HU (min={min_val:.0f}, max={max_val:.0f})",
            )

        except Exception as e:
            logger.warning(f"Could not load NIfTI to check voxel values: {e}")
            return True, "Assuming HU (could not verify voxel range)"

    def run_segmentation(
        self,
        nifti_file: Path,
        output_dir: Path,
        model: str = "3d_fullres",
    ) -> bool:
        """
        Run TotalSegmentator segmentation on CT image.

        Parameters
        ----------
        nifti_file : Path
            Path to CT NIfTI image
        output_dir : Path
            Directory to save segmentation outputs
        model : str
            TotalSegmentator model to use (placeholder)

        Returns
        -------
        bool
            True if segmentation succeeded, False otherwise
        """
        logger.info(f"[PLACEHOLDER] Would run TotalSegmentator with model={model}")
        logger.info(f"  Input: {nifti_file}")
        logger.info(f"  Output: {output_dir}")
        return True
