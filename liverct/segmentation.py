"""
CT segmentation utilities for BIDS datasets.

Handles series selection, HU unit validation, and TotalSegmentator integration.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np

from .stats import compute_task_statistics, save_statistics_json, find_ct_from_source_json

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
    def _save_source_metadata(nifti_file: Path, output_dir: Path) -> None:
        """
        Save metadata about the source CT file used for segmentation.
        
        Creates a source.json file in the output directory that records
        the CT file path, filename, and associated metadata (if available).
        This allows derivatives to be traced back to their source CT series.
        
        Parameters
        ----------
        nifti_file : Path
            Path to the source CT NIfTI file
        output_dir : Path
            Directory where segmentation outputs are saved
        """
        metadata = {
            "source_file": str(nifti_file.absolute()),
            "source_filename": nifti_file.name,
        }
        
        # Try to load BIDS JSON sidecar if it exists
        json_sidecar = nifti_file.with_suffix("").with_suffix(".json")
        if json_sidecar.exists():
            try:
                with open(json_sidecar) as f:
                    bids_metadata = json.load(f)
                
                # Store key metadata fields
                metadata["SeriesDescription"] = bids_metadata.get("SeriesDescription", "")
                metadata["SeriesNumber"] = bids_metadata.get("SeriesNumber", "")
                metadata["AcquisitionTime"] = bids_metadata.get("AcquisitionTime", "")
                metadata["Manufacturer"] = bids_metadata.get("Manufacturer", "")
                metadata["ManufacturersModelName"] = bids_metadata.get("ManufacturersModelName", "")
                
            except Exception as e:
                logger.warning(f"Could not read source CT metadata from {json_sidecar}: {e}")
        
        # Save metadata file
        metadata_file = output_dir / "source.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"  ✓ Source metadata saved to: {metadata_file}")
        except Exception as e:
            logger.warning(f"  ⚠ Could not save source metadata: {e}")

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
        task: str = "total",
        statistics: bool = True,
        license_number: Optional[str] = None,
        device: str = "gpu",
        nr_thr_resamp: int = 1,
        nr_thr_saving: int = 6,
        overwrite: bool = False,
        **kwargs,
    ) -> bool:
        """
        Run TotalSegmentator segmentation on CT image.

        Parameters
        ----------
        nifti_file : Path
            Path to CT NIfTI image
        output_dir : Path
            Directory to save segmentation outputs
        task : str
            TotalSegmentator task/model to use.
            Options: "total" (default), "tissue_types", "liver_segments",
            "abdominal_muscles", etc.
        statistics : bool
            Whether to generate volume and intensity statistics
        license_number : str, optional
            License key for premium models (e.g., tissue_types)
        device : str
            Device to use: "gpu" or "cpu"
        nr_thr_resamp : int
            Number of threads for resampling (lower to reduce memory)
        nr_thr_saving : int
            Number of threads for saving (lower to reduce memory)
        **kwargs
            Additional arguments to pass to totalsegmentator

        Returns
        -------
        bool
            True if segmentation succeeded, False otherwise
        """
        try:
            from totalsegmentator.python_api import totalsegmentator
        except ImportError:
            logger.error(
                "TotalSegmentator not installed. Install with: pip install TotalSegmentator"
            )
            return False

        if nib is None:
            logger.error("nibabel not installed. Install with: pip install nibabel")
            return False

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up output paths
        seg_output = output_dir / task
        seg_output.mkdir(exist_ok=True)

        # Check if output already exists
        if not overwrite:
            stats_file = seg_output / "statistics.json"
            # Check if segmentation outputs exist
            existing_masks = list(seg_output.glob("*.nii*"))
            if existing_masks:
                if statistics and task != "total":
                    try:
                        stats = compute_task_statistics(nifti_file, seg_output, task=task)
                        save_statistics_json(stats, stats_file)
                        logger.info(f"  ✓ Statistics saved to: {stats_file}")
                    except Exception as e:
                        logger.warning(f"  ⚠ Failed to compute statistics for {task}: {e}")
                logger.info(f"  ↷ Output already exists for task: {task} (use overwrite=True to re-run)")
                return True
            if statistics and stats_file.exists():
                logger.info(f"  ↷ Output already exists for task: {task} (use overwrite=True to re-run)")
                return True

        logger.info(f"Running TotalSegmentator task: {task}")
        logger.info(f"  Input: {nifti_file}")
        logger.info(f"  Output: {seg_output}")
        logger.info(f"  Statistics: {statistics}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Threads (resamp/saving): {nr_thr_resamp}/{nr_thr_saving}")

        try:
            # Run TotalSegmentator
            totalsegmentator(
                input=str(nifti_file),
                output=str(seg_output),
                task=task,
                statistics=statistics,
                license_number=license_number,
                device=device,
                nr_thr_resamp=nr_thr_resamp,
                nr_thr_saving=nr_thr_saving,
                **kwargs,
            )

            # Save source CT metadata
            self._save_source_metadata(nifti_file, seg_output)

            # Check if output was created
            if statistics:
                stats_file = seg_output / "statistics.json"
                if task != "total":
                    try:
                        stats = compute_task_statistics(nifti_file, seg_output, task=task)
                        save_statistics_json(stats, stats_file)
                        logger.info(f"  ✓ Statistics saved to: {stats_file}")
                    except Exception as e:
                        logger.warning(f"  ⚠ Failed to compute statistics for {task}: {e}")
                else:
                    if stats_file.exists():
                        logger.info(f"  ✓ Statistics saved to: {stats_file}")
                    else:
                        logger.warning(f"  ⚠ Statistics file not found: {stats_file}")

            logger.info(f"  ✓ Segmentation completed for task: {task}")
            return True

        except Exception as e:
            logger.error(f"  ✗ Segmentation failed for task {task}: {e}")
            return False

    def run_multiple_segmentations(
        self,
        nifti_file: Path,
        output_dir: Path,
        tasks: List[str],
        statistics: bool = True,
        license_number: Optional[str] = None,
        device: str = "gpu",
        nr_thr_resamp: int = 1,
        nr_thr_saving: int = 6,
        overwrite: bool = False,
        **kwargs,
    ) -> Dict[str, bool]:
        """
        Run multiple TotalSegmentator tasks on the same CT image.

        Parameters
        ----------
        nifti_file : Path
            Path to CT NIfTI image
        output_dir : Path
            Directory to save segmentation outputs
        tasks : list of str
            List of TotalSegmentator tasks to run.
            Examples: ["total", "tissue_types", "liver_segments", "abdominal_muscles"]
        statistics : bool
            Whether to generate statistics for each task
        license_number : str, optional
            License key for premium models
        device : str
            Device to use: "gpu" or "cpu"
        nr_thr_resamp : int
            Number of threads for resampling (lower to reduce memory)
        nr_thr_saving : int
            Number of threads for saving (lower to reduce memory)
        **kwargs
            Additional arguments to pass to totalsegmentator

        Returns
        -------
        dict
            Dictionary mapping task names to success/failure (bool)
        """
        results = {}

        for task in tasks:
            logger.info(f"\n--- Running task: {task} ---")
            success = self.run_segmentation(
                nifti_file=nifti_file,
                output_dir=output_dir,
                task=task,
                statistics=statistics,
                license_number=license_number,
                device=device,
                nr_thr_resamp=nr_thr_resamp,
                nr_thr_saving=nr_thr_saving,
                overwrite=overwrite,
                **kwargs,
            )
            results[task] = success

        # Summary
        successful = sum(1 for v in results.values() if v)
        failed = len(results) - successful
        logger.info(f"\n--- Segmentation summary ---")
        logger.info(f"  Successful tasks: {successful}/{len(tasks)}")
        logger.info(f"  Failed tasks: {failed}/{len(tasks)}")

        return results

    def compute_statistics_only(
        self,
        output_dir: Path,
        task: str,
        nifti_file: Optional[Path] = None,
        overwrite: bool = True,
    ) -> bool:
        """
        Compute statistics.json for an existing segmentation task without re-running.

        Parameters
        ----------
        output_dir : Path
            Base derivatives directory containing the task folder
        task : str
            Task name (e.g., "tissue_types", "liver_segments", "total")
        nifti_file : Path, optional
            CT NIfTI file. If None, attempts to read task/source.json
        overwrite : bool
            Whether to overwrite an existing statistics.json (default: True)

        Returns
        -------
        bool
            True if statistics were computed, False otherwise
        """
        if nib is None:
            logger.error("nibabel not installed. Install with: pip install nibabel")
            return False

        task_dir = Path(output_dir) / task
        if not task_dir.exists():
            logger.error(f"Task directory not found: {task_dir}")
            return False

        stats_file = task_dir / "statistics.json"
        if stats_file.exists() and not overwrite:
            logger.info(f"  ↷ Statistics already exist: {stats_file}")
            return True

        if nifti_file is None:
            nifti_file = find_ct_from_source_json(task_dir)
        if nifti_file is None or not Path(nifti_file).exists():
            logger.error("CT file not found. Provide nifti_file or ensure source.json exists.")
            return False

        try:
            stats = compute_task_statistics(nifti_file, task_dir, task=task)
            save_statistics_json(stats, stats_file)
            logger.info(f"  ✓ Statistics saved to: {stats_file}")
            return True
        except Exception as e:
            logger.error(f"  ✗ Failed to compute statistics for {task}: {e}")
            return False
