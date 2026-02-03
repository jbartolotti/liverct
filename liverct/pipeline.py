"""
High-level processing pipelines for CT BIDS datasets.

Coordinates multiple processing steps and handles subject/session iteration.
"""

import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union
from .segmentation import CTSegmentationPipeline
from .figures import create_vertebrae_slice_report

logger = logging.getLogger(__name__)


class BIDSProcessingPipeline:
    """Orchestrate processing of BIDS CT datasets."""

    def __init__(self, bids_root: Path):
        """
        Initialize processing pipeline.

        Parameters
        ----------
        bids_root : Path
            Root directory of BIDS dataset
        """
        self.bids_root = Path(bids_root)
        if not self.bids_root.exists():
            raise FileNotFoundError(f"BIDS root not found: {self.bids_root}")

        self.segmentation = CTSegmentationPipeline()

    def process_all_subjects(
        self,
        process_func: Callable,
        log_summary: bool = True,
        subjects: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Apply processing function to all subjects in BIDS dataset.

        Parameters
        ----------
        process_func : callable
            Function to call for each subject/session.
            Signature: process_func(subject_label, session_label=None) -> bool
        log_summary : bool
            Whether to log summary statistics
        subjects : str or list of str, optional
            Single subject ID or list of subject IDs to process.
            If None, processes all subjects. IDs can be with or without 'sub-' prefix.

        Returns
        -------
        dict
            Summary with keys: 'successful', 'failed', 'skipped'
        """
        subject_dirs = sorted([d for d in self.bids_root.glob("sub-*") if d.is_dir()])
        
        # Filter subjects if specified
        if subjects is not None:
            if isinstance(subjects, str):
                subjects = [subjects]
            # Normalize subject IDs (ensure sub- prefix)
            subjects_normalized = [s if s.startswith('sub-') else f'sub-{s}' for s in subjects]
            subject_dirs = [d for d in subject_dirs if d.name in subjects_normalized]
            if not subject_dirs:
                logger.warning(f"No matching subjects found for: {subjects}")
                return {"successful": 0, "failed": 0, "skipped": 0}
        
        logger.info(f"Found {len(subject_dirs)} subjects")

        results = {"successful": 0, "failed": 0, "skipped": 0}

        for subject_dir in subject_dirs:
            subject_label = subject_dir.name

            # Check for sessions
            session_dirs = sorted([d for d in subject_dir.glob("ses-*") if d.is_dir()])

            if session_dirs:
                # Process each session
                for session_dir in session_dirs:
                    session_label = session_dir.name
                    result = process_func(subject_label, session_label=session_label)
                    results[result] += 1
            else:
                # Process without session
                result = process_func(subject_label, session_label=None)
                results[result] += 1

        if log_summary:
            self._log_summary(results)

        return results

    def segment_ct_series(
        self,
        subject_label: str,
        session_label: Optional[str] = None,
        series_description_pattern: Optional[str] = None,
        tasks: Union[str, List[str]] = "total",
        statistics: bool = True,
        license_number: Optional[str] = None,
        device: str = "gpu",
        nr_thr_resamp: int = 1,
        nr_thr_saving: int = 6,
        overwrite: bool = False,
        output_dir: Optional[Path] = None,
    ) -> bool:
        """
        Find, validate, and segment CT series for a single subject/session.

        Parameters
        ----------
        subject_label : str
            Subject identifier (with or without "sub-" prefix)
        session_label : str, optional
            Session identifier (with or without "ses-" prefix)
        series_description_pattern : str, optional
            Regex pattern to match series description
        tasks : str or list of str
            TotalSegmentator task(s) to run.
            Single task: "total" or
            Multiple tasks: ["total", "tissue_types", "liver_segments", "abdominal_muscles"]
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
        overwrite : bool
            If True, re-run segmentation even if output exists. Default: False
        output_dir : Path, optional
            Where to save segmentation derivatives. If None, uses
            bids_root/derivatives/totalsegmentator/

        Returns
        -------
        bool
            True if all segmentations succeeded, False otherwise
        """
        subject_id = subject_label.replace("sub-", "")
        display_id = f"{subject_id}"
        if session_label:
            session_id = session_label.replace("ses-", "")
            display_id += f" {session_id}"

        logger.info(f"\nProcessing: {display_id}")

        # Find matching CT series
        ct_file = self.segmentation.find_ct_series(
            self.bids_root,
            subject_label,
            session_label=session_label,
            series_description_pattern=series_description_pattern,
        )

        if not ct_file:
            logger.warning(f"  ✗ No matching CT series found")
            return False

        logger.info(f"  ✓ Found CT image: {ct_file.name}")

        # Check if in Hounsfield units
        is_hu, reason = self.segmentation.is_in_hounsfield_units(ct_file)
        logger.info(f"  HU check: {reason}")

        if not is_hu:
            logger.error(f"  ✗ CT image not in Hounsfield units - cannot segment")
            return False

        # Set up output directory
        if output_dir is None:
            output_dir = (
                self.bids_root
                / "derivatives"
                / "totalsegmentator"
                / subject_label
            )
            if session_label:
                output_dir = output_dir / session_label

        output_dir.mkdir(parents=True, exist_ok=True)

        # Run segmentation(s)
        if isinstance(tasks, str):
            # Single task
            success = self.segmentation.run_segmentation(
                ct_file,
                output_dir,
                task=tasks,
                statistics=statistics,
                license_number=license_number,
                device=device,
                nr_thr_resamp=nr_thr_resamp,
                nr_thr_saving=nr_thr_saving,
                overwrite=overwrite,
            )
        else:
            # Multiple tasks
            results = self.segmentation.run_multiple_segmentations(
                ct_file,
                output_dir,
                tasks=tasks,
                statistics=statistics,
                license_number=license_number,
                device=device,
                nr_thr_resamp=nr_thr_resamp,
                nr_thr_saving=nr_thr_saving,
                overwrite=overwrite,
            )
            success = all(results.values())

        if success:
            logger.info(f"  ✓ All segmentations completed successfully")
            
            # Generate vertebrae slice reports if "total" task was run
            if (isinstance(tasks, str) and tasks == "total") or (isinstance(tasks, list) and "total" in tasks):
                try:
                    # Check if total directory exists
                    total_dir = output_dir / "total"
                    if total_dir.exists():
                        # Check if reports already exist
                        summary_file = output_dir / f"{subject_label}_vertebrae_slices.tsv"
                        lookup_file = output_dir / f"{subject_label}_slice_vertebrae.tsv"
                        
                        if not (summary_file.exists() and lookup_file.exists()):
                            logger.info(f"  Generating vertebrae slice reports...")
                            create_vertebrae_slice_report(
                                self.bids_root,
                                subject_label,
                                session_label=session_label,
                                output_dir=output_dir,
                            )
                            logger.info(f"  ✓ Vertebrae reports generated")
                        else:
                            logger.debug(f"  Vertebrae reports already exist, skipping")
                    else:
                        logger.debug(f"  Total segmentation directory not found, skipping vertebrae reports")
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to generate vertebrae reports: {e}")
            
            return True
        else:
            logger.error(f"  ✗ One or more segmentations failed")
            return False

    @staticmethod
    def _log_summary(results: Dict[str, int]):
        """Log processing results summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete:")
        logger.info(f"  Successful: {results['successful']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Skipped: {results['skipped']}")
        logger.info(f"{'='*60}")
