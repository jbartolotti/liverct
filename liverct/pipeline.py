"""
High-level processing pipelines for CT BIDS datasets.

Coordinates multiple processing steps and handles subject/session iteration.
"""

import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from .segmentation import CTSegmentationPipeline

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

        Returns
        -------
        dict
            Summary with keys: 'successful', 'failed', 'skipped'
        """
        subject_dirs = sorted([d for d in self.bids_root.glob("sub-*") if d.is_dir()])
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
        segmentation_model: str = "3d_fullres",
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
        segmentation_model : str
            TotalSegmentator model to use
        output_dir : Path, optional
            Where to save segmentation derivatives. If None, uses
            bids_root/derivatives/totalsegmentator/

        Returns
        -------
        bool
            True if segmentation succeeded, False otherwise
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

        # Run segmentation
        success = self.segmentation.run_segmentation(
            ct_file, output_dir, model=segmentation_model
        )

        if success:
            logger.info(f"  ✓ Segmentation completed successfully")
            return True
        else:
            logger.error(f"  ✗ Segmentation failed")
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
