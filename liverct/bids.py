"""
BIDS conversion utilities for CT imaging data.

Uses dcm2bids4ct for converting DICOM CT data to BIDS format following
the proposed CT extension: https://bids.neuroimaging.io/extensions/beps/bep_024.html
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class CTBIDSConverter:
    """Convert DICOM CT data to BIDS format using dcm2bids4ct."""

    def __init__(self, dcm2bids4ct_path: Optional[str] = None):
        """
        Initialize the converter.

        Parameters
        ----------
        dcm2bids4ct_path : str, optional
            Path to dcm2bids4ct executable. If None, assumes it's in PATH.
        """
        self.dcm2bids4ct_path = dcm2bids4ct_path or "dcm2bids4ct"

    def convert(
        self,
        dicom_dir: str,
        bids_root: str,
        subject_id: str,
        session_id: Optional[str] = None,
        config_file: Optional[str] = None,
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
        subject_id : str
            Subject identifier (e.g., "001", "sub-001").
        session_id : str, optional
            Session identifier (e.g., "01", "ses-01").
        config_file : str, optional
            Path to custom dcm2bids configuration file.
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

        # Ensure subject_id has correct format
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{subject_id}"

        # Build command
        cmd = [
            self.dcm2bids4ct_path,
            "-d",
            str(dicom_path),
            "-o",
            str(bids_path),
            "-sub",
            subject_id.replace("sub-", ""),  # dcm2bids4ct expects without "sub-" prefix
        ]

        if session_id:
            if not session_id.startswith("ses-"):
                session_id = f"ses-{session_id}"
            cmd.extend(["-ses", session_id.replace("ses-", "")])

        if config_file:
            cmd.extend(["-c", str(config_file)])

        # Add any additional kwargs
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

        # Run conversion
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("DICOM to BIDS conversion completed successfully")
            if result.stdout:
                logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"DICOM to BIDS conversion failed: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error(
                f"dcm2bids4ct not found. Please install it or provide path to executable."
            )
            return False
