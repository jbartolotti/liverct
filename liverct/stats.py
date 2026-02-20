"""
Statistics utilities for CT segmentations.

Provides volume and intensity calculations that mirror TotalSegmentator's
statistics.json schema: {label: {volume, intensity}}.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

logger = logging.getLogger(__name__)

TISSUE_LABEL_PATTERNS: Dict[str, List[str]] = {
    "SAT": ["subcutaneous_fat", "subcutaneous_adipose", "subcutaneous", "sat"],
    "VAT": ["visceral_fat", "visceral_adipose", "visceral", "vat", "torso_fat"],
    "Muscle": ["skeletal_muscle", "muscle"],
}


def _match_label_file(files: List[Path], patterns: List[str]) -> Optional[Path]:
    for f in files:
        name = f.name.lower()
        if any(p in name for p in patterns):
            return f
    return None


def find_tissue_type_mask_files(seg_dir: Path) -> Dict[str, Path]:
    if not seg_dir.exists():
        raise FileNotFoundError(f"Tissue types directory not found: {seg_dir}")

    files = sorted(seg_dir.glob("*.nii*"))
    if not files:
        raise FileNotFoundError(f"No segmentation files found in: {seg_dir}")

    label_files: Dict[str, Path] = {}
    for label, patterns in TISSUE_LABEL_PATTERNS.items():
        match = _match_label_file(files, patterns)
        if match:
            label_files[label] = match
        else:
            logger.warning(f"Label not found for {label} in {seg_dir}")

    if not label_files:
        raise FileNotFoundError(f"No tissue type labels matched in: {seg_dir}")

    return label_files


def find_mask_files_in_dir(seg_dir: Path) -> Dict[str, Path]:
    if not seg_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")

    files = sorted(seg_dir.glob("*.nii*"))
    if not files:
        raise FileNotFoundError(f"No segmentation files found in: {seg_dir}")

    label_files: Dict[str, Path] = {}
    for f in files:
        if f.name.endswith(".nii.gz"):
            label = f.name[:-7]
        elif f.name.endswith(".nii"):
            label = f.stem
        else:
            continue

        if label:
            label_files[label] = f

    if not label_files:
        raise FileNotFoundError(f"No mask files matched in: {seg_dir}")

    return label_files


def _clip_mask_to_shape(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    if mask.shape == target_shape:
        return mask

    clipped = np.zeros(target_shape, dtype=bool)
    x_max = min(mask.shape[0], target_shape[0])
    y_max = min(mask.shape[1], target_shape[1])
    z_max = min(mask.shape[2], target_shape[2])
    clipped[:x_max, :y_max, :z_max] = mask[:x_max, :y_max, :z_max]
    return clipped


def compute_mask_statistics(
    ct_file: Path,
    mask_files: Dict[str, Path],
) -> Dict[str, Dict[str, float]]:
    if nib is None:
        raise ImportError("nibabel not installed. Install with: pip install nibabel")

    ct_img = nib.load(str(ct_file))
    logger.info(f"Loaded CT: {ct_file}")
    logger.info(f"CT shape: {ct_img.shape}, zooms: {ct_img.header.get_zooms()[:3]}")
    logger.info(f"Computing statistics for {len(mask_files)} label(s)")
    resample_cache: Dict[tuple, np.ndarray] = {}

    stats: Dict[str, Dict[str, float]] = {}

    for label, mask_path in mask_files.items():
        try:
            logger.info(f"Processing label: {label}")
            mask_img = nib.load(str(mask_path))
            mask = mask_img.get_fdata() > 0
            logger.info(
                f"Mask shape: {mask_img.shape}, zooms: {mask_img.header.get_zooms()[:3]}"
            )

            voxel_count = int(mask.sum())
            if voxel_count == 0:
                stats[label] = {"volume": 0.0, "intensity": 0.0}
                logger.info("Voxel count is 0; volume/intensity set to 0.0")
                continue

            cache_key = (mask_img.shape, mask_img.affine.tobytes())
            if cache_key in resample_cache:
                ct_data = resample_cache[cache_key]
            else:
                try:
                    from nibabel.processing import resample_from_to
                    logger.info(
                        f"Resampling CT to mask grid for {label} (shape={mask_img.shape})"
                    )
                    ct_resampled = resample_from_to(ct_img, mask_img, order=1)
                    ct_data = ct_resampled.get_fdata().astype(np.float32)
                    logger.info(f"Resampling complete for {label}")
                except Exception as e:
                    logger.warning(f"Resampling failed for {label}, using nearest-shape clip: {e}")
                    ct_data = ct_img.get_fdata().astype(np.float32)
                    if ct_data.shape != mask.shape:
                        ct_data = _clip_mask_to_shape(ct_data, mask.shape)
                resample_cache[cache_key] = ct_data

            voxel_volume_mm3 = float(np.prod(mask_img.header.get_zooms()[:3]))
            values = ct_data[mask]
            volume = float(voxel_count * voxel_volume_mm3)
            intensity = float(np.mean(values))
            stats[label] = {
                "volume": volume,
                "intensity": intensity,
            }
            logger.info(
                f"Computed: voxels={voxel_count}, volume_mm3={volume:.6f}, intensity={intensity:.6f}"
            )
        except Exception as e:
            logger.warning(f"Failed to compute statistics for {label}: {e}")

    return stats


def compute_tissue_types_statistics(ct_file: Path, tissue_types_dir: Path) -> Dict[str, Dict[str, float]]:
    mask_files = find_tissue_type_mask_files(tissue_types_dir)
    return compute_mask_statistics(ct_file=ct_file, mask_files=mask_files)


def compute_task_statistics(ct_file: Path, task_dir: Path, task: str) -> Dict[str, Dict[str, float]]:
    if task == "tissue_types":
        return compute_tissue_types_statistics(ct_file=ct_file, tissue_types_dir=task_dir)

    mask_files = find_mask_files_in_dir(task_dir)
    return compute_mask_statistics(ct_file=ct_file, mask_files=mask_files)


def load_statistics_json(stats_path: Path) -> Dict[str, Dict[str, float]]:
    stats_path = Path(stats_path)
    with open(stats_path, "r") as f:
        return json.load(f)


def compare_statistics(
    computed: Dict[str, Dict[str, float]],
    reference: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    labels = set(computed.keys()) | set(reference.keys())
    diffs: Dict[str, Dict[str, float]] = {}
    for label in sorted(labels):
        c = computed.get(label, {"volume": 0.0, "intensity": 0.0})
        r = reference.get(label, {"volume": 0.0, "intensity": 0.0})
        diffs[label] = {
            "volume_diff": float(c.get("volume", 0.0) - r.get("volume", 0.0)),
            "intensity_diff": float(c.get("intensity", 0.0) - r.get("intensity", 0.0)),
        }
    return diffs


def compare_task_statistics_to_json(
    ct_file: Path,
    task_dir: Path,
    task: str,
    reference_stats_json: Path,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    computed = compute_task_statistics(ct_file=ct_file, task_dir=task_dir, task=task)
    reference = load_statistics_json(reference_stats_json)
    diffs = compare_statistics(computed=computed, reference=reference)
    for label in sorted(set(computed.keys()) | set(reference.keys())):
        c = computed.get(label, {"volume": 0.0, "intensity": 0.0})
        r = reference.get(label, {"volume": 0.0, "intensity": 0.0})
        d = diffs.get(label, {"volume_diff": 0.0, "intensity_diff": 0.0})
        logger.info(
            "Compare %s | computed(volume=%.6f, intensity=%.6f) "
            "reference(volume=%.6f, intensity=%.6f) "
            "diff(volume=%.6f, intensity=%.6f)",
            label,
            float(c.get("volume", 0.0)),
            float(c.get("intensity", 0.0)),
            float(r.get("volume", 0.0)),
            float(r.get("intensity", 0.0)),
            float(d.get("volume_diff", 0.0)),
            float(d.get("intensity_diff", 0.0)),
        )
    return computed, diffs


def compute_segmentation_statistics(
    bids_root: Path,
    tasks: Optional[List[str]] = None,
    subjects: Optional[List[str]] = None,
    session_label: Optional[str] = None,
    overwrite: bool = True,
) -> Dict[str, Dict[str, bool]]:
    """
    Compute statistics.json for existing segmentations in a BIDS dataset.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    tasks : list of str, optional
        Task names to compute statistics for (e.g., ["tissue_types", "liver_segments"]).
        If None, attempts to find all task directories for each subject.
    subjects : list of str, optional
        Subject labels to process. If None, processes all subjects in derivatives/totalsegmentator.
    session_label : str, optional
        Session label (with or without "ses-" prefix)
    overwrite : bool
        Whether to overwrite existing statistics.json files (default: True)

    Returns
    -------
    dict
        Nested dictionary: {subject: {task: success_bool}}
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Starting statistics computation for segmentations")
    logger.info(f"BIDS root: {bids_root}")
    logger.info(f"Tasks: {tasks if tasks else 'auto-detect'}")
    logger.info(f"Subjects: {subjects if subjects else 'all'}")
    logger.info(f"Session: {session_label}")
    logger.info(f"Overwrite: {overwrite}")

    bids_root = Path(bids_root)
    derivatives_dir = bids_root / "derivatives" / "totalsegmentator"
    
    if not derivatives_dir.exists():
        logger.error(f"Derivatives directory not found: {derivatives_dir}")
        return {}

    # Get subject list
    if subjects is None:
        subject_dirs = sorted([d for d in derivatives_dir.glob("sub-*") if d.is_dir()])
        subject_labels = [d.name for d in subject_dirs]
    else:
        subject_labels = [s if s.startswith("sub-") else f"sub-{s}" for s in subjects]

    logger.info(f"Found {len(subject_labels)} subject(s) to process")

    results: Dict[str, Dict[str, bool]] = {}

    for subject_label in subject_labels:
        logger.info(f"\nProcessing {subject_label}")
        subject_dir = derivatives_dir / subject_label
        
        if session_label:
            ses_label = session_label if session_label.startswith("ses-") else f"ses-{session_label}"
            subject_dir = subject_dir / ses_label
        
        if not subject_dir.exists():
            logger.warning(f"  Subject directory not found: {subject_dir}")
            continue

        # Get task list
        if tasks is None:
            # Auto-detect task directories
            task_dirs = [d for d in subject_dir.iterdir() if d.is_dir() and d.name != "figures"]
            task_names = [d.name for d in task_dirs]
        else:
            task_names = tasks

        if not task_names:
            logger.warning(f"  No task directories found for {subject_label}")
            continue

        logger.info(f"  Tasks to process: {task_names}")
        results[subject_label] = {}

        for task in task_names:
            task_dir = subject_dir / task
            
            if not task_dir.exists():
                logger.warning(f"    Task directory not found: {task}")
                results[subject_label][task] = False
                continue

            stats_file = task_dir / "statistics.json"
            
            if stats_file.exists() and not overwrite:
                logger.info(f"    ↷ Statistics exist for {task}, skipping")
                results[subject_label][task] = True
                continue

            # Find CT from source.json
            ct_file = find_ct_from_source_json(task_dir)
            if ct_file is None or not ct_file.exists():
                logger.warning(f"    ✗ CT file not found for {task}")
                results[subject_label][task] = False
                continue

            # Compute statistics
            try:
                logger.info(f"    Computing statistics for {task}...")
                stats = compute_task_statistics(ct_file, task_dir, task=task)
                save_statistics_json(stats, stats_file)
                logger.info(f"    ✓ Statistics saved: {stats_file}")
                results[subject_label][task] = True
            except Exception as e:
                logger.error(f"    ✗ Failed to compute statistics for {task}: {e}")
                results[subject_label][task] = False

    # Summary
    total_tasks = sum(len(tasks) for tasks in results.values())
    successful = sum(sum(1 for success in tasks.values() if success) for tasks in results.values())
    failed = total_tasks - successful

    logger.info(f"\n{'='*60}")
    logger.info(f"Statistics computation complete:")
    logger.info(f"  {successful}/{total_tasks} successful")
    logger.info(f"  {failed}/{total_tasks} failed")
    logger.info(f"{'='*60}")

    return results


def save_statistics_json(stats: Dict[str, Dict[str, float]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def find_ct_from_source_json(task_dir: Path) -> Optional[Path]:
    source_file = task_dir / "source.json"
    if not source_file.exists():
        return None

    try:
        with open(source_file, "r") as f:
            metadata = json.load(f)
        source_path = Path(metadata.get("source_file", ""))
        if source_path.exists():
            return source_path
    except Exception as e:
        logger.warning(f"Failed to read source.json from {task_dir}: {e}")

    return None
