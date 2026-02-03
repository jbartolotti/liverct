"""
Figure generation utilities for liver CT segmentations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

DEFAULT_LABEL_COLORS = {
    "SAT": "#f4a261",          # subcutaneous adipose tissue
    "VAT": "#e2e750",          # visceral adipose tissue
    "Muscle": "#a32020",       # muscle
    "Skeletal": "#169C83",     # skeletal tissue/bone
}

LABEL_PATTERNS = {
    "SAT": ["subcutaneous_fat", "subcutaneous_adipose", "subcutaneous", "sat"],
    "VAT": ["visceral_fat", "visceral_adipose", "visceral", "vat", "torso_fat"],
    "Muscle": ["skeletal_muscle", "muscle"],
}

TOTAL_BONE_PATTERNS = [
    "skull",
    "clavicula"
    "scapula",
    "humerus",
    "vertebra",
    "sternum",
    "rib",
    "costal_cartilages",
    "ulna",
    "radius",
    "intervertebral_discs",
    "hip",
    "sacrum",
    "carpal",
    "metacarpal",
    "phalanges_hand",
    "femur",
    "patella",
    "fibula",
    "tibia",
    "tarsal",
    "metatarsal",
    "phalanges_foot",
]    


def _normalize_subject_label(subject_label: str) -> str:
    return subject_label if subject_label.startswith("sub-") else f"sub-{subject_label}"


def _normalize_session_label(session_label: Optional[str]) -> Optional[str]:
    if session_label is None:
        return None
    return session_label if session_label.startswith("ses-") else f"ses-{session_label}"


def _find_first_nifti(search_dir: Path, pattern: Optional[str] = None) -> Optional[Path]:
    import re
    nifti_files = sorted(search_dir.glob("*.nii*"))
    
    if not nifti_files:
        return None
    
    # If no pattern, return first file
    if not pattern:
        return nifti_files[0]
    
    # Try to match based on json sidecar SeriesDescription
    for nifti_file in nifti_files:
        json_file = nifti_file.with_suffix(".json")
        if json_file.exists():
            try:
                import json
                with open(json_file) as f:
                    metadata = json.load(f)
                    series_desc = metadata.get("SeriesDescription", "")
                    if re.search(pattern, series_desc, re.IGNORECASE):
                        return nifti_file
            except Exception as e:
                logger.debug(f"Could not read metadata for {nifti_file}: {e}")
    
    # Fall back to first file if no pattern match
    logger.warning(f"No CT matched pattern in {search_dir}, using first file")
    return nifti_files[0]


def find_ct_nifti(
    bids_root: Path,
    subject_label: str,
    session_label: Optional[str] = None,
    series_description_pattern: Optional[str] = None,
) -> Optional[Path]:
    subject_label = _normalize_subject_label(subject_label)
    session_label = _normalize_session_label(session_label)

    subject_dir = Path(bids_root) / subject_label
    if session_label:
        ct_dir = subject_dir / session_label / "ct"
        if ct_dir.exists():
            return _find_first_nifti(ct_dir, pattern=series_description_pattern)
        return None

    ct_dir = subject_dir / "ct"
    if ct_dir.exists():
        return _find_first_nifti(ct_dir, pattern=series_description_pattern)

    # Look for sessioned CTs if no top-level ct directory
    for ses_dir in sorted(subject_dir.glob("ses-*")):
        ct_dir = ses_dir / "ct"
        if ct_dir.exists():
            ct_file = _find_first_nifti(ct_dir, pattern=series_description_pattern)
            if ct_file:
                return ct_file

    return None


def find_tissue_types_dir(
    bids_root: Path,
    subject_label: str,
    session_label: Optional[str] = None,
) -> Path:
    subject_label = _normalize_subject_label(subject_label)
    session_label = _normalize_session_label(session_label)

    base_dir = Path(bids_root) / "derivatives" / "totalsegmentator" / subject_label
    if session_label:
        base_dir = base_dir / session_label
    return base_dir / "tissue_types"


def find_total_dir(
    bids_root: Path,
    subject_label: str,
    session_label: Optional[str] = None,
) -> Path:
    subject_label = _normalize_subject_label(subject_label)
    session_label = _normalize_session_label(session_label)

    base_dir = Path(bids_root) / "derivatives" / "totalsegmentator" / subject_label
    if session_label:
        base_dir = base_dir / session_label
    return base_dir / "total"


def _match_label_file(
    files: List[Path],
    patterns: List[str],
    exclude: Optional[List[str]] = None,
) -> Optional[Path]:
    exclude = exclude or []
    for f in files:
        name = f.name.lower()
        if any(p in name for p in patterns) and not any(e in name for e in exclude):
            return f
    return None


def find_tissue_type_label_files(seg_dir: Path) -> Dict[str, Path]:
    if not seg_dir.exists():
        raise FileNotFoundError(f"Tissue types directory not found: {seg_dir}")

    files = sorted(seg_dir.glob("*.nii*"))
    if not files:
        raise FileNotFoundError(f"No segmentation files found in: {seg_dir}")

    label_files: Dict[str, Path] = {}
    for label, patterns in LABEL_PATTERNS.items():
        match = _match_label_file(files, patterns)
        if match:
            label_files[label] = match
        else:
            logger.warning(f"Label not found for {label} in {seg_dir}")

    if not label_files:
        raise FileNotFoundError(f"No tissue type labels matched in: {seg_dir}")

    return label_files


def find_total_bone_files(total_dir: Path) -> List[Path]:
    if not total_dir.exists():
        raise FileNotFoundError(f"Total segmentation directory not found: {total_dir}")

    files = sorted(total_dir.glob("*.nii*"))
    if not files:
        raise FileNotFoundError(f"No segmentation files found in: {total_dir}")

    bone_files: List[Path] = []
    for f in files:
        name = f.name.lower()
        if any(p in name for p in TOTAL_BONE_PATTERNS):
            bone_files.append(f)

    if not bone_files:
        raise FileNotFoundError(f"No bone masks matched in: {total_dir}")

    return bone_files


def build_composite_mask_from_files(
    mask_files: List[Path],
    reference_shape: Tuple[int, ...],
) -> np.ndarray:
    composite = np.zeros(reference_shape, dtype=bool)
    for mask_path in mask_files:
        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata() > 0
        if mask_data.shape != reference_shape:
            logger.warning(
                f"Skipping mask with shape mismatch: {mask_path.name}"
                f" (shape {mask_data.shape} != {reference_shape})"
            )
            continue
        composite |= mask_data
    return composite


def create_and_save_skeletal_composite(
    bids_root: Path,
    subject_label: str,
    session_label: Optional[str] = None,
) -> Path:
    """
    Create composite skeletal mask from individual bone masks in total/ and save to tissue_types/.
    
    This is a derived composite file combining all individual bone masks.
    Not a direct TotalSegmentator output.
    
    Returns
    -------
    Path
        Path to saved composite skeletal mask (tissue_types/skeletal_composite.nii.gz)
    """
    subject_label = _normalize_subject_label(subject_label)
    session_label = _normalize_session_label(session_label)
    
    total_dir = find_total_dir(bids_root, subject_label, session_label=session_label)
    bone_files = find_total_bone_files(total_dir)
    
    # Load first bone file to get reference shape and affine
    first_bone = nib.load(str(bone_files[0]))
    ref_shape = first_bone.shape
    ref_affine = first_bone.affine
    
    # Build composite
    composite = build_composite_mask_from_files(bone_files, reference_shape=ref_shape)
    
    # Save to tissue_types folder
    tissue_types_dir = find_tissue_types_dir(bids_root, subject_label, session_label=session_label)
    tissue_types_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = tissue_types_dir / "skeletal_composite.nii.gz"
    composite_img = nib.Nifti1Image(composite.astype(np.uint8), affine=ref_affine)
    nib.save(composite_img, str(output_path))
    
    logger.info(f"Saved skeletal composite: {output_path}")
    return output_path


def _get_anatomical_z_range(
    bids_root: Path,
    subject_label: str,
    session_label: Optional[str] = None,
    superior_limit: Optional[str] = None,
    inferior_limit: Optional[str] = None,
    axis: int = 2,
) -> Optional[Tuple[int, int]]:
    """
    Determine anatomical Z range based on vertebrae/sacrum segmentations.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subject_label : str
        Subject label
    session_label : str, optional
        Session label
    superior_limit : str, optional
        Superior anatomical limit (e.g., "C1", "T1"). None = no limit
    inferior_limit : str, optional
        Inferior anatomical limit (e.g., "S1", "sacrum"). None = no limit
    axis : int
        Axis along which to compute range (default: 2 for axial)

    Returns
    -------
    tuple or None
        (min_slice, max_slice) or None if landmarks not found
    """
    if superior_limit is None and inferior_limit is None:
        return None

    # Normalize labels
    subject_label = subject_label if subject_label.startswith("sub-") else f"sub-{subject_label}"
    if session_label:
        session_label = session_label if session_label.startswith("ses-") else f"ses-{session_label}"

    # Find TotalSegmentator directory
    try:
        total_dir = find_total_dir(bids_root, subject_label, session_label)
        if not total_dir.exists():
            logger.warning(f"Total segmentations directory not found: {total_dir}")
            return None
    except Exception as e:
        logger.warning(f"Could not find TotalSegmentator directory: {e}")
        return None

    # Map anatomical labels to file patterns
    def get_file_pattern(label: str) -> str:
        label_lower = label.lower()
        if label_lower == "sacrum":
            return "sacrum.nii.gz"
        # Vertebrae: C1-C7, T1-T12, L1-L5, S1
        return f"vertebrae_{label.upper()}.nii.gz"

    min_slice, max_slice = None, None

    # Process superior limit
    if superior_limit:
        pattern = get_file_pattern(superior_limit)
        file_path = total_dir / pattern
        if file_path.exists():
            try:
                img = nib.load(str(file_path))
                mask = img.get_fdata() > 0
                # Find superior-most (highest index) slice with this structure
                axes = tuple(i for i in range(mask.ndim) if i != axis)
                slice_sums = mask.sum(axis=axes)
                indices = np.where(slice_sums > 0)[0]
                if indices.size > 0:
                    max_slice = int(indices.max())
                    logger.info(f"Superior limit {superior_limit}: slice {max_slice}")
            except Exception as e:
                logger.warning(f"Could not load {superior_limit} segmentation: {e}")
        else:
            logger.warning(f"Segmentation file not found: {file_path}")

    # Process inferior limit
    if inferior_limit:
        pattern = get_file_pattern(inferior_limit)
        file_path = total_dir / pattern
        if file_path.exists():
            try:
                img = nib.load(str(file_path))
                mask = img.get_fdata() > 0
                # Find inferior-most (lowest index) slice with this structure
                axes = tuple(i for i in range(mask.ndim) if i != axis)
                slice_sums = mask.sum(axis=axes)
                indices = np.where(slice_sums > 0)[0]
                if indices.size > 0:
                    min_slice = int(indices.min())
                    logger.info(f"Inferior limit {inferior_limit}: slice {min_slice}")
            except Exception as e:
                logger.warning(f"Could not load {inferior_limit} segmentation: {e}")
        else:
            logger.warning(f"Segmentation file not found: {file_path}")

    if min_slice is None and max_slice is None:
        return None

    return (min_slice, max_slice)


def _select_slices(
    mask_union: np.ndarray,
    axis: int,
    num_slices: int,
    z_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> List[int]:
    """
    Select slice indices for montage.

    Parameters
    ----------
    mask_union : np.ndarray
        Union of all masks
    axis : int
        Axis along which to slice
    num_slices : int
        Number of slices to select
    z_range : tuple, optional
        (min_slice, max_slice) to limit the range. Either can be None.

    Returns
    -------
    list
        Selected slice indices
    """
    # Determine slices with any mask content
    axes = tuple(i for i in range(mask_union.ndim) if i != axis)
    slice_sums = mask_union.sum(axis=axes)
    candidate_indices = np.where(slice_sums > 0)[0]

    if candidate_indices.size == 0:
        # Fall back to middle slices
        total_slices = mask_union.shape[axis]
        return list(np.linspace(0, total_slices - 1, num_slices, dtype=int))

    start, end = int(candidate_indices.min()), int(candidate_indices.max())

    # Apply anatomical range limits if provided
    if z_range is not None:
        min_limit, max_limit = z_range
        if min_limit is not None:
            start = max(start, min_limit)
        if max_limit is not None:
            end = min(end, max_limit)
        logger.info(f"Applied anatomical range limits: slice {start} to {end}")

    indices = np.linspace(start, end, num_slices, dtype=int)
    return list(np.unique(indices))


def create_label_overlay_montage(
    ct_file: Path,
    label_files: Dict[str, Path],
    output_png: Path,
    num_slices: int = 12,
    axis: int = 2,
    window: Tuple[int, int] = (-200, 250),
    alpha: float = 0.35,
    dpi: int = 200,
    label_colors: Optional[Dict[str, str]] = None,
    label_arrays: Optional[Dict[str, np.ndarray]] = None,
    z_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> Path:
    """
    Create montage of CT slices with label overlays.

    Parameters
    ----------
    ct_file : Path
        Path to CT NIfTI file
    label_files : dict
        Mapping of label name to NIfTI mask path
    output_png : Path
        Output PNG path
    num_slices : int
        Number of slices in montage
    axis : int
        Axis for slicing (default: 2 for axial)
    window : tuple
        HU window (min, max)
    alpha : float
        Overlay alpha
    dpi : int
        Output DPI
    label_colors : dict, optional
        Custom label colors
    z_range : tuple, optional
        (min_slice, max_slice) to limit anatomical range. Either can be None.

    Returns
    -------
    Path
        Output PNG path
    """
    label_colors = label_colors or DEFAULT_LABEL_COLORS

    ct_img = nib.load(str(ct_file))
    ct_data = ct_img.get_fdata().astype(np.float32)
    logger.info(f"CT image shape: {ct_data.shape}")

    # Load masks
    label_masks: Dict[str, np.ndarray] = {}
    mask_union = np.zeros(ct_data.shape, dtype=np.uint8)
    for label, mask_path in label_files.items():
        try:
            mask_img = nib.load(str(mask_path))
            logger.info(f"  {label}: {mask_path.name} shape={mask_img.shape}")
            if mask_img.shape != ct_data.shape:
                logger.error(
                    f"    ✗ Shape mismatch! Expected {ct_data.shape}, got {mask_img.shape}"
                )
                continue
            mask_data = mask_img.get_fdata() > 0
            label_masks[label] = mask_data
            mask_union |= mask_data.astype(np.uint8)
        except Exception as e:
            logger.error(f"  ✗ Failed to load {label} ({mask_path.name}): {e}")

    if label_arrays:
        for label, mask_data in label_arrays.items():
            try:
                logger.info(f"  {label} (composite): shape={mask_data.shape}")
                if mask_data.shape != ct_data.shape:
                    logger.error(
                        f"    ✗ Shape mismatch! Expected {ct_data.shape}, got {mask_data.shape}"
                    )
                    continue
                label_masks[label] = mask_data
                mask_union |= mask_data.astype(np.uint8)
            except Exception as e:
                logger.error(f"  ✗ Failed to add {label}: {e}")

    # Slice selection
    slice_indices = _select_slices(
        mask_union, axis=axis, num_slices=num_slices, z_range=z_range
    )
    if len(slice_indices) == 0:
        raise ValueError("No slices selected for montage")

    # Montage layout
    cols = 4
    rows = int(np.ceil(len(slice_indices) / cols))

    vmin, vmax = window

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=dpi)
    axes = np.atleast_2d(axes)

    for i, slice_idx in enumerate(slice_indices):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")

        if axis == 0:
            ct_slice = ct_data[slice_idx, :, :]
            masks = {k: v[slice_idx, :, :] for k, v in label_masks.items()}
        elif axis == 1:
            ct_slice = ct_data[:, slice_idx, :]
            masks = {k: v[:, slice_idx, :] for k, v in label_masks.items()}
        else:
            ct_slice = ct_data[:, :, slice_idx]
            masks = {k: v[:, :, slice_idx] for k, v in label_masks.items()}

        # Window and normalize
        ct_slice = np.clip(ct_slice, vmin, vmax)
        ct_slice = (ct_slice - vmin) / float(vmax - vmin)

        ax.imshow(ct_slice.T, cmap="gray", origin="lower")

        # Create RGBA overlay
        overlay = np.zeros((*ct_slice.shape, 4), dtype=np.float32)
        for label, mask_slice in masks.items():
            if label not in label_colors:
                continue
            color = mcolors.to_rgba(label_colors[label], alpha=alpha)
            overlay[mask_slice.T > 0] = color

        ax.imshow(overlay, origin="lower")
        ax.set_title(f"Slice {slice_idx}")

    # Turn off any unused axes
    for j in range(len(slice_indices), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    # Legend
    legend_labels = list(label_masks.keys())
    legend_patches = [
        Patch(color=label_colors[label], label=label)
        for label in legend_labels
        if label in label_colors
    ]
    if legend_patches:
        fig.legend(handles=legend_patches, loc="lower center", ncol=4)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved montage: {output_png}")
    return output_png


def create_tissue_types_montage_from_bids(
    bids_root: Path,
    subject_label: str,
    session_label: Optional[str] = None,
    output_dir: Optional[Path] = None,
    series_description_pattern: Optional[str] = None,
    include_skeletal: bool = True,
    superior_limit: Optional[str] = None,
    inferior_limit: Optional[str] = None,
    **kwargs,
) -> Path:
    """
    Convenience wrapper to generate a tissue types montage from BIDS structure.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subject_label : str
        Subject label
    session_label : str, optional
        Session label
    output_dir : Path, optional
        Output directory for montage
    series_description_pattern : str, optional
        Pattern to match CT series
    include_skeletal : bool
        Include skeletal composite overlay
    superior_limit : str, optional
        Superior anatomical limit (e.g., "C1", "T1"). None = no limit.
    inferior_limit : str, optional
        Inferior anatomical limit (e.g., "S1", "sacrum", "L5"). None = no limit.
        Common choices: "sacrum" (includes entire sacrum), "S1" (first sacral vertebra),
        "L5" (stop at lowest lumbar vertebra).
    **kwargs
        Additional arguments passed to create_label_overlay_montage

    Returns
    -------
    Path
        Output montage file path
    """
    ct_file = find_ct_nifti(
        bids_root,
        subject_label,
        session_label=session_label,
        series_description_pattern=series_description_pattern,
    )
    if not ct_file:
        raise FileNotFoundError(f"CT NIfTI not found for subject: {subject_label}")

    seg_dir = find_tissue_types_dir(bids_root, subject_label, session_label=session_label)
    label_files = find_tissue_type_label_files(seg_dir)

    label_arrays: Dict[str, np.ndarray] = {}
    if include_skeletal:
        try:
            # Check if composite already exists
            skeletal_file = seg_dir / "skeletal_composite.nii.gz"
            if not skeletal_file.exists():
                logger.info("Creating skeletal composite from total masks...")
                create_and_save_skeletal_composite(
                    bids_root, subject_label, session_label=session_label
                )
            
            # Load the composite
            if skeletal_file.exists():
                skeletal_img = nib.load(str(skeletal_file))
                logger.info(f"Loaded skeletal composite: shape={skeletal_img.shape}")
                ct_img = nib.load(str(ct_file))
                if skeletal_img.shape == ct_img.shape:
                    skeletal_mask = skeletal_img.get_fdata() > 0
                    if skeletal_mask.any():
                        label_arrays["Skeletal"] = skeletal_mask
                    else:
                        logger.warning(f"Skeletal composite mask is empty for {subject_label}")
                else:
                    logger.error(
                        f"Skeletal composite shape mismatch: expected {ct_img.shape}, got {skeletal_img.shape}"
                    )
            else:
                logger.warning(f"Skeletal composite not found and could not be created")
        except Exception as e:
            logger.warning(f"Skipping skeletal composite for {subject_label}: {e}")

    if output_dir is None:
        output_dir = seg_dir / "figures"

    subject_label = _normalize_subject_label(subject_label)
    session_label = _normalize_session_label(session_label)
    output_name = f"{subject_label}"
    if session_label:
        output_name += f"_{session_label}"
    output_name += "_tissue_types_montage.png"

    # Compute anatomical Z range if limits specified
    z_range = None
    axis = kwargs.get("axis", 2)  # Default to axial
    if superior_limit or inferior_limit:
        z_range = _get_anatomical_z_range(
            bids_root,
            subject_label,
            session_label=session_label,
            superior_limit=superior_limit,
            inferior_limit=inferior_limit,
            axis=axis,
        )

    output_png = Path(output_dir) / output_name
    return create_label_overlay_montage(
        ct_file,
        label_files,
        output_png,
        label_arrays=label_arrays if label_arrays else None,
        z_range=z_range,
        **kwargs,
    )


def create_tissue_types_montages(
    bids_root: Path,
    subjects: Optional[Union[str, List[str]]] = None,
    session_label: Optional[str] = None,
    output_dir: Optional[Path] = None,
    series_description_pattern: Optional[str] = None,
    include_skeletal: bool = True,
    superior_limit: Optional[str] = None,
    inferior_limit: Optional[str] = None,
    **kwargs,
) -> Dict[str, Path]:
    """
    Create tissue types montages for multiple subjects.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subjects : str or list, optional
        Subject(s) to process. None = all subjects.
    session_label : str, optional
        Session label
    output_dir : Path, optional
        Output directory
    series_description_pattern : str, optional
        Pattern to match CT series
    include_skeletal : bool
        Include skeletal composite overlay
    superior_limit : str, optional
        Superior anatomical limit (e.g., "C1", "T1").
    inferior_limit : str, optional
        Inferior anatomical limit (e.g., "S1", "sacrum", "L5").
    **kwargs
        Additional arguments passed to montage creation

    Returns
    -------
    dict
        Mapping of subject_label to output file path
    """
    bids_root = Path(bids_root)
    if subjects is None:
        subject_dirs = sorted([d for d in bids_root.glob("sub-*") if d.is_dir()])
        subject_labels = [d.name for d in subject_dirs]
    elif isinstance(subjects, str):
        subject_labels = [subjects]
    else:
        subject_labels = subjects

    results: Dict[str, Path] = {}
    for subject_label in subject_labels:
        try:
            output = create_tissue_types_montage_from_bids(
                bids_root=bids_root,
                subject_label=subject_label,
                session_label=session_label,
                output_dir=output_dir,
                series_description_pattern=series_description_pattern,
                include_skeletal=include_skeletal,
                superior_limit=superior_limit,
                inferior_limit=inferior_limit,
                **kwargs,
            )
            results[subject_label] = output
        except Exception as e:
            logger.error(f"Failed montage for {subject_label}: {e}")

    return results


def create_vertebrae_slice_report(
    bids_root: Path,
    subject_label: str,
    session_label: Optional[str] = None,
    output_dir: Optional[Path] = None,
    axis: int = 2,
    include_sacrum: bool = True,
) -> Tuple[Path, Path]:
    """
    Generate vertebrae slice location reports for a subject.

    Creates two TSV files:
    1. vertebrae_slices.tsv: One row per vertebra with min/max/center slice info
    2. slice_vertebrae.tsv: One row per slice with vertebrae present

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subject_label : str
        Subject label
    session_label : str, optional
        Session label
    output_dir : Path, optional
        Output directory. If None, uses derivatives/totalsegmentator/sub-XXX/
    axis : int
        Axis for slice indexing (default: 2 for axial)
    include_sacrum : bool
        Include sacrum in addition to individual vertebrae

    Returns
    -------
    tuple
        (vertebrae_summary_path, slice_lookup_path)
    """
    import csv
    from collections import defaultdict

    # Normalize labels
    subject_label = subject_label if subject_label.startswith("sub-") else f"sub-{subject_label}"
    if session_label:
        session_label = session_label if session_label.startswith("ses-") else f"ses-{session_label}"

    # Find total directory
    total_dir = find_total_dir(bids_root, subject_label, session_label)
    if not total_dir.exists():
        raise FileNotFoundError(f"Total segmentations directory not found: {total_dir}")

    # Determine output directory
    if output_dir is None:
        base_dir = Path(bids_root) / "derivatives" / "totalsegmentator" / subject_label
        if session_label:
            base_dir = base_dir / session_label
        output_dir = base_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define vertebrae to process
    vertebrae_labels = []
    # Cervical
    for i in range(1, 8):
        vertebrae_labels.append(f"C{i}")
    # Thoracic
    for i in range(1, 13):
        vertebrae_labels.append(f"T{i}")
    # Lumbar
    for i in range(1, 6):
        vertebrae_labels.append(f"L{i}")
    # Sacral
    vertebrae_labels.append("S1")
    if include_sacrum:
        vertebrae_labels.append("sacrum")

    # Collect vertebra information
    vertebra_data = []
    slice_to_vertebrae = defaultdict(list)

    for vertebra in vertebrae_labels:
        # Construct filename
        if vertebra.lower() == "sacrum":
            filename = "sacrum.nii.gz"
        else:
            filename = f"vertebrae_{vertebra}.nii.gz"
        
        file_path = total_dir / filename
        if not file_path.exists():
            logger.debug(f"Vertebra segmentation not found: {filename}")
            continue

        try:
            # Load mask
            img = nib.load(str(file_path))
            mask = img.get_fdata() > 0

            # Get slice indices where vertebra is present
            axes = tuple(i for i in range(mask.ndim) if i != axis)
            slice_sums = mask.sum(axis=axes)
            indices = np.where(slice_sums > 0)[0]

            if indices.size == 0:
                logger.warning(f"No voxels found in {filename}")
                continue

            min_slice = int(indices.min())
            max_slice = int(indices.max())
            extent = max_slice - min_slice + 1
            num_voxels = int(mask.sum())

            # Compute center of mass along the specified axis
            # Get coordinates of all voxels in the mask
            coords = np.where(mask)
            axis_coords = coords[axis]
            center_slice = int(np.round(axis_coords.mean()))

            # Record vertebra data
            vertebra_data.append({
                "vertebra": vertebra,
                "min_slice": min_slice,
                "max_slice": max_slice,
                "center_slice": center_slice,
                "extent": extent,
                "num_voxels": num_voxels,
            })

            # Record slice-to-vertebra mapping
            for slice_idx in range(min_slice, max_slice + 1):
                slice_to_vertebrae[slice_idx].append(vertebra)

            logger.info(f"{vertebra}: slices {min_slice}-{max_slice}, center={center_slice}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    if not vertebra_data:
        raise ValueError(f"No vertebrae segmentations found for {subject_label}")

    # Write vertebrae summary TSV
    summary_file = output_dir / f"{subject_label}_vertebrae_slices.tsv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["vertebra", "min_slice", "max_slice", "center_slice", "extent", "num_voxels"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(vertebra_data)

    logger.info(f"Saved vertebrae summary: {summary_file}")

    # Write slice lookup TSV
    lookup_file = output_dir / f"{subject_label}_slice_vertebrae.tsv"
    with open(lookup_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["slice", "vertebrae"])
        for slice_idx in sorted(slice_to_vertebrae.keys()):
            vertebrae_list = ",".join(slice_to_vertebrae[slice_idx])
            writer.writerow([slice_idx, vertebrae_list])

    logger.info(f"Saved slice lookup: {lookup_file}")

    return summary_file, lookup_file


def create_vertebrae_slice_reports(
    bids_root: Path,
    subjects: Optional[Union[str, List[str]]] = None,
    session_label: Optional[str] = None,
    **kwargs,
) -> Dict[str, Tuple[Path, Path]]:
    """
    Create vertebrae slice reports for multiple subjects.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subjects : str or list, optional
        Subject(s) to process. None = all subjects.
    session_label : str, optional
        Session label
    **kwargs
        Additional arguments passed to create_vertebrae_slice_report

    Returns
    -------
    dict
        Mapping of subject_label to (summary_file, lookup_file) tuple
    """
    bids_root = Path(bids_root)
    if subjects is None:
        subject_dirs = sorted([d for d in bids_root.glob("sub-*") if d.is_dir()])
        subject_labels = [d.name for d in subject_dirs]
    elif isinstance(subjects, str):
        subject_labels = [subjects]
    else:
        subject_labels = subjects

    results: Dict[str, Tuple[Path, Path]] = {}
    for subject_label in subject_labels:
        try:
            output = create_vertebrae_slice_report(
                bids_root=bids_root,
                subject_label=subject_label,
                session_label=session_label,
                **kwargs,
            )
            results[subject_label] = output
            logger.info(f"✓ Generated reports for {subject_label}")
        except Exception as e:
            logger.error(f"✗ Failed reports for {subject_label}: {e}")

    return results
