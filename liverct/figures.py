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

# Organ-specific colors
ORGAN_COLORS = {
    "liver": "#8B0000",        # dark red
    "pancreas": "#FFD700",     # gold
    "spleen": "#9370DB",       # medium purple
    "kidneys": "#FF69B4",      # hot pink
    "lungs": "#87CEEB",        # sky blue
    "prostate": "#FF8C00",     # dark orange
    "heart": "#DC143C",        # crimson
}

# Sub-segmentation colors (reusable across different organs)
# These are distinct, perceptually-uniform colors suitable for categorical data
SUBSEGMENT_COLORS = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#FFFF33",  # yellow
    "#A65628",  # brown
    "#F781BF",  # pink
    "#999999",  # grey
    "#66C2A5",  # teal
    "#FC8D62",  # salmon
    "#8DA0CB",  # lavender
]

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


def find_ct_matching_segmentation(
    bids_root: Path,
    subject_label: str,
    reference_seg_path: Path,
    session_label: Optional[str] = None,
) -> Optional[Path]:
    """
    Find CT file that matches the dimensions/affine of a segmentation file.
    
    First checks for a source.json metadata file in the segmentation directory.
    If not found, falls back to dimension matching across all CT files.
    
    This ensures we get the same CT that was used to create the segmentation,
    which is critical when subjects have multiple CT series.
    
    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subject_label : str
        Subject label
    reference_seg_path : Path
        Path to segmentation file to match
    session_label : str, optional
        Session label
        
    Returns
    -------
    Path or None
        CT file path that matches segmentation dimensions
    """
    subject_label = _normalize_subject_label(subject_label)
    session_label = _normalize_session_label(session_label)
    
    # First, try to find source.json metadata file
    # reference_seg_path should be in a directory like .../total/liver.nii.gz
    # source.json should be in .../total/source.json
    seg_dir = reference_seg_path.parent
    source_metadata_file = seg_dir / "source.json"
    
    if source_metadata_file.exists():
        try:
            import json
            with open(source_metadata_file) as f:
                metadata = json.load(f)
            
            # Get source file path
            source_file = Path(metadata.get("source_file", ""))
            if source_file.exists():
                logger.info(f"  Using source CT from metadata: {source_file.name}")
                return source_file
            else:
                logger.warning(
                    f"  Source CT in metadata not found: {source_file}, "
                    f"falling back to dimension matching"
                )
        except Exception as e:
            logger.warning(f"  Could not read source metadata: {e}, falling back to dimension matching")
    
    # Fall back to dimension matching
    logger.info("  Finding CT by dimension matching...")
    
    # Load reference segmentation to get shape/affine
    ref_img = nib.load(str(reference_seg_path))
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine
    
    # Find all CT files for this subject
    subject_dir = Path(bids_root) / subject_label
    ct_files = []
    
    if session_label:
        ct_dir = subject_dir / session_label / "ct"
        if ct_dir.exists():
            ct_files.extend(ct_dir.glob("*.nii*"))
    else:
        ct_dir = subject_dir / "ct"
        if ct_dir.exists():
            ct_files.extend(ct_dir.glob("*.nii*"))
        
        # Also check session directories
        for ses_dir in sorted(subject_dir.glob("ses-*")):
            ct_dir = ses_dir / "ct"
            if ct_dir.exists():
                ct_files.extend(ct_dir.glob("*.nii*"))
    
    # Find CT with matching dimensions
    for ct_file in ct_files:
        try:
            ct_img = nib.load(str(ct_file))
            if ct_img.shape == ref_shape and np.allclose(ct_img.affine, ref_affine):
                logger.info(f"  Matched CT by dimensions: {ct_file.name}")
                return ct_file
        except Exception:
            continue
    
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


def _load_organ_masks(
    bids_root: Path,
    subject_label: str,
    organs: Union[List[str], str],
    ct_shape: Tuple[int, ...],
    session_label: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Load organ masks from total directory.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subject_label : str
        Subject label
    organs : str or list
        "all" or list of specific organs
    ct_shape : tuple
        Reference CT shape for validation
    session_label : str, optional
        Session label

    Returns
    -------
    dict
        Mapping of organ name to mask array
    """
    # Define available organs and their file patterns
    organ_file_patterns = {
        "liver": ["liver.nii.gz"],
        "pancreas": ["pancreas.nii.gz"],
        "spleen": ["spleen.nii.gz"],
        "kidneys": ["kidney_left.nii.gz", "kidney_right.nii.gz"],
        "lungs": [
            "lung_upper_lobe_left.nii.gz",
            "lung_lower_lobe_left.nii.gz",
            "lung_upper_lobe_right.nii.gz",
            "lung_middle_lobe_right.nii.gz",
            "lung_lower_lobe_right.nii.gz",
        ],
        "prostate": ["prostate.nii.gz"],
        "heart": ["heart.nii.gz"],
    }

    # Determine which organs to load
    if organs == "all":
        organs_to_load = list(organ_file_patterns.keys())
    elif isinstance(organs, str):
        organs_to_load = [organs]
    else:
        organs_to_load = list(organs)

    # Find total directory
    total_dir = find_total_dir(bids_root, subject_label, session_label)
    if not total_dir.exists():
        logger.warning(f"Total directory not found: {total_dir}")
        return {}

    organ_masks = {}
    for organ in organs_to_load:
        if organ not in organ_file_patterns:
            logger.warning(f"Unknown organ: {organ}")
            continue

        # Get file patterns for this organ
        patterns = organ_file_patterns[organ]
        organ_files = [total_dir / pattern for pattern in patterns]
        existing_files = [f for f in organ_files if f.exists()]

        if not existing_files:
            logger.debug(f"  No files found for {organ}")
            continue

        # Load and combine masks
        combined_mask = np.zeros(ct_shape, dtype=bool)
        for organ_file in existing_files:
            try:
                organ_img = nib.load(str(organ_file))
                organ_data = organ_img.get_fdata() > 0

                # Validate shape
                if organ_data.shape != ct_shape:
                    logger.warning(
                        f"  Shape mismatch for {organ_file.name}: "
                        f"{organ_data.shape} vs {ct_shape}, clipping"
                    )
                    # Clip to CT shape
                    organ_data_clipped = np.zeros(ct_shape, dtype=bool)
                    x_max = min(organ_data.shape[0], ct_shape[0])
                    y_max = min(organ_data.shape[1], ct_shape[1])
                    z_max = min(organ_data.shape[2], ct_shape[2])
                    organ_data_clipped[:x_max, :y_max, :z_max] = organ_data[:x_max, :y_max, :z_max]
                    organ_data = organ_data_clipped

                combined_mask |= organ_data
            except Exception as e:
                logger.warning(f"  Failed to load {organ_file.name}: {e}")

        if combined_mask.any():
            organ_masks[organ] = combined_mask
            logger.debug(f"  Loaded {organ}")

    return organ_masks


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
    include_organs: Union[List[str], str, None] = None,
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
    include_organs : str, list, or None
        Organs to include as overlays. Options:
        - None: No organs (default)
        - "all": All available organs
        - List of specific organs: ["liver", "pancreas", "spleen"]
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

    # Load organ masks if requested
    if include_organs is not None:
        try:
            ct_img = nib.load(str(ct_file))
            organ_masks = _load_organ_masks(
                bids_root,
                subject_label,
                include_organs,
                ct_img.shape,
                session_label=session_label,
            )
            # Add organs to label_arrays with colors from ORGAN_COLORS
            for organ_name, organ_mask in organ_masks.items():
                label_arrays[organ_name] = organ_mask
                logger.info(f"Added {organ_name} to montage")
        except Exception as e:
            logger.warning(f"Skipping organ overlays for {subject_label}: {e}")

    if output_dir is None:
        # Place figures at subject level, not within tissue_types
        base_dir = Path(bids_root) / "derivatives" / "totalsegmentator" / subject_label
        if session_label:
            base_dir = base_dir / session_label
        output_dir = base_dir / "figures"

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
    
    # Merge organ colors into label colors
    label_colors = dict(DEFAULT_LABEL_COLORS)
    label_colors.update(ORGAN_COLORS)
    
    return create_label_overlay_montage(
        ct_file,
        label_files,
        output_png,
        label_arrays=label_arrays if label_arrays else None,
        label_colors=label_colors,
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
    include_organs: Union[List[str], str, None] = None,
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
    include_organs : str, list, or None
        Organs to include as overlays (None, "all", or list of organs)
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
                include_organs=include_organs,
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


def create_vertebrae_cross_subject_montage(
    bids_root: Path,
    subjects: Union[List[str], str],
    vertebrae: Optional[List[str]] = None,
    num_vertebrae: int = 7,
    session_label: Optional[str] = None,
    output_dir: Optional[Path] = None,
    series_description_pattern: Optional[str] = None,
    include_skeletal: bool = True,
    include_organs: Union[List[str], str, None] = None,
    window: Tuple[int, int] = (-200, 250),
    alpha: float = 0.35,
    dpi: int = 200,
    label_colors: Optional[Dict[str, str]] = None,
    label_arrays_fn: Optional[callable] = None,
) -> Path:
    """
    Create a cross-subject montage with subjects in rows and vertebrae in columns.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subjects : str or list of str
        Single subject or list of subjects to include
    vertebrae : list, optional
        Specific vertebrae to include (e.g., ["C5", "T1", "T6", "T12", "L3", "L5"]).
        If None, selects evenly-spaced vertebrae across the spine.
    num_vertebrae : int
        Number of vertebrae to show if vertebrae list not provided. Default: 7.
    session_label : str, optional
        Session label
    output_dir : Path, optional
        Output directory for montage
    series_description_pattern : str, optional
        Pattern to match CT series
    include_skeletal : bool
        Include skeletal composite overlay
    include_organs : str, list, or None
        Organs to include as overlays (None, "all", or list of organs)
    window : tuple
        HU window (min, max)
    alpha : float
        Overlay alpha
    dpi : int
        Output DPI
    label_colors : dict, optional
        Custom label colors
    label_arrays_fn : callable, optional
        Function to create additional label arrays (e.g., for skeletal composite)

    Returns
    -------
    Path
        Output montage file path
    """
    import csv

    bids_root = Path(bids_root)
    
    # Merge default colors with organ colors
    if label_colors is None:
        label_colors = dict(DEFAULT_LABEL_COLORS)
        label_colors.update(ORGAN_COLORS)
    
    # Normalize subjects list
    if isinstance(subjects, str):
        subject_list = [subjects]
    else:
        subject_list = list(subjects)

    # Normalize subject labels
    subject_list = [s if s.startswith("sub-") else f"sub-{s}" for s in subject_list]

    logger.info(f"Creating cross-subject montage for {len(subject_list)} subjects")

    # Determine which vertebrae to display
    if vertebrae is None:
        # Read vertebrae from first subject's summary file
        summary_file = (
            Path(bids_root)
            / "derivatives"
            / "totalsegmentator"
            / subject_list[0]
        )
        if session_label:
            session_label_norm = session_label if session_label.startswith("ses-") else f"ses-{session_label}"
            summary_file = summary_file / session_label_norm
        summary_file = summary_file / f"{subject_list[0]}_vertebrae_slices.tsv"

        if not summary_file.exists():
            raise FileNotFoundError(
                f"Vertebrae summary file not found: {summary_file}. "
                f"Please run vertebrae report generation first."
            )

        # Read all vertebrae from summary
        all_vertebrae = []
        with open(summary_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                all_vertebrae.append(row["vertebra"])

        # Select evenly-spaced vertebrae
        if len(all_vertebrae) > num_vertebrae:
            indices = np.linspace(0, len(all_vertebrae) - 1, num_vertebrae, dtype=int)
            vertebrae = [all_vertebrae[i] for i in indices]
        else:
            vertebrae = all_vertebrae
            logger.warning(
                f"Only {len(all_vertebrae)} vertebrae available, "
                f"requested {num_vertebrae}"
            )

    logger.info(f"Displaying vertebrae: {', '.join(vertebrae)}")

    # Load CT data and tissue type masks for each subject
    # Structure: subject_data[subject][vertebra] = (ct_slice, masks_dict)
    subject_data: Dict[str, Dict[str, Tuple[np.ndarray, Dict]]] = {}

    for subject_label in subject_list:
        subject_data[subject_label] = {}
        logger.info(f"Loading data for {subject_label}")

        # Find CT file
        ct_file = find_ct_nifti(
            bids_root,
            subject_label,
            session_label=session_label,
            series_description_pattern=series_description_pattern,
        )
        if not ct_file:
            logger.warning(f"  CT not found for {subject_label}, skipping")
            continue

        # Load CT data once
        ct_img = nib.load(str(ct_file))
        ct_data = ct_img.get_fdata().astype(np.float32)

        # Load tissue type masks
        seg_dir = find_tissue_types_dir(bids_root, subject_label, session_label=session_label)
        label_files = find_tissue_type_label_files(seg_dir)

        label_masks: Dict[str, np.ndarray] = {}
        for label, mask_path in label_files.items():
            try:
                mask_img = nib.load(str(mask_path))
                if mask_img.shape != ct_data.shape:
                    logger.debug(f"  Shape mismatch for {label}, skipping")
                    continue
                label_masks[label] = mask_img.get_fdata() > 0
            except Exception as e:
                logger.debug(f"  Failed to load {label}: {e}")

        # Add skeletal composite if requested
        if include_skeletal:
            try:
                skeletal_file = seg_dir / "skeletal_composite.nii.gz"
                if skeletal_file.exists():
                    skeletal_img = nib.load(str(skeletal_file))
                    if skeletal_img.shape == ct_data.shape:
                        skeletal_mask = skeletal_img.get_fdata() > 0
                        if skeletal_mask.any():
                            label_masks["Skeletal"] = skeletal_mask
            except Exception as e:
                logger.debug(f"  Failed to load skeletal composite: {e}")

        # Add organ masks if requested
        if include_organs is not None:
            try:
                organ_masks = _load_organ_masks(
                    bids_root,
                    subject_label,
                    include_organs,
                    ct_data.shape,
                    session_label=session_label,
                )
                label_masks.update(organ_masks)
            except Exception as e:
                logger.debug(f"  Failed to load organ masks: {e}")

        # Get vertebrae center slices
        vertebrae_file = (
            Path(bids_root)
            / "derivatives"
            / "totalsegmentator"
            / subject_label
        )
        if session_label:
            session_label_norm = session_label if session_label.startswith("ses-") else f"ses-{session_label}"
            vertebrae_file = vertebrae_file / session_label_norm
        vertebrae_file = vertebrae_file / f"{subject_label}_vertebrae_slices.tsv"

        if not vertebrae_file.exists():
            logger.warning(f"  Vertebrae file not found for {subject_label}, skipping")
            continue

        # Read vertebrae center slices
        vertebra_centers = {}
        with open(vertebrae_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                vertebra_centers[row["vertebra"]] = int(row["center_slice"])

        # Extract slices for each vertebra
        for vertebra in vertebrae:
            if vertebra not in vertebra_centers:
                logger.debug(f"  Vertebra {vertebra} not found for {subject_label}")
                continue

            slice_idx = vertebra_centers[vertebra]
            ct_slice = ct_data[:, :, slice_idx]
            masks = {k: v[:, :, slice_idx] for k, v in label_masks.items()}

            subject_data[subject_label][vertebra] = (ct_slice, masks)

    if not subject_data or all(not v for v in subject_data.values()):
        raise ValueError("No data loaded for any subject/vertebra combination")

    # Create montage grid
    num_rows = len(subject_list)
    num_cols = len(vertebrae)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 3.5), dpi=dpi
    )

    # Handle single row/column cases
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = np.atleast_2d(axes)

    vmin, vmax = window

    # Fill grid
    for row_idx, subject_label in enumerate(subject_list):
        for col_idx, vertebra in enumerate(vertebrae):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            # Get data if available
            if subject_label not in subject_data or vertebra not in subject_data[subject_label]:
                # Empty cell
                ax.text(
                    0.5,
                    0.5,
                    f"N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )
                continue

            ct_slice, masks = subject_data[subject_label][vertebra]

            # Window and normalize
            ct_slice = np.clip(ct_slice, vmin, vmax)
            ct_slice = (ct_slice - vmin) / float(vmax - vmin)

            # Display CT
            ax.imshow(ct_slice.T, cmap="gray", origin="lower")

            # Create RGBA overlay
            overlay = np.zeros((*ct_slice.shape, 4), dtype=np.float32)
            for label, mask_slice in masks.items():
                if label not in label_colors:
                    continue
                color = mcolors.to_rgba(label_colors[label], alpha=alpha)
                overlay[mask_slice.T > 0] = color

            ax.imshow(overlay, origin="lower")

        # Row label (subject)
        subject_id = subject_list[row_idx].replace("sub-", "")
        axes[row_idx, 0].text(
            -0.15,
            0.5,
            subject_id,
            transform=axes[row_idx, 0].transAxes,
            fontsize=10,
            weight="bold",
            ha="right",
            va="center",
        )

    # Column labels (vertebrae)
    for col_idx, vertebra in enumerate(vertebrae):
        axes[0, col_idx].text(
            0.5,
            1.05,
            vertebra,
            transform=axes[0, col_idx].transAxes,
            fontsize=10,
            weight="bold",
            ha="center",
            va="bottom",
        )

    # Legend
    legend_labels = [
        label for labels in subject_data.values()
        for label in labels.get(vertebrae[0], (None, {}))[1].keys()
    ]
    legend_labels = list(dict.fromkeys(legend_labels))  # Remove duplicates, preserve order

    legend_patches = [
        Patch(color=label_colors[label], label=label)
        for label in legend_labels
        if label in label_colors
    ]

    if legend_patches:
        fig.legend(handles=legend_patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))

    # Determine output path
    if output_dir is None:
        output_dir = Path(bids_root) / "derivatives" / "totalsegmentator" / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    subject_prefix = "_".join([s.replace("sub-", "") for s in subject_list[:3]])
    if len(subject_list) > 3:
        subject_prefix += f"+{len(subject_list)-3}more"
    vertebra_range = f"{vertebrae[0]}-{vertebrae[-1]}"

    output_name = f"{subject_prefix}_vertebrae_{vertebra_range}_cross_subject.png"
    output_png = output_dir / output_name

    fig.tight_layout(rect=[0.1, 0.05, 1, 1])
    fig.savefig(output_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    logger.info(f"Saved cross-subject montage: {output_png}")
    return output_png


def create_organ_montage(
    bids_root: Path,
    subject_label: str,
    organ: str,
    session_label: Optional[str] = None,
    num_slices: int = 12,
    output_dir: Optional[Path] = None,
    series_description_pattern: Optional[str] = None,
    include_skeletal: bool = True,
    window: Tuple[int, int] = (-200, 250),
    alpha: float = 0.35,
    dpi: int = 200,
    label_colors: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Create organ-specific montage with Z extent determined by organ mask.

    Automatically determines the superior and inferior extent of the organ
    and creates a montage spanning that range.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subject_label : str
        Subject label (with or without "sub-" prefix)
    organ : str
        Organ name (e.g., "liver", "pancreas", "spleen", "kidneys",
        "lungs", "prostate", "heart")
    session_label : str, optional
        Session label
    num_slices : int
        Number of slices in montage
    output_dir : Path, optional
        Output directory for montage
    series_description_pattern : str, optional
        Pattern to match CT series
    include_skeletal : bool
        Include skeletal composite overlay
    window : tuple
        HU window (min, max)
    alpha : float
        Overlay alpha
    dpi : int
        Output DPI
    label_colors : dict, optional
        Custom label colors (will be merged with defaults and organ color)

    Returns
    -------
    Path
        Output montage file path
    """
    bids_root = Path(bids_root)
    subject_label = subject_label if subject_label.startswith("sub-") else f"sub-{subject_label}"

    # Setup colors
    colors = dict(DEFAULT_LABEL_COLORS)
    if label_colors:
        colors.update(label_colors)

    # Add organ to colors if not already present
    organ_display = organ.replace("_", " ")
    if organ not in colors:
        if organ in ORGAN_COLORS:
            colors[organ] = ORGAN_COLORS[organ]
        else:
            # Fallback color
            colors[organ] = "#808080"  # gray

    logger.info(f"Creating organ montage: {subject_label} - {organ}")

    # Find organ mask file(s) in total directory first
    total_dir = find_total_dir(bids_root, subject_label, session_label)
    
    # Map organ names to file patterns (can be list for multi-part organs)
    organ_file_patterns = {
        "liver": ["liver.nii.gz"],
        "pancreas": ["pancreas.nii.gz"],
        "spleen": ["spleen.nii.gz"],
        "kidneys": ["kidney_left.nii.gz", "kidney_right.nii.gz"],
        "lungs": [
            "lung_upper_lobe_left.nii.gz",
            "lung_lower_lobe_left.nii.gz",
            "lung_upper_lobe_right.nii.gz",
            "lung_middle_lobe_right.nii.gz",
            "lung_lower_lobe_right.nii.gz",
        ],
        "prostate": ["prostate.nii.gz"],
        "heart": ["heart.nii.gz"],
    }

    if organ not in organ_file_patterns:
        raise ValueError(f"Unknown organ: {organ}. Available: {list(organ_file_patterns.keys())}")

    # Get list of files for this organ
    organ_files = [total_dir / pattern for pattern in organ_file_patterns[organ]]
    
    # Check which files exist
    existing_files = [f for f in organ_files if f.exists()]
    if not existing_files:
        raise FileNotFoundError(
            f"No organ segmentation files found for {organ}. "
            f"Expected: {[f.name for f in organ_files]}"
        )
    
    if len(existing_files) < len(organ_files):
        missing = [f.name for f in organ_files if f not in existing_files]
        logger.warning(f"  Some {organ} files missing: {missing}")
    
    logger.info(f"  Found {len(existing_files)} file(s) for {organ}")

    # Use first file as reference for CT matching
    reference_file = existing_files[0]

    # Find CT file that matches the segmentation dimensions
    # This ensures we use the same CT that was used for segmentation
    ct_file = find_ct_matching_segmentation(
        bids_root,
        subject_label,
        reference_seg_path=reference_file,
        session_label=session_label,
    )
    
    # Fall back to pattern matching if dimension matching fails
    if not ct_file:
        logger.warning(
            f"  Could not find CT matching segmentation dimensions, "
            f"falling back to pattern matching"
        )
        ct_file = find_ct_nifti(
            bids_root,
            subject_label,
            session_label=session_label,
            series_description_pattern=series_description_pattern,
        )
    
    if not ct_file:
        raise FileNotFoundError(f"CT not found for {subject_label}")
    
    logger.info(f"  Using CT: {ct_file.name}")
    
    # Load CT data
    ct_img = nib.load(str(ct_file))
    ct_data = ct_img.get_fdata().astype(np.float32)

    # Load and combine organ masks
    organ_mask = np.zeros(ct_data.shape, dtype=bool)
    for organ_file in existing_files:
        logger.info(f"    Loading: {organ_file.name}")
        organ_img = nib.load(str(organ_file))
        part_mask = organ_img.get_fdata() > 0
        
        # Validate shapes match
        if part_mask.shape != ct_data.shape:
            logger.warning(
                f"    Shape mismatch for {organ_file.name}: "
                f"{part_mask.shape} vs CT {ct_data.shape}, skipping"
            )
            continue
        
        # Combine with OR operation
        organ_mask |= part_mask
    
    # Check if combined mask is empty
    if not organ_mask.any():
        raise ValueError(f"Combined {organ} mask is empty after loading all parts")

    # Find Z extent (axis 2 for axial)
    axes = (0, 1)  # Sum over X and Y to get Z extent
    z_sums = organ_mask.sum(axis=axes)
    z_indices = np.where(z_sums > 0)[0]

    if z_indices.size == 0:
        raise ValueError(f"Organ mask is empty: {organ}")

    z_min = int(z_indices.min())
    z_max = int(z_indices.max())
    
    # Clip to CT data bounds
    z_min = max(0, z_min)
    z_max = min(ct_data.shape[2] - 1, z_max)
    
    logger.info(f"  Organ extent: slices {z_min}-{z_max} (CT shape: {ct_data.shape})")

    # Select slices within organ extent
    slice_indices = np.linspace(z_min, z_max, num_slices, dtype=int)
    slice_indices = np.unique(slice_indices)

    # Load tissue type masks
    seg_dir = find_tissue_types_dir(bids_root, subject_label, session_label=session_label)
    label_files = find_tissue_type_label_files(seg_dir)

    label_masks: Dict[str, np.ndarray] = {}
    for label, mask_path in label_files.items():
        try:
            mask_img = nib.load(str(mask_path))
            if mask_img.shape != ct_data.shape:
                logger.debug(f"  Shape mismatch for {label}, skipping")
                continue
            label_masks[label] = mask_img.get_fdata() > 0
        except Exception as e:
            logger.debug(f"  Failed to load {label}: {e}")

    # Add skeletal composite if requested
    if include_skeletal:
        try:
            skeletal_file = seg_dir / "skeletal_composite.nii.gz"
            if skeletal_file.exists():
                skeletal_img = nib.load(str(skeletal_file))
                if skeletal_img.shape == ct_data.shape:
                    skeletal_mask = skeletal_img.get_fdata() > 0
                    if skeletal_mask.any():
                        label_masks["Skeletal"] = skeletal_mask
        except Exception as e:
            logger.debug(f"  Failed to load skeletal composite: {e}")

    # Add organ mask
    # Ensure organ_mask matches CT shape exactly
    if organ_mask.shape != ct_data.shape:
        # Pad or trim to match
        organ_mask_fixed = np.zeros(ct_data.shape, dtype=organ_mask.dtype)
        x_max = min(organ_mask.shape[0], ct_data.shape[0])
        y_max = min(organ_mask.shape[1], ct_data.shape[1])
        z_max = min(organ_mask.shape[2], ct_data.shape[2])
        organ_mask_fixed[:x_max, :y_max, :z_max] = organ_mask[:x_max, :y_max, :z_max]
        label_masks[organ] = organ_mask_fixed
    else:
        label_masks[organ] = organ_mask

    # Create montage
    cols = 4
    rows = int(np.ceil(len(slice_indices) / cols))

    vmin, vmax = window

    fig = None
    try:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=dpi)
        axes = np.atleast_2d(axes)

        for i, slice_idx in enumerate(slice_indices):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.axis("off")

            ct_slice = ct_data[:, :, slice_idx]
            masks = {k: v[:, :, slice_idx] for k, v in label_masks.items()}

            # Window and normalize
            ct_slice = np.clip(ct_slice, vmin, vmax)
            ct_slice = (ct_slice - vmin) / float(vmax - vmin)

            ax.imshow(ct_slice.T, cmap="gray", origin="lower")

            # Create RGBA overlay
            overlay = np.zeros((*ct_slice.shape, 4), dtype=np.float32)
            
            # Draw tissue types first (background), then organ on top
            for label in ["SAT", "VAT", "Muscle", "Skeletal"]:
                if label in masks and label in colors:
                    color = mcolors.to_rgba(colors[label], alpha=alpha)
                    overlay[masks[label].T > 0] = color

            # Draw organ with higher alpha on top
            if organ in masks and organ in colors:
                organ_alpha = min(1.0, alpha * 1.5)  # Slightly more opaque
                color = mcolors.to_rgba(colors[organ], alpha=organ_alpha)
                overlay[masks[organ].T > 0] = color

            ax.imshow(overlay, origin="lower")
            ax.set_title(f"Slice {slice_idx}")

        # Turn off unused axes
        for j in range(len(slice_indices), rows * cols):
            r, c = divmod(j, cols)
            axes[r, c].axis("off")

        # Legend
        legend_labels = [
            l for l in ["SAT", "VAT", "Muscle", "Skeletal", organ]
            if l in label_masks
        ]
        legend_patches = [
            Patch(color=colors[label], label=label.replace("_", " "))
            for label in legend_labels
            if label in colors
        ]

        if legend_patches:
            fig.legend(handles=legend_patches, loc="lower center", ncol=5)

        # Determine output path
        if output_dir is None:
            # Place figures at subject level alongside tissue_types/ and total/
            base_dir = Path(bids_root) / "derivatives" / "totalsegmentator" / subject_label
            if session_label:
                session_label_norm = session_label if session_label.startswith("ses-") else f"ses-{session_label}"
                base_dir = base_dir / session_label_norm
            output_dir = base_dir / "figures"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        subject_id = subject_label.replace("sub-", "")
        output_name = f"{subject_id}_{organ}_montage.png"
        output_png = output_dir / output_name

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(output_png, bbox_inches="tight")

        logger.info(f"  ✓ Saved organ montage: {output_png}")
        return output_png
    finally:
        # Always close figure to prevent memory leak
        if fig is not None:
            plt.close(fig)


def create_organ_montages(
    bids_root: Path,
    subject_label: str,
    organs: Union[List[str], str] = "all",
    session_label: Optional[str] = None,
    **kwargs,
) -> Dict[str, Path]:
    """
    Create organ-specific montages for a single subject.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subject_label : str
        Subject label
    organs : str or list, optional
        Organs to generate montages for. Options:
        - "all": liver, pancreas, spleen, kidneys, lungs, prostate, heart
        - List of specific organs: ["liver", "pancreas", "spleen"]
        Default: "all"
    session_label : str, optional
        Session label
    **kwargs
        Additional arguments passed to create_organ_montage

    Returns
    -------
    dict
        Mapping of organ name to output file path
    """
    # Define all available organs
    all_organs = [
        "liver",
        "pancreas",
        "spleen",
        "kidneys",
        "lungs",
        "prostate",
        "heart",
    ]

    # Determine which organs to process
    if organs == "all":
        organs_to_process = all_organs
    elif isinstance(organs, str):
        organs_to_process = [organs]
    else:
        organs_to_process = list(organs)

    logger.info(f"Creating organ montages for {subject_label}: {organs_to_process}")

    results = {}
    for organ in organs_to_process:
        try:
            output = create_organ_montage(
                bids_root=bids_root,
                subject_label=subject_label,
                organ=organ,
                session_label=session_label,
                **kwargs,
            )
            results[organ] = output
        except Exception as e:
            logger.error(f"  ✗ Failed to create {organ} montage: {e}")

    return results


def create_liver_segments_montage(
    bids_root: Path,
    subject_label: str,
    session_label: Optional[str] = None,
    num_slices: int = 12,
    output_dir: Optional[Path] = None,
    series_description_pattern: Optional[str] = None,
    include_tissue: bool = True,
    window: Tuple[int, int] = (-200, 250),
    alpha: float = 0.35,
    dpi: int = 200,
) -> Path:
    """
    Create montage showing liver with its 8 sub-segments color-coded.

    Parameters
    ----------
    bids_root : Path
        Root BIDS directory
    subject_label : str
        Subject label
    session_label : str, optional
        Session label
    num_slices : int, optional
        Number of slices to display (default: 12)
    output_dir : Path, optional
        Output directory for PNG file
    series_description_pattern : str, optional
        Pattern to match CT series description
    include_tissue : bool, optional
        Whether to include tissue overlays (SAT, VAT, Muscle, Skeletal)
        Default: True
    window : tuple, optional
        HU window for display (default: (-200, 250))
    alpha : float, optional
        Transparency for overlays (0-1, default: 0.35)
    dpi : int, optional
        Output resolution (default: 200)

    Returns
    -------
    Path
        Output PNG file path
    """
    fig = None
    try:
        # Normalize labels
        subject_label = _normalize_subject_label(subject_label)
        session_label = _normalize_session_label(session_label)

        logger.info(f"Creating liver segments montage for {subject_label}")

        # Build paths
        derivatives_dir = Path(bids_root) / "derivatives" / "totalsegmentator" / subject_label
        if session_label:
            derivatives_dir = derivatives_dir / session_label

        # Find liver segment files
        liver_seg_dir = derivatives_dir / "liver_segments"
        if not liver_seg_dir.exists():
            raise FileNotFoundError(f"Liver segments directory not found: {liver_seg_dir}")

        # Load all 8 liver segments
        segment_files = sorted(liver_seg_dir.glob("liver_segment_*.nii.gz"))
        if not segment_files:
            raise FileNotFoundError(f"No liver segment files found in {liver_seg_dir}")

        logger.info(f"  Found {len(segment_files)} liver segments")

        # Use first segment file to find matching CT
        first_seg_file = segment_files[0]
        ct_file = find_ct_matching_segmentation(
            bids_root=bids_root,
            subject_label=subject_label,
            reference_seg_path=first_seg_file,
            session_label=session_label,
        )

        # Load CT
        ct_nifti = nib.load(ct_file)
        ct_data = ct_nifti.get_fdata()

        # Load all segments and assign colors
        segment_masks = {}
        for i, seg_file in enumerate(segment_files):
            # Extract segment number from filename (handle .nii.gz double extension)
            # liver_segment_1.nii.gz -> extract "1"
            filename = seg_file.name.replace(".nii.gz", "").replace(".nii", "")
            seg_num = int(filename.split("_")[-1])
            seg_nifti = nib.load(seg_file)
            seg_data = seg_nifti.get_fdata()

            # Validate and clip shape
            if seg_data.shape != ct_data.shape:
                logger.warning(f"  Shape mismatch for {seg_file.name}: {seg_data.shape} vs CT {ct_data.shape}")
                seg_data = seg_data[:ct_data.shape[0], :ct_data.shape[1], :ct_data.shape[2]]

            segment_masks[f"Segment {seg_num}"] = seg_data > 0

        # Load tissue overlays if requested
        tissue_masks = {}
        if include_tissue:
            tissue_dir = derivatives_dir / "tissue_types"
            if tissue_dir.exists():
                for tissue_name, patterns in LABEL_PATTERNS.items():
                    for pattern in patterns:
                        tissue_files = list(tissue_dir.glob(f"*{pattern}*.nii.gz"))
                        if tissue_files:
                            tissue_nifti = nib.load(tissue_files[0])
                            tissue_data = tissue_nifti.get_fdata()
                            if tissue_data.shape != ct_data.shape:
                                tissue_data = tissue_data[:ct_data.shape[0], :ct_data.shape[1], :ct_data.shape[2]]
                            tissue_masks[tissue_name] = tissue_data > 0
                            break

                # Add skeletal if present
                if "Skeletal" not in tissue_masks:
                    total_dir = derivatives_dir / "total"
                    if total_dir.exists():
                        all_bone_mask = None
                        for bone_pattern in TOTAL_BONE_PATTERNS:
                            bone_files = list(total_dir.glob(f"*{bone_pattern}*.nii.gz"))
                            for bone_file in bone_files:
                                bone_nifti = nib.load(bone_file)
                                bone_data = bone_nifti.get_fdata()
                                if bone_data.shape != ct_data.shape:
                                    bone_data = bone_data[:ct_data.shape[0], :ct_data.shape[1], :ct_data.shape[2]]
                                bone_mask = bone_data > 0
                                if all_bone_mask is None:
                                    all_bone_mask = bone_mask
                                else:
                                    all_bone_mask = all_bone_mask | bone_mask
                        if all_bone_mask is not None:
                            tissue_masks["Skeletal"] = all_bone_mask

        # Find liver extent
        all_liver_mask = np.zeros_like(ct_data, dtype=bool)
        for seg_mask in segment_masks.values():
            all_liver_mask = all_liver_mask | seg_mask

        z_indices = np.where(np.any(all_liver_mask, axis=(0, 1)))[0]
        if len(z_indices) == 0:
            raise ValueError("No liver voxels found in segmentation")

        z_min, z_max = z_indices[0], z_indices[-1]
        z_slices = np.linspace(z_min, z_max, num_slices, dtype=int)

        # Create figure
        fig, axes = plt.subplots(1, num_slices, figsize=(num_slices * 2, 2), dpi=dpi)
        if num_slices == 1:
            axes = [axes]

        for idx, (ax, z) in enumerate(zip(axes, z_slices)):
            ct_slice = ct_data[:, :, z].T
            ct_slice_display = np.clip(ct_slice, window[0], window[1])
            ct_slice_norm = (ct_slice_display - window[0]) / (window[1] - window[0])

            ax.imshow(ct_slice_norm, cmap="gray", origin="lower", aspect="auto")

            # Overlay tissue types
            if include_tissue and tissue_masks:
                for tissue_name, tissue_mask in tissue_masks.items():
                    tissue_slice = tissue_mask[:, :, z].T
                    if np.any(tissue_slice):
                        color = DEFAULT_LABEL_COLORS.get(tissue_name, "#FFFFFF")
                        overlay = np.zeros((*tissue_slice.shape, 4))
                        rgb = mcolors.to_rgb(color)
                        overlay[tissue_slice] = [*rgb, alpha]
                        ax.imshow(overlay, origin="lower", aspect="auto")

            # Overlay liver segments with distinct colors
            for seg_idx, (seg_name, seg_mask) in enumerate(segment_masks.items()):
                seg_slice = seg_mask[:, :, z].T
                if np.any(seg_slice):
                    color = SUBSEGMENT_COLORS[seg_idx % len(SUBSEGMENT_COLORS)]
                    overlay = np.zeros((*seg_slice.shape, 4))
                    rgb = mcolors.to_rgb(color)
                    overlay[seg_slice] = [*rgb, alpha + 0.15]  # Slightly more opaque for segments
                    ax.imshow(overlay, origin="lower", aspect="auto")

            ax.set_title(f"{z}", fontsize=8, pad=2)
            ax.axis("off")

        # Create legend
        legend_patches = []
        
        # Add tissue types to legend
        if include_tissue and tissue_masks:
            for tissue_name in ["SAT", "VAT", "Muscle", "Skeletal"]:
                if tissue_name in tissue_masks:
                    color = DEFAULT_LABEL_COLORS[tissue_name]
                    legend_patches.append(Patch(color=color, label=tissue_name))

        # Add liver segments to legend
        for seg_idx, seg_name in enumerate(sorted(segment_masks.keys())):
            color = SUBSEGMENT_COLORS[seg_idx % len(SUBSEGMENT_COLORS)]
            legend_patches.append(Patch(color=color, label=seg_name))

        if legend_patches:
            fig.legend(handles=legend_patches, loc="lower center", ncol=6)

        # Determine output path
        if output_dir is None:
            base_dir = Path(bids_root) / "derivatives" / "totalsegmentator" / subject_label
            if session_label:
                base_dir = base_dir / session_label
            output_dir = base_dir / "figures"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        subject_id = subject_label.replace("sub-", "")
        output_name = f"{subject_id}_liver_segments_montage.png"
        output_png = output_dir / output_name

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(output_png, bbox_inches="tight")

        logger.info(f"  ✓ Saved liver segments montage: {output_png}")
        return output_png
    finally:
        # Always close figure to prevent memory leak
        if fig is not None:
            plt.close(fig)
