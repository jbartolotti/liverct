# liverct

`liverct` is a Python package for end-to-end CT processing:

1. Convert CT DICOM data to a BIDS-style dataset via [dcm2bids4ct](https://github.com/ChristianHinge/dcm2bids4ct)
2. Run [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/) tasks and produce per-label statistics
3. Consolidate participant-level results and generate figures


## Installation (conda environment)

### 1. Create and activate environment

```bash
conda create -n liverct python=3.10 -y
conda activate liverct
```

### 2. Install core dependencies

```bash
pip install --upgrade pip setuptools wheel build uv_build
pip install numpy scipy nibabel matplotlib pytest TotalSegmentator
pip install --upgrade git+https://github.com/ChristianHinge/dcm2bids4ct.git
pip install --upgrade git+https://github.com/jbartolotti/liverct.git
```

Notes:

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/) — some tasks require a license key (passed as `license_number`). Request a non-commercial license at [https://backend.totalsegmentator.com/license-academic/](https://backend.totalsegmentator.com/license-academic/)

## Minimal end-to-end workflow

### Step 1: Convert raw DICOM folders to BIDS

```python
from pathlib import Path
from liverct import convert_dicom_directory_to_bids

raw_data_dir = Path("/path/to/raw_ct_data")
bids_root = Path("/path/to/bids_dataset")

results = convert_dicom_directory_to_bids(
	raw_data_dir=raw_data_dir,
	bids_root=bids_root,
	dicom_subdir="DICOM",  # change if your folder name differs
)
print(results)
```

### Step 2: Run segmentation for all subjects

```python
from liverct import BIDSProcessingPipeline

bids_root = Path("/path/to/bids_dataset")
pipeline = BIDSProcessingPipeline(bids_root)

results = pipeline.segment_all_subjects(
    tasks=["total", "tissue_types", "liver_segments"],
    license_number=None,   # required for select subtasks, see https://github.com/wasserth/TotalSegmentator/?tab=readme-ov-file#subtasks
)
print(results)
```

All `sub-*` folders in the BIDS root are discovered and processed automatically. Sessions (`ses-*`) are detected and iterated automatically when present. To limit processing to specific subjects, pass `subjects=["sub-001", "sub-002"]`. 

You can append additional advanced settings that are passed through to TotalSegmentator https://github.com/wasserth/TotalSegmentator/?tab=readme-ov-file#advanced-settings.


### Step 3: Consolidate statistics and generate figures

```python
from liverct import (
	consolidate_group_statistics,
	generate_montages_from_bids,
)

bids_root = Path("/path/to/bids_dataset")

# Consolidate per-subject statistics.json files into one TSV
group_tsv = consolidate_group_statistics(bids_root)
print(group_tsv)

# Generate visual outputs
figure_summary = generate_montages_from_bids(
	bids_root=bids_root,
	montage_types=["individual", "cross-subject"],
	organs_to_montage=["liver", "spleen", "pancreas"],
	generate_liver_segments=True,
	include_skeletal=True,
	include_organs=["liver"],
)
print(figure_summary)
```

## Common onfiguration Options

### Conversion

- `dicom_subdir`: name of DICOM subfolder in each source case directory
- `subject_id` and `session_id` (when using `CTBIDSConverter.convert` directly)
- `dcm2bids4ct_path`: path to converter executable if not on PATH. Ignore if installed in your conda environment.

### Segmentation

- `series_description_pattern`: choose the intended CT if multiple are present
- `tasks`: one [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/?tab=readme-ov-file#subtasks) task or list of tasks
- `statistics`: whether stats files (volume and mean intensity) are produced.
- `license_number`: required for certain TotalSegmentator models
- `device`: `gpu` or `cpu`. Defaults to gpu, falls back to cpu if none available.
- `nr_thr_resamp`, `nr_thr_saving`: lower values can reduce memory pressure
- `overwrite`: rerun even if outputs already exist
- advanced TotalSegmentator options are passed through directly via keyword arguments (for example: `preview=True`, `radiomics=True`)

### Figure generation

- `montage_types`: `individual`, `cross-subject`, or both
- `organs_to_montage`: `all` or selected organ names
- `include_organs`: overlay specific organs on tissue montages
- `include_skeletal`: include skeletal composite overlay
- `num_slices`, `window`, `alpha`, `axis`, `dpi`
- `superior_limit` and `inferior_limit`: constrain montage range anatomically

#### Figure generation behavior details

- `montage_types` controls tissue-type montage families only:
	- `individual`: one tissue montage per subject
	- `cross-subject`: one vertebra-aligned cross-subject tissue montage
- `include_organs` applies to tissue montages (`individual` and `cross-subject`) as additional overlays on top of tissue classes.
- `organs_to_montage` is separate from `include_organs`: it controls dedicated organ-specific montage images (one per requested organ per subject), regardless of `montage_types`.
- `generate_liver_segments=True` is also separate: it creates liver subsegment montages from `liver_segments` outputs.

Supported organ names for both `include_organs` and `organs_to_montage`:

- `liver`
- `pancreas`
- `spleen`
- `kidneys`
- `lungs`
- `prostate`
- `heart`

These organ names map to masks in TotalSegmentator `total` outputs:

- `liver` -> `total/liver.nii.gz`
- `pancreas` -> `total/pancreas.nii.gz`
- `spleen` -> `total/spleen.nii.gz`
- `kidneys` -> `total/kidney_left.nii.gz` + `total/kidney_right.nii.gz`
- `lungs` -> all lung-lobe masks in `total/`
- `prostate` -> `total/prostate.nii.gz`
- `heart` -> `total/heart.nii.gz`

Practical requirement summary:

- Tissue montages require `tissue_types` outputs.
- Organ overlays/montages require `total` outputs.
- Liver segment montages require `liver_segments` outputs.

## Output structure (typical)

Under your BIDS root:

- `sub-XXX[/ses-YYY]/ct/`: converted CT NIfTI + JSON sidecars
- `derivatives/totalsegmentator/sub-XXX[/ses-YYY]/<task>/`: segmentation outputs per task
- `.../<task>/statistics.json`: per-label volume and intensity summary
- `.../<task>/source.json`: traceability metadata to the source CT file
- `derivatives/totalsegmentator/group/statistics.tsv`: group-level consolidated table
- `derivatives/totalsegmentator/.../figures/`: generated montage PNG files
- `sub-XXX_vertebrae_slices.tsv` and `sub-XXX_slice_vertebrae.tsv`: vertebrae slice reports

## Common pitfalls

- No matching CT found: set `series_description_pattern` explicitly
- Segmentation skipped: output already exists and `overwrite=False`
- Stats computation fails: check that `source.json` exists and points to a valid CT file
- Empty/incorrect figures: verify segmentation mask files exist for selected tasks/organs

## License

MIT License
