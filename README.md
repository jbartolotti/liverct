# liverct

`liverct` is a Python package for end-to-end CT processing:

1. Convert CT DICOM data to a BIDS-style dataset via [dcm2bids4ct](https://github.com/ChristianHinge/dcm2bids4ct)
2. Run the full pipeline: TotalSegmentator segmentation, per-label statistics, per-subject figures, and cohort-level outputs — all in one call


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

### Step 2: Run the pipeline

```python
from liverct import BIDSProcessingPipeline

bids_root = Path("/path/to/bids_dataset")
pipeline = BIDSProcessingPipeline(bids_root)

results = pipeline.run_pipeline(
    tasks=["total", "tissue_types", "liver_segments"],
    statistics=True,
    run_figures=True,
    run_group_statistics=True,
    run_cross_subject_montage=True,
)
print(results)
```

All `sub-*` folders in the BIDS root are discovered and processed automatically. Sessions (`ses-*`) are detected and iterated automatically when present. To limit processing to specific subjects, pass `subjects=["sub-001", "sub-002"]`.

You can append additional advanced settings that are passed through to TotalSegmentator https://github.com/wasserth/TotalSegmentator/?tab=readme-ov-file#advanced-settings.

Pass `tasks=None` to skip segmentation entirely and run only cohort-level jobs against existing outputs (useful for re-running statistics or figures after a completed segmentation run).

## Common Configuration Options

### Conversion

- `dicom_subdir` (default `"DICOM"`): name of DICOM subfolder in each source case directory
- `overwrite` (default `False`): if `True`, re-convert all subjects even if they already exist in BIDS. If `False`, automatically skips subjects that have already been converted to BIDS (i.e., have existing `.nii.gz` files in `bids_root/sub-*/ct/`). Useful for re-running the function when new participants are added without re-converting existing data.

### Segmentation

- `tasks` (default `"total"`): one [TotalSegmentator](https://github.com/wasserth/TotalSegmentator/?tab=readme-ov-file#subtasks) task or list of tasks. Pass `tasks=None` to explicitly skip segmentation step.
- `subjects` (default `None` = all subjects): limit to specific subject IDs, e.g. `["sub-001", "sub-002"]`
- `series_description_pattern` (default `None`): regex pattern to choose the intended CT when multiple series are present
- `statistics` (default `True`): generate volume and mean-intensity stats files per segmented label in each task.
- `license_number` (default `None`): required for certain TotalSegmentator tasks (e.g. `tissue_types`)
- `device` (default `"gpu"`): `"gpu"` or `"cpu"`. Automatically falls back to `"cpu"` if no GPU is available
- `nr_thr_resamp` (default `1`), `nr_thr_saving` (default `6`): thread counts for resampling and saving; lower values reduce memory pressure
- `overwrite` (default `False`): if `True`, rerun segmentation even if outputs already exist
- `output_dir` (default `None` = `bids_root/derivatives/totalsegmentator/`): override the base derivatives directory
- advanced TotalSegmentator options are passed through directly via keyword arguments (e.g. `preview=True`, `radiomics=True`)

### Cohort outputs

All outputs default to `bids_root/derivatives/totalsegmentator/`.

- `run_group_statistics` (default `False`): consolidate per-subject stats into a single group TSV after all subjects complete
- `group_statistics_output_file` (default `None` = `bids_root/derivatives/totalsegmentator/group_statistics.tsv`): override the output TSV path
- `run_cross_subject_montage` (default `False`): generate a vertebra-aligned cross-subject montage after all subjects complete; requires the `total` task
- `cross_subject_vertebrae`: explicit list of vertebra labels to use, e.g. `["L1", "L2", "L3"]`. Default `None` = auto-select
- `cross_subject_num_vertebrae` (default `7`): how many vertebrae to auto-select when `cross_subject_vertebrae=None`
- `cross_subject_output_dir`: override the output directory for the montage image
- `cross_subject_include_skeletal` (default `True`): include skeletal composite overlay
- `cross_subject_include_organs` (default `None`): add organ overlays, e.g. `["liver", "spleen"]`
- `cross_subject_window` (default `(-200, 250)`): HU display window `(min, max)`
- `cross_subject_alpha` (default `0.35`): overlay transparency
- `cross_subject_dpi` (default `200`): output image resolution

### Scheduler / parallelism

- `parallel` (default `False`): enable scheduler-based parallel execution
- `max_workers` (default `None` = `1`): maximum concurrent jobs when `parallel=True`
- `gpu_workers` (default `None`): explicit GPU pool size; overrides `max_workers` for GPU jobs
- `cpu_workers` (default `1`): CPU pool size for postprocessing and cohort jobs
- `parallel_tasks` (default `False`): split task list into one scheduled job per task per subject
- `split_postprocessing` (default `False`): run stats and reports as separate CPU jobs that depend on segmentation completing; required when using cohort jobs in parallel mode
- `max_retries` (default `0`): retry failed jobs this many times before marking them failed
- `continue_on_error` (default `True`): if `False`, raise an error as soon as any job fails

### Run summaries

Both default to `bids_root/derivatives/totalsegmentator/` with a `YYYYMMDD_HHMMSS` timestamp so repeated runs don't overwrite each other.

- `manifest_path`: output path for a structured JSON run manifest (per-job status, timing, retry history)
- `timeline_figure_path`: output path for a PNG scheduler timeline (time on x-axis, worker lanes on y-axis, boxes color-coded by job type)

### Per-subject figures

Controlled via `run_figures=True` in `run_pipeline()`. A figure job is queued per subject after segmentation completes.

- `run_figures` (default `False`): generate per-subject figure montages after segmentation
- `figures_montage_types` (default `"individual"`): tissue-type montage family — `"individual"`, `"cross-subject"`, or a list of both. Note: the cohort cross-subject montage is already handled by `run_cross_subject_montage`; this controls whether a cross-subject tissue montage is also produced from the per-subject job
- `figures_organs_to_montage` (default `None`): dedicated organ montages — `None` = skip, `"all"` = all supported organs, or a list e.g. `["liver", "spleen"]`
- `figures_generate_liver_segments` (default `False`): generate liver subsegment montages from `liver_segments` outputs
- `figures_include_skeletal` (default `True`): include skeletal composite overlay on tissue montages
- `figures_include_organs` (default `None`): organ overlays drawn on top of tissue montages (distinct from `figures_organs_to_montage`)
- `figures_num_slices` (default `12`): number of slices in each montage
- `figures_window` (default `(-200, 250)`): HU display window `(min, max)`
- `figures_alpha` (default `0.5`): overlay transparency
- `figures_axis` (default `2`): slice axis — `0` sagittal, `1` coronal, `2` axial
- `figures_dpi` (default `200`): output image resolution
- `figures_superior_limit` (default `None`): superior anatomical limit, e.g. `"C1"` or `"T1"`
- `figures_inferior_limit` (default `None`): inferior anatomical limit, e.g. `"L5"` or `"sacrum"`

#### Figure type behavior

- `figures_montage_types` controls tissue-type montage families only:
	- `"individual"`: one tissue overlay montage per subject
	- `"cross-subject"`: one vertebra-aligned grid comparing all subjects
- `figures_include_organs` applies to tissue montages as overlays on top of tissue classes.
- `figures_organs_to_montage` is separate: dedicated single-organ montages (one image per organ per subject).
- `figures_generate_liver_segments` is also separate: liver subsegment montages from `liver_segments` task outputs.

Supported organ names for both `figures_include_organs` and `figures_organs_to_montage`:

| Name | Source masks in `total/` |
|------|--------------------------|
| `liver` | `liver.nii.gz` |
| `pancreas` | `pancreas.nii.gz` |
| `spleen` | `spleen.nii.gz` |
| `kidneys` | `kidney_left.nii.gz` + `kidney_right.nii.gz` |
| `lungs` | all lung-lobe masks |
| `prostate` | `prostate.nii.gz` |
| `heart` | `heart.nii.gz` |

Task requirements:

- Tissue montages require `tissue_types` outputs.
- Organ overlays/montages require `total` outputs.
- Liver segment montages require `liver_segments` outputs.

### Figure generation (standalone API)

`generate_montages_from_bids` is also available for direct use when you want to run figure generation independently of the pipeline. It accepts the same figure options as the `figures_*` parameters above (without the prefix). See `help(generate_montages_from_bids)` for the full parameter list.

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
