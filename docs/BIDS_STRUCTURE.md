# BIDS Dataset Structure

This document describes the directory layout produced by `liverct` after conversion and pipeline processing.

## Full example tree

```
bids_root/
│
├── sub-0001/
│   └── ct/
│       ├── sub-0001_ct.nii.gz          # CT volume (NIfTI, Hounsfield units)
│       └── sub-0001_ct.json            # BIDS sidecar (SeriesDescription, acquisition metadata)
│
├── sub-0002/
│   └── ct/
│       ├── sub-0002_ct.nii.gz
│       └── sub-0002_ct.json
│
└── derivatives/
    └── totalsegmentator/
        │
        ├── sub-0001/
        │   ├── total/                           # TotalSegmentator "total" task
        │   │   ├── liver.nii.gz
        │   │   ├── spleen.nii.gz
        │   │   ├── pancreas.nii.gz
        │   │   ├── kidney_left.nii.gz
        │   │   ├── kidney_right.nii.gz
        │   │   ├── heart.nii.gz
        │   │   ├── vertebrae_L1.nii.gz
        │   │   ├── vertebrae_L2.nii.gz
        │   │   ├── ...                          # one .nii.gz per segmented label
        │   │   ├── statistics.json              # per-label volume + mean intensity
        │   │   └── source.json                  # path to source CT used for this task
        │   │
        │   ├── tissue_types/                    # TotalSegmentator "tissue_types" task
        │   │   ├── subcutaneous_fat.nii.gz
        │   │   ├── visceral_fat.nii.gz
        │   │   ├── skeletal_muscle.nii.gz
        │   │   ├── statistics.json
        │   │   └── source.json
        │   │
        │   ├── liver_segments/                  # TotalSegmentator "liver_segments" task
        │   │   ├── liver_segment_1.nii.gz
        │   │   ├── liver_segment_2.nii.gz
        │   │   ├── ...
        │   │   ├── statistics.json
        │   │   └── source.json
        │   │
        │   ├── abdominal_muscles/               # TotalSegmentator "abdominal_muscles" task
        │   │   ├── rectus_abdominis_left.nii.gz
        │   │   ├── rectus_abdominis_right.nii.gz
        │   │   ├── ...
        │   │   ├── statistics.json
        │   │   └── source.json
        │   │
        │   ├── figures/                         # generated montage PNGs (run_figures=True)
        │   │   ├── sub-0001_tissue_types_montage.png
        │   │   ├── sub-0001_liver_montage.png
        │   │   └── sub-0001_liver_segments_montage.png
        │   │
        │   ├── sub-0001_vertebrae_slices.tsv    # vertebra → slice index mapping
        │   └── sub-0001_slice_vertebrae.tsv     # slice index → vertebra mapping
        │
        ├── sub-0002/
        │   ├── total/
        │   │   ├── liver.nii.gz
        │   │   ├── ...
        │   │   ├── statistics.json
        │   │   └── source.json
        │   ├── tissue_types/
        │   │   ├── ...
        │   │   ├── statistics.json
        │   │   └── source.json
        │   ├── liver_segments/
        │   │   ├── ...
        │   │   ├── statistics.json
        │   │   └── source.json
        │   ├── abdominal_muscles/
        │   │   ├── ...
        │   │   ├── statistics.json
        │   │   └── source.json
        │   ├── figures/
        │   │   ├── sub-0002_tissue_types_montage.png
        │   │   └── ...
        │   ├── sub-0002_vertebrae_slices.tsv
        │   └── sub-0002_slice_vertebrae.tsv
        │
        ├── group/                               # cohort-level outputs
        │   ├── statistics.tsv                   # consolidated per-subject, per-label statistics
        │   └── cross_subject_montage.png        # vertebra-aligned grid across all subjects
        │
        ├── run_manifest_20260402_103045.json    # scheduler run manifest (timestamped)
        └── run_timeline_20260402_103045.png     # scheduler timeline figure (timestamped)
```

## Notes on key files

### `ct/*.nii.gz` and `ct/*.json`
Produced by `convert_dicom_directory_to_bids`. The JSON sidecar stores BIDS-compliant acquisition metadata
(SeriesDescription, Manufacturer, AcquisitionTime, etc.) extracted from the original DICOM headers.

### `<task>/statistics.json`
Per-label statistics computed over the segmentation masks and corresponding CT intensities:
```json
{
  "liver": { "volume": 1523456.0, "intensity": 55.3 },
  "spleen": { "volume": 214300.0, "intensity": 48.7 }
}
```
Volume is in mm³. Intensity is mean Hounsfield units within the mask.

### `<task>/source.json`
Traceability record linking this task's outputs back to the exact CT file used:
```json
{
  "source_file": "/path/to/bids_root/sub-0001/ct/sub-0001_ct.nii.gz",
  "source_filename": "sub-0001_ct.nii.gz",
  "SeriesDescription": "ABD/PEL 2.5mm STND DLIR"
}
```

### `run_manifest_*.json`
Written incrementally during execution (checkpoint after every job state change).
Contains per-job status, start/end timestamps, attempt records, and retry history.
Survives process crashes. Can be used with `timeline_from_manifest()` to regenerate
the timeline figure after a crashed run.

### Session support
If sessions (`ses-*`) are present, an additional level is inserted throughout:
```
sub-0001/
└── ses-01/
    └── ct/
        ├── sub-0001_ses-01_ct.nii.gz
        └── sub-0001_ses-01_ct.json

derivatives/totalsegmentator/
└── sub-0001/
    └── ses-01/
        ├── total/
        │   └── ...
        └── figures/
            └── ...
```
