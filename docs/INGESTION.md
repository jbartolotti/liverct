# CT archive ingestion

The ingestion subsystem prepares clinical DICOM archives for the existing BIDS converter.

```text
archive -> inventory.tsv -> inventory_scored.tsv -> review.tsv -> manifest.tsv -> sourcedata/
```

Install the optional dependencies with:

```bash
pip install -e ".[ingestion]"
```

Run the stages from the repository root:

```bash
python scripts/inventory_ct_archive.py --root /path/to/archive --output inventory.tsv
python scripts/score_ct_inventory.py --input inventory.tsv --output inventory_scored.tsv --config config/ingestion.example.yaml
python scripts/review_ct_inventory.py --input inventory_scored.tsv --output-dir review --config config/ingestion.example.yaml
# Edit review/review.tsv. Change reviewer_decision to PRIMARY, SECONDARY, or REJECT, then:
python scripts/build_ct_manifest.py --inventory inventory_scored.tsv --review review/review.tsv --output manifest.tsv --config config/ingestion.example.yaml
# Add --include-secondary to retain both PRIMARY and SECONDARY selections.
python scripts/stage_ct_sourcedata.py --manifest manifest.tsv --archive-root /path/to/archive --bids-root /path/to/bids
```

For a safe archive smoke test, add `--test` to the inventory command. It scans
only the first top-level directory below the archive root, writes a partial
inventory, and logs which directory was selected:

```bash
python scripts/inventory_ct_archive.py \
	--root /path/to/archive \
	--output inventory_test.tsv \
	--test
```

All stage runners accept `--log-level DEBUG|INFO|WARNING|ERROR`. The default
`INFO` level reports input/output paths, subject/session/series context, file
and series counts, tier counts, review montage counts, and staging totals.
Use `--log-level DEBUG` when you need per-file details such as existing staged
files that were skipped.

`inventory.tsv`, the complete scored audit inventory, subject-specific HTML reports, and sourcedata are generated. Scoring groups series by StudyInstanceUID, falling back to subject and study date, and recommends one PRIMARY series per scan date with other eligible acquisitions marked SECONDARY. Selection is geometry-first: coverage, reconstruction diameter, kernel, phase, and slice thickness dominate the score, while naming conventions and `ORIGINAL`/`PRIMARY`/`AXIAL` flags provide supporting evidence. The human-facing report may summarize every scan date, but `review.tsv` is an intentionally small queue containing only ambiguous `REVIEW_REQUIRED` candidates. Automatic primary selections and `NO_CANDIDATE` dates are omitted from `review.tsv`; all series and explainability fields remain in `inventory_scored.tsv`. `review.tsv` is manually edited and preserved; its editable column is `reviewer_decision` with values `PRIMARY`, `SECONDARY`, or `REJECT`. By default the manifest includes PRIMARY rows only; `--include-secondary` also includes SECONDARY rows, and REJECT rows are never staged. `manifest.tsv` is the authoritative selected-series import list.

Review HTML and TSV artifacts use the same subject, study-date, and numeric
series-number ordering. The TSV includes an `index` column and an `is_data`
column (`1` for series rows, `0` for date-separator rows); separator rows have
blank metadata fields to make the file easier to scan and filter.
The `review_row_type` column identifies `SERIES`, `STATUS`, and `SEPARATOR`
rows.

Review montage generation uses metadata-only DICOM reads while indexing slice
locations, decodes only the selected evenly spaced slices, and reuses an
existing PNG when its expected path is already present. `review.montage_workers`
defaults to `1`; larger values enable process-based montage generation. Fixed
`review.window_min` and `review.window_max` values are used when configured,
with the previous percentile fallback available when either bound is absent.

Automatic candidate selection is performed once per subject and study date,
even when that date contains multiple `StudyInstanceUID` values. Include and
exclude terms are matched against the combined `series_description`,
`study_description`, and `body_part_examined` text, so anatomy recorded only
at the study level still affects eligibility. A candidate must be CT, have
sufficient z coverage and slice count, and not be a scout/localizer, derived
reconstruction, screenshot, volume rendering, MIP/MPR, or sagittal/coronal
only series. `ORIGINAL`, `PRIMARY`, `AXIAL`, and abdomen terms are scoring
evidence, not hard requirements. The scorer adds geometry and reconstruction
explainability fields including `phase_type`, `coverage_score`, `fov_score`,
`kernel_score`, `phase_score`, `thickness_score`, `organ_focus_penalty`, and
`candidate_score`, along with `candidate_rank`, `candidate_status`,
`is_auto_primary`, and `candidate_reason`.

The default automatic score threshold is 60, with a minimum 10-point margin
over the runner-up. A single eligible candidate is marked `AUTO_PRIMARY`
without requiring the score threshold; when candidates compete, the threshold
and margin are applied. A clear winner is marked `AUTO_PRIMARY`; an ambiguous
group is marked `REVIEW_REQUIRED` and is the only kind of group written to
`review.tsv`; and a date with no eligible candidates is
affirmatively marked `NO_CANDIDATE`. The thresholds and term lists are
configurable under `tiering`.
