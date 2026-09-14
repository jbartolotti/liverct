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

`inventory.tsv`, scored inventory, subject-specific HTML reports, and sourcedata are generated. Scoring groups series by StudyInstanceUID, falling back to subject and study date, and recommends one PRIMARY series per study with other eligible acquisitions marked SECONDARY. The report table includes every series, while montages are generated only for PRIMARY and SECONDARY recommendations unless `review.detailed_review` is enabled. `review.tsv` is manually edited and preserved; its editable column is `reviewer_decision` with values `PRIMARY`, `SECONDARY`, or `REJECT`. By default the manifest includes PRIMARY rows only; `--include-secondary` also includes SECONDARY rows, and REJECT rows are never staged. `manifest.tsv` is the authoritative selected-series import list.

Review HTML and TSV artifacts use the same subject, study-date, and numeric
series-number ordering. The TSV includes an `index` column and an `is_data`
column (`1` for series rows, `0` for date-separator rows); separator rows have
blank metadata fields to make the file easier to scan and filter.
