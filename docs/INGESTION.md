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
# Edit review/review.tsv, then:
python scripts/build_ct_manifest.py --inventory inventory_scored.tsv --review review/review.tsv --output manifest.tsv --config config/ingestion.example.yaml
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

`inventory.tsv`, scored inventory, HTML reports, and sourcedata are generated. `review.tsv` is manually edited and preserved. `manifest.tsv` is the authoritative selected-series import list.
