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

`inventory.tsv`, scored inventory, HTML reports, and sourcedata are generated. `review.tsv` is manually edited and preserved. `manifest.tsv` is the authoritative selected-series import list.
