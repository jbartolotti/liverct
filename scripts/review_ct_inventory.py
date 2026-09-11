#!/usr/bin/env python3
"""Generate image-based HTML review reports."""
import argparse
from pathlib import Path
from liverct.ingestion import generate_review_reports, load_config

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="inventory_scored.tsv", type=Path)
parser.add_argument("--output-dir", default=".", type=Path)
parser.add_argument("--config", type=Path)
args = parser.parse_args()
decisions = generate_review_reports(args.input, args.output_dir, load_config(args.config))
print("Wrote review reports and {}".format(decisions))
