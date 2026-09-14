#!/usr/bin/env python3
"""Generate subject-specific study-level HTML review reports."""
import argparse
import logging
from pathlib import Path
from liverct.ingestion import generate_review_reports, load_config

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="inventory_scored.tsv", type=Path)
parser.add_argument("--output-dir", default=".", type=Path)
parser.add_argument("--config", type=Path)
parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
args = parser.parse_args()
logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger(__name__).info("Starting inventory review: input=%s output_dir=%s", args.input, args.output_dir)
decisions = generate_review_reports(args.input, args.output_dir, load_config(args.config))
print("Wrote review reports and {}".format(decisions))
