#!/usr/bin/env python3
"""Materialize a selected manifest as BIDS sourcedata."""
import argparse
import logging
from pathlib import Path
from liverct.ingestion import stage_sourcedata

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="manifest.tsv", type=Path)
parser.add_argument("--archive-root", required=True, type=Path)
parser.add_argument("--bids-root", required=True, type=Path)
parser.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy")
parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
args = parser.parse_args()
logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger(__name__).info("Starting sourcedata staging: manifest=%s bids_root=%s mode=%s", args.manifest, args.bids_root, args.mode)
output = stage_sourcedata(args.manifest, args.archive_root, args.bids_root, args.mode)
print("Wrote sourcedata to {}".format(output))
