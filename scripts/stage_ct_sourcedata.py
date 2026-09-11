#!/usr/bin/env python3
"""Materialize a selected manifest as BIDS sourcedata."""
import argparse
from pathlib import Path
from liverct.ingestion import stage_sourcedata

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="manifest.tsv", type=Path)
parser.add_argument("--archive-root", required=True, type=Path)
parser.add_argument("--bids-root", required=True, type=Path)
parser.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy")
args = parser.parse_args()
output = stage_sourcedata(args.manifest, args.archive_root, args.bids_root, args.mode)
print("Wrote sourcedata to {}".format(output))
