#!/usr/bin/env python3
"""Build the authoritative selected-series manifest."""
import argparse
from pathlib import Path
from liverct.ingestion import build_manifest, load_config

parser = argparse.ArgumentParser()
parser.add_argument("--inventory", default="inventory_scored.tsv", type=Path)
parser.add_argument("--review", default="review.tsv", type=Path)
parser.add_argument("--output", default="manifest.tsv", type=Path)
parser.add_argument("--config", type=Path)
args = parser.parse_args()
manifest = build_manifest(args.inventory, args.review, args.output, load_config(args.config))
print("Selected {} series; wrote {}".format(len(manifest), args.output))
