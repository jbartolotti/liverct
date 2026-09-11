#!/usr/bin/env python3
"""Build the authoritative selected-series manifest."""
import argparse
import logging
from pathlib import Path
from liverct.ingestion import build_manifest, load_config

parser = argparse.ArgumentParser()
parser.add_argument("--inventory", default="inventory_scored.tsv", type=Path)
parser.add_argument("--review", default="review.tsv", type=Path)
parser.add_argument("--output", default="manifest.tsv", type=Path)
parser.add_argument("--config", type=Path)
parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
args = parser.parse_args()
logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger(__name__).info("Starting manifest generation: inventory=%s review=%s output=%s", args.inventory, args.review, args.output)
manifest = build_manifest(args.inventory, args.review, args.output, load_config(args.config))
print("Selected {} series; wrote {}".format(len(manifest), args.output))
