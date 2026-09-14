#!/usr/bin/env python3
"""Score an inventory into legacy tiers and study-level recommendations."""
import argparse
import logging
from pathlib import Path
from liverct.ingestion import load_config, score_inventory

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="inventory.tsv", type=Path)
parser.add_argument("--output", default="inventory_scored.tsv", type=Path)
parser.add_argument("--config", type=Path)
parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
args = parser.parse_args()
logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger(__name__).info("Starting inventory scoring: input=%s output=%s", args.input, args.output)
frame = score_inventory(args.input, args.output, load_config(args.config))
print("Scored {} series; wrote {}".format(len(frame), args.output))
