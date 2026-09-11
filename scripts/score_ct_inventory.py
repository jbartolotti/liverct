#!/usr/bin/env python3
"""Score an inventory into four review tiers."""
import argparse
from pathlib import Path
from liverct.ingestion import load_config, score_inventory

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="inventory.tsv", type=Path)
parser.add_argument("--output", default="inventory_scored.tsv", type=Path)
parser.add_argument("--config", type=Path)
args = parser.parse_args()
frame = score_inventory(args.input, args.output, load_config(args.config))
print("Scored {} series; wrote {}".format(len(frame), args.output))
