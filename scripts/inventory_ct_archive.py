#!/usr/bin/env python3
"""Create an archive-agnostic CT series inventory."""
import argparse
from pathlib import Path
from liverct.ingestion import inventory_archive

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, type=Path)
parser.add_argument("--output", default="inventory.tsv", type=Path)
args = parser.parse_args()
frame = inventory_archive(args.root, args.output)
print("Discovered {} series; wrote {}".format(len(frame), args.output))
