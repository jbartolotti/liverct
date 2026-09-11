#!/usr/bin/env python3
"""Create an archive-agnostic CT series inventory."""
import argparse
import logging
from pathlib import Path
from liverct.ingestion import inventory_archive

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, type=Path)
parser.add_argument("--output", default="inventory.tsv", type=Path)
parser.add_argument(
	"--test",
	action="store_true",
	help="Scan only the first top-level archive directory and stop",
)
parser.add_argument(
	"--log-level",
	choices=("DEBUG", "INFO", "WARNING", "ERROR"),
	default="INFO",
	help="Logging verbosity (default: INFO)",
)
args = parser.parse_args()
logging.basicConfig(
	level=getattr(logging, args.log_level),
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger(__name__).info("Starting CT archive inventory: root=%s test_mode=%s", args.root, args.test)
frame = inventory_archive(args.root, args.output, test_mode=args.test)
print("Discovered {} series; wrote {}".format(len(frame), args.output))
