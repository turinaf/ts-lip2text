#!/usr/bin/env python3
"""Count unique speaker folders in data/lipdata-digit/.

Each subset (e.g. subset_01) contains folders named by speaker id.
This script collects all speaker folder names across subsets and reports
the unique count. It can also list names or write them to a file.
"""

from pathlib import Path
import argparse
import sys


def find_speakers(root: Path):
	root = Path(root)
	if not root.exists():
		raise FileNotFoundError(f"Path not found: {root}")
	speakers = set()
	for subset in root.iterdir():
		if not subset.is_dir():
			continue
		for speaker in subset.iterdir():
			if speaker.is_dir():
				speakers.add(speaker.name)
	return speakers


def main(argv=None):
	p = argparse.ArgumentParser(description="Count unique speaker folders")
	p.add_argument("path", nargs="?", default="data/lipdata-digit/",
				   help="Root path containing subset folders (default: data/lipdata-digit/)")
	p.add_argument("-v", "--verbose", action="store_true", help="List speaker ids")
	p.add_argument("-o", "--out", help="Write speaker ids to a file")
	args = p.parse_args(argv)

	try:
		speakers = find_speakers(Path(args.path))
	except FileNotFoundError as e:
		print(e, file=sys.stderr)
		return 2

	if args.verbose:
		for s in sorted(speakers):
			print(s)

	print(f"Found {len(speakers)} unique speakers in '{args.path}'")

	if args.out:
		Path(args.out).write_text("\n".join(sorted(speakers)))


if __name__ == "__main__":
	raise SystemExit(main())
