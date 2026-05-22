#!/usr/bin/env python3
"""concat_csv.py – Concatenate two CSV files row‑by‑row.

Usage
-----
    python concat_csv.py first.csv second.csv merged.csv [--dedup] [--skip-header-b]

Positional arguments
~~~~~~~~~~~~~~~~~~~~
first.csv        The primary CSV file (header is preserved).
second.csv       Rows from this file are appended to *first.csv*.
merged.csv       Output file name you choose.

Optional flags
~~~~~~~~~~~~~~
--dedup          Remove duplicate data rows (header is never removed).
--skip-header-b  Force‑skip the header row of *second.csv*. If not set, the
                 script **automatically** skips it when the header matches the
                 first file.

Behaviour
~~~~~~~~~
* The header row from *first.csv* is kept.
* The script auto‑detects if the first row of *second.csv* matches that header
  and skips it to avoid duplication.
* Handles arbitrary row length / delimiter quirks via the standard ``csv``
  module (uses default comma delimiter).
* Streams data so it works on very large files.

"""
from __future__ import annotations
import csv
import argparse
from pathlib import Path
from typing import List, Sequence


def _read_csv(path: Path) -> List[List[str]]:
    """Read *all* rows of ``path`` and return them as a list of lists."""
    with path.open(newline="") as f:
        return list(csv.reader(f))


def _write_csv(path: Path, rows: Sequence[Sequence[str]]) -> None:
    """Write *rows* to *path*."""
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def deduplicate(rows: Sequence[Sequence[str]]) -> List[List[str]]:
    """Return *rows* with duplicates removed, preserving order."""
    seen: set[tuple[str, ...]] = set()
    out: List[List[str]] = []
    for row in rows:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            out.append(list(row))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate two CSV files.")
    p.add_argument("file_a", type=Path, help="First (primary) CSV file")
    p.add_argument("file_b", type=Path, help="Second CSV file to append")
    p.add_argument("output", type=Path, help="Name of the merged CSV file")
    p.add_argument("--dedup", action="store_true", help="Remove duplicate rows")
    p.add_argument(
        "--skip-header-b",
        action="store_true",
        help="Force skipping header row in second CSV (if it differs)",
    )
    args = p.parse_args()

    # --- Read inputs ---
    rows_a = _read_csv(args.file_a)
    rows_b = _read_csv(args.file_b)

    if not rows_a:
        raise SystemExit("Error: first file is empty – nothing to merge.")

    header = rows_a[0]

    # Decide from where to start copying rows_b
    b_start = 0
    if rows_b:
        if args.skip_header_b or rows_b[0] == header:
            b_start = 1  # skip header in B

    merged = rows_a + rows_b[b_start:]

    if args.dedup:
        merged = deduplicate(merged)

    # --- Write output ---
    _write_csv(args.output, merged)
    print(f"Merged {len(rows_a)} + {len(rows_b) - b_start} rows → {args.output}")


if __name__ == "__main__":
    main()
