"""Drop duplicate rows from `results/sweep_summary.jsonl`, keeping the first.

Needed once, because stage B appended unconditionally before the guard was added
(see `sweep.py`): an interrupted stage B that was re-run wrote a second eval row
for seeds it had already scored, and the duplicates entered the median and IQR
silently -- an `n = 9` where 5 seeds were requested.

    uv run python scripts/dedupe_summary.py --check     # report, change nothing
    uv run python scripts/dedupe_summary.py --write     # rewrite in place
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "sweep_summary.jsonl"
KEY = ("arch", "cadence", "shaping", "seed", "split")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="rewrite the file; otherwise report only")
    ap.add_argument(
        "--check", action="store_true", help="report only (the default; accepted for symmetry)"
    )
    ap.add_argument("--path", type=Path, default=SUMMARY)
    a = ap.parse_args()

    rows = [json.loads(line) for line in a.path.open()]
    counts = Counter(tuple(r[k] for k in KEY) for r in rows)
    dupes = {k: n for k, n in counts.items() if n > 1}

    print(f"{len(rows)} rows, {len(counts)} unique, {len(dupes)} duplicated")
    for key, n in sorted(dupes.items()):
        print(f"  x{n}  " + " ".join(f"{k}={v}" for k, v in zip(KEY, key)))
    if not dupes:
        print("nothing to do")
        return

    seen: set[tuple] = set()
    kept = []
    for row in rows:
        key = tuple(row[k] for k in KEY)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)

    if not a.write:
        print(f"\n--write would keep {len(kept)} rows (first occurrence of each)")
        return
    with a.path.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")
    print(f"\nrewrote {a.path} with {len(kept)} rows")


if __name__ == "__main__":
    main()
