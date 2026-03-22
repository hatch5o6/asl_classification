#!/usr/bin/env python3
"""Create per-language test CSVs with multilingual class IDs.

All three are filtered from data/multilingual/test.csv, which uses the
correct multilingual label space (0-3266).

  AUTSL  labels 0   - 225   → data/multilingual/test_autsl.csv
  ASL    labels 226 - 2956  → data/multilingual/test_asl.csv
  GSL    labels 2957- 3266  → data/multilingual/test_gsl.csv

Note: data/test.csv and data/asl_citizen/test.csv use per-language class IDs
and must NOT be used directly with the multilingual model.
"""

import csv
import os

DATA_DIR = "data/multilingual"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

LANG_RANGES = {
    "autsl": (0,    225),
    "asl":   (226,  2956),
    "gsl":   (2957, 3266),
}

counts = {}
for lang, (lo, hi) in LANG_RANGES.items():
    out_path = os.path.join(DATA_DIR, f"test_{lang}.csv")
    count = 0
    with open(TEST_CSV) as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["rgb_path", "depth_path", "skel_path", "label"])
        for row in reader:
            label = int(row["label"])
            if lo <= label <= hi:
                writer.writerow([row["rgb_path"], row["depth_path"], row["skel_path"], row["label"]])
                count += 1
    counts[lang] = count
    print(f"Wrote {count:>6} rows to {out_path}")

print(f"\nTotal: {sum(counts.values())} rows")
