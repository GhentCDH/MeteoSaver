"""Deterministic train/val/test split of the tesseract cell dataset (stems only, no files copied).

Usage: python make_split.py [--data DIR] [--val 0.02] [--test 0.08] [--seed 0]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common import DATA_DIR, SPLIT_PATH, read_gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--val", type=float, default=0.02)
    ap.add_argument("--test", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stems = sorted(p.name[: -len(".png")] for p in args.data.glob("*.png") if (args.data / f"{p.stem}.gt.txt").exists())
    random.Random(args.seed).shuffle(stems)
    n_val, n_test = round(len(stems) * args.val), round(len(stems) * args.test)
    split = {"val": stems[:n_val], "test": stems[n_val : n_val + n_test], "train": stems[n_val + n_test :]}
    SPLIT_PATH.write_text(json.dumps(split))

    print({k: len(v) for k, v in split.items()}, "total", len(stems))
    assert not (set(split["train"]) & set(split["test"])) and not (set(split["train"]) & set(split["val"]))
    labels = {s: read_gt(args.data, s) for s in stems}
    print("label length:", sorted(collections.Counter(len(t) for t in labels.values()).items()))
    print("charset:", "".join(sorted(set("".join(labels.values())))))
    train_labels = {labels[s] for s in split["train"]}
    dup = sum(labels[s] in train_labels for s in split["test"]) / max(len(split["test"]), 1)
    print(f"test cells whose exact label also occurs in train: {dup:.1%} (expected to be high for short numerics)")
    print("empty labels:", sum(not t for t in labels.values()))
    print(f"wrote {SPLIT_PATH}")


if __name__ == "__main__":
    main()
