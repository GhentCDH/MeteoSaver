"""Column-wise row grouping: a tilt-robust replacement for transcription.organize_contours_top/_midpoint.

1. Group contours into columns by cutting the sorted x-centres at the (n_cols-1) largest gaps.
2. A column with exactly n_rows cells defines row r by rank order. These "reference" columns give the
   y of each row at their x.
3. Every other column's cells are assigned to the nearest reference row after interpolating the
   reference y-profile to that column's x (so skew/curvature cancels out).
Returns rows as lists of contours, like the originals.
"""
from __future__ import annotations

import cv2
import numpy as np


def column_wise_rows(contours, n_rows: int, n_cols: int = 24):
    if not contours:
        return []
    rects = [cv2.boundingRect(c) for c in contours]
    cx = np.array([x + w / 2 for x, y, w, h in rects])
    cy = np.array([y + h / 2 for x, y, w, h in rects])
    order = np.argsort(cx)
    cuts = np.sort(np.argsort(np.diff(cx[order]))[-(n_cols - 1):]) + 1
    cols = [order[a:b] for a, b in zip([0, *cuts], [*cuts, len(order)])]

    refs = [(cx[c].mean(), np.sort(cy[c])) for c in cols if len(c) == n_rows]
    if not refs:  # no complete column: fall back to the column with the most cells
        best = max(cols, key=len)
        ys = np.sort(cy[best])
        refs = [(cx[best].mean(), np.interp(np.linspace(0, len(ys) - 1, n_rows), np.arange(len(ys)), ys))]
    refs.sort(key=lambda t: t[0])
    ref_x = np.array([r[0] for r in refs])
    ref_y = np.stack([r[1] for r in refs])  # (n_ref, n_rows)

    rows = [[] for _ in range(n_rows)]
    for c in cols:
        x = cx[c].mean()
        profile = np.array([np.interp(x, ref_x, ref_y[:, r]) for r in range(n_rows)])
        for i in c:
            rows[int(np.argmin(np.abs(profile - cy[i])))].append(contours[i])
    return [sorted(r, key=lambda c: cv2.boundingRect(c)[0]) for r in rows]
