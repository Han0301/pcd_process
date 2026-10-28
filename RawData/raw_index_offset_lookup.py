#!/usr/bin/env python3
"""Recover raw CustomPoint row IDs using unchanged FAST-LIO float-ms offsets.

This does not infer geometry. It refuses duplicate source offsets, missing offsets,
unexpected intensities, or a mismatched frame; it writes no input files.
"""

import mmap
from pathlib import Path

import numpy as np

from RawData.audit_metabag import POINT


def read_raw_frame(source_root, index_row):
    path = Path(source_root) / "data" / index_row["source_file"]
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        return np.frombuffer(
            m,
            dtype=POINT,
            count=int(index_row["raw_points"]),
            offset=int(index_row["point_array_byte_offset"]),
        ).copy()


def recover_indices(raw_points, exported_offset_ms, exported_intensity=None):
    offsets = np.asarray(exported_offset_ms, dtype=np.float32)
    raw_ms = raw_points["offset_ns"].astype(np.float32) / np.float32(1000000)
    if offsets.ndim != 1 or not np.isfinite(offsets).all():
        raise ValueError("invalid exported offsets")
    order = np.argsort(raw_ms, kind="stable")
    sorted_ms = raw_ms[order]
    if len(sorted_ms) > 1 and np.any(sorted_ms[1:] == sorted_ms[:-1]):
        raise ValueError("source float-ms offsets not unique; exact raw index unresolved")
    jj = np.searchsorted(sorted_ms, offsets)
    if np.any(jj >= len(sorted_ms)) or not np.array_equal(sorted_ms[jj], offsets):
        raise ValueError("exported offsets not an exact source subset")
    indices = order[jj].astype(np.int64)
    if len(indices) != len(np.unique(indices)):
        raise ValueError("exported points repeat source row IDs")
    if exported_intensity is not None:
        intensity = np.asarray(exported_intensity)
        if intensity.shape != offsets.shape or not np.array_equal(
            raw_points["intensity"][indices], intensity
        ):
            raise ValueError("raw/export reflectivity mismatch")
    return indices
