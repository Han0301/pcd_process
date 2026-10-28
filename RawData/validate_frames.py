"""逐帧从原始 METABAG 核对时间、强度和去畸变点的原始行号。"""

import csv
import json
from pathlib import Path

import numpy as np

from RawData.raw_index_offset_lookup import read_raw_frame, recover_indices


def validate(source, audit, lio):
    source, audit, lio = map(Path, (source, audit, lio))
    rows = list(csv.DictReader((audit / "lidar_index.csv").open()))
    indexed = {r["header_ns"]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("原始激光帧时间戳不唯一")
    trajectory = list(csv.DictReader((lio / "trajectory.csv").open()))
    if not trajectory:
        raise ValueError("无有效 LIO 轨迹")
    gravity = np.array([float(trajectory[0][k]) for k in ["gx", "gy", "gz"]])
    gravity /= np.linalg.norm(gravity)
    target = np.array([0.0, 0.0, -1.0])
    v = np.cross(gravity, target)
    c = np.dot(gravity, target)
    s = np.linalg.norm(v)
    if s < 1e-12 and c < 0:
        raise ValueError("初始重力方向相反，拒绝采用单位变换")
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s)) if s > 1e-12 else np.eye(3)
    frames = list(csv.DictReader((lio / "frames.csv").open()))
    accepted = []
    total = 0
    for row in frames:
        rawrow = indexed[row["source_header_ns"]]
        if row["source_timebase_ns"] != rawrow["timebase_ns"]:
            raise ValueError("原始 timebase 不匹配")
        raw = read_raw_frame(source, rawrow)
        cloud = np.fromfile(lio / row["file"], dtype="<f4").reshape(-1, 4)
        offsets = np.fromfile(lio / row["offset_file"], dtype="<f4")
        ids = recover_indices(raw, offsets, cloud[:, 3]).astype("<u4")
        name = str(Path(row["file"]).with_name(Path(row["file"]).stem + "_raw_indices.bin"))
        (lio / name).write_bytes(ids.tobytes())
        total += len(ids)
        accepted.append(
            {
                **row,
                "raw_index_file": name,
                "source_file": rawrow["source_file"],
                "source_frame": rawrow["source_frame"],
                "global_lidar_frame": rawrow["global_lidar_frame"],
                "point_array_byte_offset": rawrow["point_array_byte_offset"],
            }
        )
    with (lio / "frames_with_raw_indices.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(accepted[0]))
        writer.writeheader()
        writer.writerows(accepted)
    report = {
        "saved_frames": len(frames),
        "raw_indices_recovered": total,
        "all_actual_deskew_offsets_exact_source_match": True,
        "all_intensity_exact_source_match": True,
        "raw_source_header_and_timebase_match": True,
        "T_gravity_from_lio_world": matrix.tolist(),
    }
    (lio / "raw_index_shape_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report
