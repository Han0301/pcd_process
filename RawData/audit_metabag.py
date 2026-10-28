#!/usr/bin/env python3
"""Read-only indexed METABAG audit; no LIO, no map writes, no input mutation."""

import argparse
import collections
import csv
import hashlib
import json
import mmap
import struct
from pathlib import Path

import numpy as np

U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")
POINT = np.dtype(
    [
        ("offset_ns", "<u4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "u1"),
        ("tag", "u1"),
        ("line", "u1"),
    ]
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fields(m, start, size):
    out = {}
    p, end = start, start + size
    while p < end:
        n = U32.unpack_from(m, p)[0]
        p += 4
        if p + n > end:
            raise ValueError("truncated header field")
        k, v = m[p : p + n].split(b"=", 1)
        out[k.decode()] = v
        p += n
    if p != end:
        raise ValueError("invalid header end")
    return out


def record(m, p, end):
    if p + 8 > end:
        raise ValueError("truncated record length")
    hs = U32.unpack_from(m, p)[0]
    if hs > 16 << 20 or p + 8 + hs > end:
        raise ValueError("invalid record header length")
    h = fields(m, p + 4, hs)
    ds = U32.unpack_from(m, p + 4 + hs)[0]
    ps = p + 8 + hs
    if ps + ds > end:
        raise ValueError("truncated record payload")
    return h, ps, ds, ps + ds


def ros_header(m, p, size):
    seq, sec, ns, n = struct.unpack_from("<IIII", m, p)
    if n + 16 > size or ns >= 1000000000:
        raise ValueError("bad ROS header")
    return seq, sec * 1000000000 + ns, m[p + 16 : p + 16 + n].decode(), p + 16 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "data").glob("data_*"), key=lambda p: int(p.name[5:]))
    report = {
        "source": str(root),
        "source_mutated": False,
        "lio_run": False,
        "format": "METABAG V2.0 / ROS1 record encoding",
        "files": [],
        "unsupported_compression": [],
        "unknown_types": [],
        "samples": [],
    }
    if not files:
        raise FileNotFoundError(root / "data/data_*")
    rows, imurows = [], []
    types_total = collections.Counter()
    all_imu = []
    sampled_arrays = {}
    lidar_global = 0
    for path in files:
        entry = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
        declarations = {}
        counts = collections.Counter()
        file_lidar = 0
        sample_candidates = []
        with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            newline = m.find(b"\n")
            if m[:newline] != b"#METABAG V2.0":
                raise ValueError("bad magic " + str(path))
            p, chunks, outer_ops = newline + 1, 0, collections.Counter()
            while p < len(m):
                h, ps, ds, p = record(m, p, len(m))
                op = h["op"][0]
                outer_ops[str(op)] += 1
                if op != 5:
                    continue
                compression = h["compression"].decode()
                if compression != "none":
                    report["unsupported_compression"].append([path.name, compression])
                    continue
                chunks += 1
                q, end = ps, ps + ds
                while q < end:
                    rh, rp, n, q = record(m, q, end)
                    rop = rh["op"][0]
                    if rop == 7:
                        cid = U32.unpack(rh["conn"])[0]
                        cf = fields(m, rp, n)
                        declarations[cid] = {k: v.decode(errors="replace") for k, v in cf.items()}
                        declarations[cid]["record_topic"] = rh.get("topic", b"").decode()
                        continue
                    if rop != 2:
                        continue
                    cid = U32.unpack(rh["conn"])[0]
                    if cid not in declarations:
                        raise ValueError("undeclared connection")
                    typ = declarations[cid]["type"]
                    counts[typ] += 1
                    bagsec, bagns = struct.unpack("<II", rh["time"])
                    record_ns = bagsec * 1000000000 + bagns
                    if typ == "sensor_msgs/Imu":
                        seq, stamp, frame, b = ros_header(m, rp, n)
                        if rp + n - b != 296:
                            raise ValueError("bad IMU serialized size")
                        values = struct.unpack_from("<37d", m, b)
                        # q4, qcov9, angular3, angularcov9, acc3, acccov9
                        gyro = values[13:16]
                        acc = values[25:28]
                        all_imu.append((*gyro, *acc))
                        imurows.append(
                            [
                                path.name,
                                counts[typ] - 1,
                                rp,
                                n,
                                record_ns,
                                stamp,
                                seq,
                                frame,
                                *gyro,
                                *acc,
                            ]
                        )
                    elif "CustomMsg" in typ:
                        seq, stamp, frame, b = ros_header(m, rp, n)
                        base = U64.unpack_from(m, b)[0]
                        num = U32.unpack_from(m, b + 8)[0]
                        lidar_id = m[b + 12]
                        array_count = U32.unpack_from(m, b + 16)[0]
                        array_pos = b + 20
                        if num != array_count or array_pos + 19 * num != rp + n:
                            raise ValueError("bad Livox point array")
                        firstoff = U32.unpack_from(m, array_pos)[0] if num else 0
                        lastoff = U32.unpack_from(m, array_pos + 19 * (num - 1))[0] if num else 0
                        row = [
                            lidar_global,
                            path.name,
                            file_lidar,
                            rp,
                            n,
                            array_pos,
                            record_ns,
                            stamp,
                            base,
                            seq,
                            frame,
                            lidar_id,
                            num,
                            firstoff,
                            lastoff,
                        ]
                        rows.append(row)
                        sample_candidates.append(row)
                        file_lidar += 1
                        lidar_global += 1
            for candidate in [
                sample_candidates[0],
                sample_candidates[len(sample_candidates) // 2],
                sample_candidates[-1],
            ]:
                key = f"{path.name}_frame_{candidate[2]}"
                pts = np.frombuffer(m, dtype=POINT, count=candidate[12], offset=candidate[5]).copy()
                ids = np.unique(
                    np.rint(np.linspace(0, len(pts) - 1, min(16, len(pts)))).astype(np.int64)
                )
                xyz = np.column_stack([pts[x] for x in ("x", "y", "z")])
                sampled_arrays[key + "_raw_points"] = pts[ids]
                sampled_arrays[key + "_raw_indices"] = ids
                report["samples"].append(
                    {
                        "key": key,
                        "global_frame": candidate[0],
                        "source_file": path.name,
                        "source_frame": candidate[2],
                        "payload_byte_offset": candidate[3],
                        "array_byte_offset": candidate[5],
                        "raw_point_count": len(pts),
                        "finite_xyz_fraction": float(np.isfinite(xyz).all(axis=1).mean()),
                        "offset_min_ns": int(pts["offset_ns"].min()),
                        "offset_max_ns": int(pts["offset_ns"].max()),
                        "offset_monotonic": bool(
                            np.all(np.diff(pts["offset_ns"].astype(np.int64)) >= 0)
                        ),
                        "tag_counts": dict(
                            zip(*[x.tolist() for x in np.unique(pts["tag"], return_counts=True)])
                        ),
                        "line_counts": dict(
                            zip(*[x.tolist() for x in np.unique(pts["line"], return_counts=True)])
                        ),
                        "frame_id": candidate[10],
                    }
                )
        entry.update(
            {
                "chunks": chunks,
                "outer_ops": dict(outer_ops),
                "counts": dict(counts),
                "connections": declarations,
            }
        )
        report["files"].append(entry)
        types_total.update(counts)
        print(path.name, dict(counts), flush=True)
    header = [
        "global_lidar_frame",
        "source_file",
        "source_frame",
        "payload_byte_offset",
        "payload_bytes",
        "point_array_byte_offset",
        "record_ns",
        "header_ns",
        "timebase_ns",
        "header_seq",
        "frame_id",
        "lidar_id",
        "raw_points",
        "first_offset_ns",
        "last_offset_ns",
    ]
    with (out / "lidar_index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with (out / "imu.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source_file",
                "source_imu",
                "payload_byte_offset",
                "payload_bytes",
                "record_ns",
                "header_ns",
                "header_seq",
                "frame_id",
                "gx",
                "gy",
                "gz",
                "ax",
                "ay",
                "az",
            ]
        )
        w.writerows(imurows)
    np.savez_compressed(out / "raw_index_decode_samples.npz", **sampled_arrays)
    report["counts"] = dict(types_total)
    report["lidar_frames"] = len(rows)
    report["imu_messages"] = len(imurows)
    report["raw_lidar_points"] = sum(x[12] for x in rows)
    for name, r, col in [("lidar", rows, 7), ("imu", imurows, 5)]:
        t = np.array([x[col] for x in r], dtype=np.int64)
        dt = np.diff(t)
        report[name + "_timing"] = {
            "first_ns": int(t[0]),
            "last_ns": int(t[-1]),
            "duration_s": float((t[-1] - t[0]) / 1e9),
            "nonpositive_steps": int((dt <= 0).sum()),
            "dt_s_quantiles": np.quantile(dt / 1e9, [0, 0.01, 0.5, 0.99, 1]).tolist(),
        }
    iv = np.array(all_imu)
    report["imu_values"] = {
        "finite_fraction": float(np.isfinite(iv).all(axis=1).mean()),
        "acceleration_norm_quantiles_native": np.quantile(
            np.linalg.norm(iv[:, 3:], axis=1), [0, 0.01, 0.5, 0.99, 1]
        ).tolist(),
        "gyro_norm_quantiles": np.quantile(
            np.linalg.norm(iv[:, :3], axis=1), [0, 0.01, 0.5, 0.99, 1]
        ).tolist(),
    }
    diff = np.array([x[7] - x[8] for x in rows], dtype=np.int64)
    rdiff = np.array([x[6] - x[7] for x in rows], dtype=np.int64)
    report["lidar_header_minus_timebase_ns_quantiles"] = np.quantile(diff, [0, 0.5, 1]).tolist()
    report["lidar_record_minus_header_ns_quantiles"] = np.quantile(rdiff, [0, 0.5, 1]).tolist()
    report["metadata_files"] = [
        {"file": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in [
            root / "metadata.yaml",
            *(root / "info").glob("*.json"),
            root / "log.7z",
            root / "camera/manual_capture.csv",
        ]
        if p.is_file()
    ]
    cams = {}
    for side in ["left", "right"]:
        images = sorted((root / "camera" / side).glob("*.png"))
        times = np.array([int(p.stem) for p in images], dtype=np.int64)
        if not len(times):
            cams[side] = {"count": 0}
            continue
        cams[side] = {
            "count": len(images),
            "bytes": sum(p.stat().st_size for p in images),
            "first_ns": int(times[0]),
            "last_ns": int(times[-1]),
            "outside_lidar_span": int(((times < rows[0][7]) | (times > rows[-1][7])).sum()),
        }
    report["cameras"] = cams
    report["unknown_types"] = [
        x for x in types_total if x != "sensor_msgs/Imu" and "CustomMsg" not in x
    ]
    report["outputs"] = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
        for p in [out / "lidar_index.csv", out / "imu.csv", out / "raw_index_decode_samples.npz"]
    }
    (out / "raw_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "counts",
                    "lidar_frames",
                    "imu_messages",
                    "raw_lidar_points",
                    "lidar_timing",
                    "imu_timing",
                    "cameras",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
