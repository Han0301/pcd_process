#!/usr/bin/env python3
"""将去畸变帧定位到同次录制的 LAS 几何，并检查留出误差和连续运动。

定位用于坐标一致性，不能作为独立的绝对精度检验。
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SCAN = HERE = LIO = SOURCE = None


def read_pcd(path):
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def vox_ids(p, r):
    _, idx = np.unique(np.floor(p / r).astype(np.int32), axis=0, return_index=True)
    return np.sort(idx)


def transform(p, T):
    return p @ T[:3, :3].T + T[:3, 3]


def stats(d):
    d = np.asarray(d)
    if not len(d):
        return {"count": 0, "median_m": None, "p95_m": None, "within_10cm": 0.0, "within_15cm": 0.0}
    return {
        "count": len(d),
        "median_m": float(np.median(d)),
        "p95_m": float(np.quantile(d, 0.95)),
        "within_10cm": float((d < 0.10).mean()),
        "within_15cm": float((d < 0.15).mean()),
    }


def target_cache():
    cache = HERE / "target_12cm.npz"
    source_sha = sha(SOURCE)
    if cache.exists():
        ar = np.load(cache)
        if str(ar["source_sha256"]) != source_sha:
            raise ValueError("target source mismatch")
        return (
            ar["points"].astype("f8"),
            ar["normals"].astype("f8"),
            ar["source_voxel_row"],
            source_sha,
        )
    source = np.asarray(read_pcd(SOURCE)[:, :3], dtype="f4")
    mask = np.isfinite(source).all(axis=1) & (source[:, 2] > -1.5) & (source[:, 2] < 3.65)
    ids = np.flatnonzero(mask)
    ids = ids[vox_ids(source[ids], 0.12)]
    p = source[ids].astype("f8")
    tree = cKDTree(p)
    normals = np.empty_like(p)
    curv = np.empty(len(p))
    far = np.empty(len(p))
    for start in range(0, len(p), 25000):
        d, nn = tree.query(p[start : start + 25000], k=18, workers=4)
        q = p[nn]
        q -= q.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", q, q) / 18
        e, v = np.linalg.eigh(cov)
        normals[start : start + len(q)] = v[:, :, 0]
        curv[start : start + len(q)] = e[:, 0] / np.maximum(e.sum(axis=1), 1e-12)
        far[start : start + len(q)] = d[:, -1]
    keep = (
        (curv < 0.015) & (far < 0.40) & ((abs(normals[:, 2]) > 0.95) | (abs(normals[:, 2]) < 0.30))
    )
    p, n, ids = p[keep], normals[keep], ids[keep]
    np.savez_compressed(
        cache,
        points=p.astype("f4"),
        normals=n.astype("f4"),
        source_voxel_row=ids,
        source_sha256=source_sha,
    )
    write(
        HERE / "target_manifest.json",
        {
            "source": str(SOURCE),
            "source_sha256": source_sha,
            "target_points": len(p),
            "voxel_m": 0.12,
            "normal_neighbors": 18,
            "curvature_max": 0.015,
            "neighbor18_distance_max_m": 0.4,
            "z_band_m": [-1.5, 3.65],
            "normal_horizontal_vertical_filter": [0.95, 0.3],
            "opaque_material_verified": False,
            "source_modified": False,
            "cache_is_pose_geometry_only": True,
        },
    )
    return p, n, ids, source_sha


def load():
    frames = list(csv.DictReader((LIO / "frames_with_raw_indices.csv").open()))
    poses = {
        r["frame_id"]: np.array([float(r[f"m{i}{j}"]) for i in range(4) for j in range(4)]).reshape(
            4, 4
        )
        for r in csv.DictReader((LIO / "trajectory.csv").open())
    }
    G = np.array(
        json.loads((LIO / "raw_index_shape_validation.json").read_text())[
            "T_gravity_from_lio_world"
        ]
    )
    return frames, poses, G


def frame_split(row):
    raw = np.fromfile(LIO / row["file"], dtype="<f4").reshape(-1, 4)
    rawids = np.fromfile(LIO / row["raw_index_file"], dtype="<u4")
    if len(raw) != len(rawids):
        raise ValueError("index length")
    xyz = raw[:, :3].astype("f8")
    ran = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (ran > 0.7) & (ran < 10)
    xyz, rawids = xyz[valid], rawids[valid]
    tile = np.floor(xyz / 0.24).astype(np.int64)
    tileid = tile[:, 0] * 73856093 ^ tile[:, 1] * 19349663 ^ tile[:, 2] * 83492791
    held = (tileid % 3) == 0
    train_ids = np.flatnonzero(~held)
    test_ids = np.flatnonzero(held)
    train_ids = train_ids[vox_ids(xyz[train_ids], 0.12)]
    test_ids = test_ids[vox_ids(xyz[test_ids], 0.12)]
    if len(train_ids) > 4000:
        train_ids = train_ids[np.linspace(0, len(train_ids) - 1, 4000).astype(int)]
    if len(test_ids) > 2200:
        test_ids = test_ids[np.linspace(0, len(test_ids) - 1, 2200).astype(int)]
    if np.intersect1d(rawids[train_ids], rawids[test_ids]).size:
        raise AssertionError("train holdout overlap")
    return xyz[train_ids], xyz[test_ids], rawids[train_ids], rawids[test_ids]


def robust_p2p(source, target, tree, T0, iterations=35):
    T = T0.copy()
    hist = []
    for it in range(iterations):
        current = transform(source, T)
        d, idx = tree.query(current, workers=4)
        gate = max(0.10, 0.6 * 0.9**it)
        ok = d < gate
        if ok.sum() < 150:
            break
        ok &= d <= np.quantile(d[ok], 0.8)
        a, b = current[ok], target[idx[ok]]
        am, bm = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - am).T @ (b - bm))
        R = vt.T @ u.T
        if np.linalg.det(R) < 0:
            vt[-1] *= -1
            R = vt.T @ u.T
        dt = bm - R @ am
        inc = np.eye(4)
        inc[:3, :3] = R
        inc[:3, 3] = dt
        T = inc @ T
        hist.append(
            {"iteration": it, "correspondences": int(ok.sum()), "median_m": float(np.median(d))}
        )
        if np.linalg.norm(dt) < 2e-5 and Rotation.from_matrix(R).magnitude() < 2e-5:
            break
    return T, hist


def bootstrap(frames, poses, G, target, tree):
    path = HERE / "bootstrap.json"
    if path.exists():
        return np.array(json.loads(path.read_text())["T_las_from_lio_world"])
    clouds = []
    hold = []
    for ii, row in enumerate(frames):
        if float(row["stamp"]) - float(frames[0]["stamp"]) >= 120:
            break
        if ii % 4:
            continue
        train, test, _, _ = frame_split(row)
        if ii % 3 == 0:
            hold.append(transform(np.vstack([train, test]), poses[row["frame_id"]]))
        else:
            clouds.append(transform(train, poses[row["frame_id"]]))
    source = np.concatenate(clouds)
    source = source[vox_ids(transform(source, G), 0.18)]
    if len(source) > 45000:
        source = source[np.linspace(0, len(source) - 1, 45000).astype(int)]
    candidates = []
    for yaw in [0, -90, 90, 180]:
        initial = G.copy()
        initial[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ G[:3, :3]
        # Candidate ranking uses training geometry only; wrong headings are refusal controls.
        T, h = robust_p2p(source, target, tree, initial)
        candidates.append(
            {
                "seed_yaw_deg": yaw,
                "T": T.tolist(),
                "training": stats(tree.query(transform(source, T), workers=4)[0]),
                "history": h,
            }
        )
    candidates.sort(key=lambda x: (-x["training"]["within_15cm"], x["training"]["median_m"]))
    T = np.array(candidates[0]["T"])
    test = np.concatenate(hold)
    test = test[vox_ids(transform(test, G), 0.18)]
    if len(test) > 45000:
        test = test[np.linspace(0, len(test) - 1, 45000).astype(int)]
    before, idx = tree.query(transform(test, G), workers=4)
    after = tree.query(transform(test, T), workers=4)[0]
    fixed = np.linalg.norm(transform(test, T) - target[idx], axis=1)
    candidate_T = T.copy()
    fixed_degrades = float(np.quantile(fixed, 0.95)) > float(np.quantile(before, 0.95)) + 0.002
    if fixed_degrades:
        # A denser NN rematch can improve while frozen pairs worsen. Keep the
        # measured gravity seed and defer corrections to independently gated frames.
        T = G.copy()
    r = {
        "T_las_from_lio_world": T.tolist(),
        "candidate_T_las_from_lio_world": candidate_T.tolist(),
        "bootstrap_candidate_adopted": not fixed_degrades,
        "bootstrap_decision": "reject candidate: frozen point pairs worsened; retain gravity seed"
        if fixed_degrades
        else "candidate passed frozen point-pair non-worsening",
        "gravity_initial": G.tolist(),
        "candidates": candidates,
        "training_points": len(source),
        "heldout_points": len(test),
        "holdout_before": stats(before),
        "holdout_after_rematched": stats(after),
        "holdout_after_fixed": stats(fixed),
        "training_selection": "first 120s, every fourth export frame, frame index modulo3 !=0; per-frame heldout tiles excluded",
        "holdout_selection": "first120s every fourth export frame, frame index modulo3 ==0; never used for candidate ranking",
        "coarse_source": "raw LIO initial gravity transform, translation0; verified by independent nearest-neighbor probe before optimization",
        "same_capture_target_contains_source_observations": True,
        "is_independent_accuracy_validation": False,
        "source_map_modified": False,
    }
    write(path, r)
    np.save(HERE / "T_las_from_lio_world_initial.npy", T)
    return T


def solve(source, T0, target, normals, tree):
    T = T0.copy()
    last = {}
    valid = None
    for it in range(20):
        q = source @ T[:3, :3].T
        current = q + T[:3, 3]
        d, idx = tree.query(current, workers=4, distance_upper_bound=0.30)
        idx = np.minimum(idx, len(target) - 1)
        n = normals[idx]
        residual = np.einsum("ij,ij->i", current - target[idx], n)
        valid = (d < 0.30) & (abs(residual) < 0.12)
        if valid.sum() < 150:
            return T, {"fit_ok": False, "correspondences": int(valid.sum())}
        nv = n[valid]
        rv = residual[valid]
        J = np.column_stack([np.cross(q[valid], nv), nv])
        weights = np.minimum(1, 0.025 / np.maximum(abs(rv), 1e-12))
        axis = np.argmax(abs(nv), axis=1)
        counts = np.bincount(axis, minlength=3)
        weights *= np.minimum(3, np.sqrt(len(axis) / np.maximum(3 * counts[axis], 1)))
        H = (J.T * weights) @ J
        rhs = -(J.T * weights) @ rv
        scale = np.sqrt(np.maximum(np.diag(H), 1e-12))
        ev = np.linalg.eigvalsh(H / scale[:, None] / scale[None, :])
        ne = np.linalg.eigvalsh(nv.T @ nv / len(nv))
        last = {
            "fit_ok": True,
            "iterations": it + 1,
            "correspondences": int(valid.sum()),
            "training_correspondence_fraction": float(valid.mean()),
            "hessian_scaled_condition": float(ev[-1] / max(ev[0], 1e-12)),
            "hessian_scaled_min_eigenvalue": float(ev[0]),
            "translation_normal_eigenvalues": ne.tolist(),
        }
        delta = np.linalg.lstsq(H + np.eye(6) * 1e-7, rhs, rcond=1e-7)[0]
        delta *= min(
            1,
            0.08 / max(np.linalg.norm(delta[3:]), 1e-10),
            np.deg2rad(1) / max(np.linalg.norm(delta[:3]), 1e-10),
        )
        T[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ T[:3, :3]
        T[:3, 3] += delta[3:]
        if np.linalg.norm(delta[:3]) < 2e-5 and np.linalg.norm(delta[3:]) < 1e-4:
            break
    return T, last


def evaluate(test, T0, T, target, normals, tree, fulltree):
    b = transform(test, T0)
    a = transform(test, T)
    bd, bi = fulltree.query(b, workers=4)
    ad = fulltree.query(a, workers=4)[0]
    fd, fi = tree.query(b, workers=4, distance_upper_bound=0.20)
    support = np.isfinite(fd)
    fi = np.minimum(fi, len(target) - 1)
    beforep = abs(np.einsum("ij,ij->i", b - target[fi], normals[fi]))
    afterp = abs(np.einsum("ij,ij->i", a - target[fi], normals[fi]))
    afters = tree.query(a, workers=4)[0]
    return {
        "holdout_before": stats(bd),
        "holdout_after": stats(ad),
        "fixed_holdout_support_fraction": float(support.mean()),
        "fixed_holdout_plane_p95_before_m": float(np.quantile(beforep[support], 0.95))
        if support.any()
        else None,
        "fixed_holdout_plane_p95_after_m": float(np.quantile(afterp[support], 0.95))
        if support.any()
        else None,
        "holdout_structure_within20cm": float((afters < 0.20).mean()),
    }


def gates(q, T0, T):
    reasons = []
    corr = float(np.linalg.norm(T[:3, 3] - T0[:3, 3]))
    angle = float(np.rad2deg(Rotation.from_matrix(T[:3, :3] @ T0[:3, :3].T).magnitude()))
    if not q.get("fit_ok"):
        reasons.append("insufficient_correspondences")
    # Plane-anchor support is a subset of the full scan (furniture is not planar).
    # Full-cloud holdout coverage below remains a separate strict requirement.
    if q.get("training_correspondence_fraction", 0) < 0.25:
        reasons.append("training_support")
    if (
        q.get("hessian_scaled_condition", 1e12) > 1000
        or q.get("hessian_scaled_min_eigenvalue", 0) < 0.001
    ):
        reasons.append("degenerate_hessian")
    if min(q.get("translation_normal_eigenvalues", [0])) < 0.01:
        reasons.append("insufficient_normal_directions")
    if q["holdout_after"]["within_10cm"] < 0.90 or q["holdout_after"]["p95_m"] > 0.15:
        reasons.append("holdout_full_geometry")
    if q["fixed_holdout_support_fraction"] < 0.20 or q["holdout_structure_within20cm"] < 0.20:
        reasons.append("heldout_structure_support")
    pa, pb = q["fixed_holdout_plane_p95_after_m"], q["fixed_holdout_plane_p95_before_m"]
    if pa is None or pa > 0.06:
        reasons.append("fixed_plane_residual")
    if pa is not None and pb is not None and pa > pb + 0.002:
        reasons.append("fixed_plane_worsened")
    if q["holdout_after"]["p95_m"] > q["holdout_before"]["p95_m"] + 0.002:
        reasons.append("euclidean_p95_worsened")
    if corr > 0.15 or angle > 1.5:
        reasons.append("large_correction")
    if not np.isfinite(T).all() or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-6:
        reasons.append("invalid_SE3")
    return reasons, corr, angle


def main():
    global SCAN, HERE, LIO, SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "full"], default="probe")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--input-pcd", type=Path, required=True)
    args = ap.parse_args()
    SCAN = args.work_dir
    HERE = SCAN / "localization"
    LIO = SCAN / "lio"
    SOURCE = args.input_pcd
    t = time.monotonic()
    target, normals, source_rows, source_sha = target_cache()
    tree = cKDTree(target)
    full = np.asarray(read_pcd(SOURCE)[:, :3], dtype="f8")
    fulltree = cKDTree(full)
    frames, poses, G = load()
    T0 = bootstrap(frames, poses, G, full, fulltree)
    indices = (
        sorted(set([0, len(frames) - 1] + list(range(0, len(frames), 120))))
        if args.mode == "probe"
        else list(range(len(frames)))
    )
    out = HERE / args.mode
    out.mkdir(exist_ok=True)
    results = []
    for count, idx in enumerate(indices):
        row = frames[idx]
        train, test, trids, teids = frame_split(row)
        initial = T0 @ poses[row["frame_id"]]
        T, fit = solve(train, initial, target, normals, tree)
        q = evaluate(test, initial, T, target, normals, tree, fulltree)
        q.update(fit)
        reasons, correction, angle = gates(q, initial, T)
        result = {
            **row,
            "export_sample_index": idx,
            "relative_stamp": float(row["stamp"]) - float(frames[0]["stamp"]),
            "T_las_from_lidar": T.tolist(),
            "T_initial_las_from_lidar": initial.tolist(),
            "accepted_geometry": not reasons,
            "accepted_for_free_space": False,
            "optical_and_door_guards_pass": False,
            "rejection_reasons": reasons,
            "correction_translation_m": correction,
            "correction_rotation_deg": angle,
            "training_raw_indices_sha256": hashlib.sha256(
                trids.astype("<u4").tobytes()
            ).hexdigest(),
            "holdout_raw_indices_sha256": hashlib.sha256(teids.astype("<u4").tobytes()).hexdigest(),
            "training_holdout_raw_overlap": 0,
            **q,
        }
        results.append(result)
        if count % 50 == 0:
            print(
                "PROGRESS",
                args.mode,
                count,
                len(indices),
                "geometry accepted",
                sum(x["accepted_geometry"] for x in results),
                "wall",
                time.monotonic() - t,
                flush=True,
            )
    # A frame fit alone does not establish a continuous replay. Keep a separate
    # conservative temporal flag; it still cannot authorize free-space deletion.
    motion_bad = set()
    motion_edges = []
    if args.mode == "full":
        for i in range(1, len(results)):
            a, b = results[i - 1], results[i]
            if not (a["accepted_geometry"] and b["accepted_geometry"]):
                continue
            dt = b["relative_stamp"] - a["relative_stamp"]
            A, B = np.array(a["T_las_from_lidar"]), np.array(b["T_las_from_lidar"])
            la, lb = poses[a["frame_id"]], poses[b["frame_id"]]
            measured = np.linalg.inv(A) @ B
            predicted = np.linalg.inv(la) @ lb
            translation_delta = float(np.linalg.norm(measured[:3, 3] - predicted[:3, 3]))
            angle_delta = float(
                np.rad2deg(Rotation.from_matrix(measured[:3, :3] @ predicted[:3, :3].T).magnitude())
            )
            speed = float(np.linalg.norm(measured[:3, 3]) / max(dt, 1e-10))
            passed = (
                0.35 < dt < 0.65
                and translation_delta <= 0.10
                and angle_delta <= 0.8
                and speed <= 2.0
            )
            motion_edges.append(
                {
                    "a_index": i - 1,
                    "b_index": i,
                    "dt": dt,
                    "translation_increment_difference_m": translation_delta,
                    "rotation_increment_difference_deg": angle_delta,
                    "speed_mps": speed,
                    "pass": passed,
                }
            )
            if not passed:
                motion_bad.update([i - 1, i])
    for i, r in enumerate(results):
        neighbor = (
            any(
                0 <= j < len(results) and results[j]["accepted_geometry"] and j not in motion_bad
                for j in [i - 1, i + 1]
            )
            if args.mode == "full"
            else False
        )
        r["accepted_temporal_geometry"] = bool(
            r["accepted_geometry"] and i not in motion_bad and neighbor
        )
        r["temporal_rejection_reasons"] = []
        if args.mode != "full":
            r["temporal_rejection_reasons"].append("sparse_probe_no_adjacent_check")
        elif i in motion_bad:
            r["temporal_rejection_reasons"].append("adjacent_motion_inconsistency")
        elif not neighbor:
            r["temporal_rejection_reasons"].append("no_adjacent_accepted_geometry")
        r["holdout_distance_p95_m"] = r["holdout_after"]["p95_m"]
        r["holdout_within10cm"] = r["holdout_after"]["within_10cm"]
    write(out / "motion_edges.json", motion_edges)
    write(out / "localized_frames.json", results)
    fields0 = list(frames[0]) + [
        "export_sample_index",
        "relative_stamp",
        "accepted_geometry",
        "accepted_for_free_space",
        "accepted_temporal_geometry",
        "temporal_rejection_reasons",
        "optical_and_door_guards_pass",
        "rejection_reasons",
        "correction_translation_m",
        "correction_rotation_deg",
        "training_raw_indices_sha256",
        "holdout_raw_indices_sha256",
        "training_holdout_raw_overlap",
        "training_correspondence_fraction",
        "hessian_scaled_condition",
        "hessian_scaled_min_eigenvalue",
        "fixed_holdout_support_fraction",
        "fixed_holdout_plane_p95_before_m",
        "fixed_holdout_plane_p95_after_m",
        "holdout_structure_within20cm",
        "holdout_distance_p95_m",
        "holdout_within10cm",
    ]
    fields0 += [f"m{i}{j}" for i in range(4) for j in range(4)]
    with (out / "localized_frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields0)
        w.writeheader()
        for r in results:
            a = {k: r[k] for k in fields0 if k in r}
            a["rejection_reasons"] = ";".join(r["rejection_reasons"])
            a["temporal_rejection_reasons"] = ";".join(r["temporal_rejection_reasons"])
            a.update({f"m{i}{j}": r["T_las_from_lidar"][i][j] for i in range(4) for j in range(4)})
            w.writerow(a)
    r = {
        "mode": args.mode,
        "frames": len(results),
        "accepted_geometry": sum(x["accepted_geometry"] for x in results),
        "accepted_temporal_geometry": sum(x["accepted_temporal_geometry"] for x in results),
        "accepted_for_free_space": 0,
        "source_las_voxel_sha256": source_sha,
        "target_is_same_capture": True,
        "independent_accuracy_validation": False,
        "materials_verified": False,
        "static_points_certified": False,
        "optical_guards_pending": True,
        "source_clouds_or_trajectory_modified": False,
        "training_holdout_split": "disjoint lidar-local 24cm tiles hashed modulo3; raw row IDs retained",
        "wall_seconds": time.monotonic() - t,
        "reasons": {},
        "T_las_from_lio_world_initial": T0.tolist(),
    }
    for x in results:
        for reason in x["rejection_reasons"]:
            r["reasons"][reason] = r["reasons"].get(reason, 0) + 1
    write(out / "summary.json", r)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
