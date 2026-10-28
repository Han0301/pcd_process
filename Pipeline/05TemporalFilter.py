"""融合前逐份执行的多帧命中、实际射线与低密度筛选。

输入：同次录制的米制 PCD、已去畸变的单帧 PCD/BIN 和合格位姿 CSV。
BIN 为小端 float32 N×4 XYZ/intensity，CSV 必须声明 point_format。
位姿方向为单帧到当前点云；left_transform_file 可追加本次调平变换。
mode=review 保留全部点并标记候选；mode=apply 按证据剔除。
地面、水平结构、布尔保护字段与有限保护框限制删点范围。
保留坐标和来源属性，不额外降采样；稀疏统计排除不等于已确认运动。
输出 cleaned.pcd 及旁文件、候选复核云、temporal_evidence.npz 和报告。
"""

import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parents[1])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import csv
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from Common.Configuration import cli_config, complete_parameters, load_config
from Common.PointCloudIO import (
    read_cloud,
    sha256,
    sidecar_path,
    validate_transform,
    write_cloud,
    write_report,
)


# 帧、坐标与结构保护
def _true(value):
    value = str(value).strip().lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError(f"质量标志只能为True/False或1/0：{value}")
    return value in ("true", "1")


def _frames(path, params):
    required = {
        "file",
        "stamp",
        "accepted_geometry",
        "accepted_for_free_space",
        "holdout_plane_p95_m",
        "holdout_distance_p95_m",
    }
    required |= {f"m{i}{j}" for i in range(4) for j in range(4)}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"位姿CSV缺少列：{sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    result = []
    for index, row in enumerate(rows):
        if not _true(row["accepted_geometry"]):
            continue
        source = Path(row["file"]).expanduser()
        source = (path.parent / source).resolve() if not source.is_absolute() else source.resolve()
        if source.suffix.lower() not in (".pcd", ".bin") or not source.is_file():
            raise ValueError(f"第{index + 2}行单帧必须是已存在的PCD或BIN：{source}")
        point_format = row.get("point_format", "")
        if source.suffix.lower() == ".bin":
            if point_format != "xyzi_float32_le":
                raise ValueError("BIN单帧必须显式声明point_format=xyzi_float32_le")
            if source.stat().st_size == 0 or source.stat().st_size % 16:
                raise ValueError(f"BIN单帧必须为非空的小端float32 N×4：{source}")
        matrix = validate_transform([[float(row[f"m{i}{j}"]) for j in range(4)] for i in range(4)])
        left = row.get("left_transform_file", "").strip()
        transform_file = None
        if left:
            transform_file = Path(left).expanduser()
            if not transform_file.is_absolute():
                transform_file = path.parent / transform_file
            transform_file = transform_file.resolve()
            matrix = validate_transform(np.load(transform_file, allow_pickle=False)) @ matrix
            matrix = validate_transform(matrix)
        stamp = float(row["stamp"])
        errors = np.array([float(row["holdout_plane_p95_m"]), float(row["holdout_distance_p95_m"])])
        if not np.isfinite(stamp) or np.any(errors < 0):
            raise ValueError("帧时刻无效或残差为负数")
        free_ok = (
            _true(row["accepted_for_free_space"])
            and np.isfinite(errors).all()
            and errors[0] <= params["max_plane_p95"]
            and errors[1] <= params["max_nn_p95"]
        )
        result.append(
            {
                "file": source,
                "stamp": stamp,
                "matrix": matrix,
                "free_ok": bool(free_ok),
                "csv_row": index + 2,
                "point_format": point_format,
                "left_transform_file": transform_file,
            }
        )
    result.sort(key=lambda r: r["stamp"])
    if len(result) < params["independent_views"]:
        raise ValueError(f"合格帧少于 {params['independent_views']}，不能构造独立多帧证据")
    stamps = np.array([r["stamp"] for r in result])
    if np.any(np.diff(stamps) <= 0) or len({r["file"] for r in result}) != len(result):
        raise ValueError("重复时刻或重复单帧文件，拒绝重复投票")
    return result, len(rows) - len(result)


def _horizontal(points, tree, params):
    guard = np.zeros(len(points), bool)
    if len(points) < params["normal_neighbors"]:
        return guard
    for start in range(0, len(points), 8192):
        block = points[start : start + 8192]
        distance, rows = tree.query(block, k=params["normal_neighbors"], workers=1)
        local = points[rows]
        local -= local.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(
            np.einsum("nki,nkj->nij", local, local) / params["normal_neighbors"]
        )
        values = np.maximum(values, 0)
        curve = values[:, 0] / np.maximum(values.sum(axis=1), 1e-18)
        guard[start : start + len(block)] = (
            (distance[:, -1] <= params["normal_radius"])
            & (values[:, 1] > 1e-10)
            & (np.abs(vectors[:, 2, 0]) >= params["min_normal_z"])
            & (curve <= params["max_curvature"])
        )
    return guard


def _protected(cloud, tree, ground_z, bounds, horizontal, params):
    masks = {
        "ground_height_guard": np.abs(cloud.points[:, 2] - ground_z)
        <= params["ground_guard_height"]
    }
    for key in params["protected_fields"]:
        if key in cloud.extras:
            mask = cloud.extras[key]
            if mask.shape != (len(cloud),) or mask.dtype != np.dtype(bool):
                raise ValueError(f"保护字段必须是长度N的布尔数组：{key}")
            masks[key] = mask
    if horizontal:
        masks["horizontal_geometry_guard"] = _horizontal(cloud.points, tree, params)
        dilation = params["horizontal_guard_dilation"]
        if dilation > 0 and masks["horizontal_geometry_guard"].any():
            guard_tree = cKDTree(cloud.points[masks["horizontal_geometry_guard"]])
            expanded = np.zeros(len(cloud), bool)
            for start in range(0, len(cloud), 100000):
                distance, _ = guard_tree.query(
                    cloud.points[start : start + 100000], distance_upper_bound=dilation, workers=1
                )
                expanded[start : start + len(distance)] = np.isfinite(distance)
            masks["horizontal_proximity_guard"] = expanded
    boxes = np.asarray(bounds, dtype=float)
    boxes = boxes.reshape(0, 6) if boxes.size == 0 else boxes
    if (
        boxes.ndim != 2
        or boxes.shape[1] != 6
        or not np.isfinite(boxes).all()
        or np.any(boxes[:, ::2] >= boxes[:, 1::2])
    ):
        raise ValueError("保护框须为有限的(xmin,xmax,ymin,ymax,zmin,zmax)，下界小于上界")
    box_mask = np.zeros(len(cloud), bool)
    for box in boxes:
        box_mask |= ((cloud.points >= box[::2]) & (cloud.points <= box[1::2])).all(axis=1)
    masks["finite_region_guard"] = box_mask
    return np.logical_or.reduce(list(masks.values())), masks, boxes


def _blocked(points, origin, boxes):
    """疑区框既保护内部点，也阻止利用穿过疑区的射线删除其后方点。"""
    blocked = np.zeros(len(points), bool)
    delta = points - origin
    parallel = np.abs(delta) < 1e-12
    for box in boxes:
        a = np.divide(box[::2] - origin, delta, out=np.zeros_like(delta), where=~parallel)
        b = np.divide(box[1::2] - origin, delta, out=np.zeros_like(delta), where=~parallel)
        enter = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=1)
        leave = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=1)
        outside = (parallel & ((origin < box[::2]) | (origin > box[1::2]))).any(axis=1)
        blocked |= ~outside & (leave >= enter) & (leave > 1e-6) & (enter < 1 - 1e-6)
    return blocked


def _vote(
    ids, counts, witnesses, frame, origins, stamps, params, endpoint_rows=None, endpoints=None
):
    """同一帧每点一票；三个见证的时间和原点必须两两独立。"""
    for level in range(params["independent_views"] - 1, -1, -1):
        loc = np.flatnonzero(counts[ids] == level)
        selected = ids[loc]
        valid = np.ones(len(selected), bool)
        for previous in range(level):
            last = witnesses[selected, previous]
            valid &= (
                np.linalg.norm(origins[last] - origins[frame], axis=1)
                >= params["min_origin_distance"]
            )
            valid &= np.abs(stamps[last] - stamps[frame]) >= params["min_time_distance"]
        selected, loc = selected[valid], loc[valid]
        witnesses[selected, level] = frame
        if endpoints is not None:
            endpoints[selected, level] = endpoint_rows[loc]
        counts[selected] += 1


def _read_frame(row, params):
    if row["file"].suffix.lower() == ".bin":
        if row.get("point_format") != "xyzi_float32_le":
            raise ValueError("未声明的BIN单帧格式")
        values = np.fromfile(row["file"], dtype="<f4")
        if not len(values) or len(values) % 4:
            raise ValueError(f"BIN单帧长度无效：{row['file']}")
        points = values.reshape(-1, 4)[:, :3].astype(np.float64)
    else:
        points = read_cloud(row["file"]).points
    distance = np.linalg.norm(points, axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & (distance > params["frame_range"][0])
        & (distance < params["frame_range"][1])
    )
    return points[valid], np.flatnonzero(valid)


def _ray_query(delta, target_range, rays, params):
    """按距离分组限制角度查询半径；只跳过不可能满足原距离门槛的近邻。"""
    length = np.linalg.norm(rays, axis=1)
    tree = cKDTree(rays / length[:, None])
    unit = delta / target_range[:, None]
    free = np.zeros(len(delta), bool)
    nearest = np.full(len(delta), -1, dtype=np.int64)
    bands = np.floor(np.log2(target_range)).astype(np.int64)
    for band in np.unique(bands):
        rows = np.flatnonzero(bands == band)
        # 自由判据要求 d*r < ray_distance；组内最小r给出保守查询上界。
        # 向上扩展一ULP，避免除法舍入把门槛内的点排除。
        bound = np.nextafter(params["ray_distance"] / target_range[rows].min(), np.inf)
        distance, index = tree.query(unit[rows], distance_upper_bound=bound, workers=1)
        finite = np.isfinite(distance)
        selected = rows[finite]
        nearest[selected] = index[finite]
        free[selected] = (distance[finite] * target_range[selected] < params["ray_distance"]) & (
            length[index[finite]] > target_range[selected] + params["endpoint_margin"]
        )
    return free, nearest


def _sparse_proposal(cloud, tree, hit_frames, protected, stable, params):
    """低密度与弱命中的联合统计排除；不是自由射线或运动身份判定。"""
    proposed = np.zeros(len(cloud), bool)
    rows = np.flatnonzero(
        (cloud.raw_count <= params["sparse_raw_count_max"])
        & (hit_frames <= params["sparse_max_hit_frames"])
        & ~protected
        & ~stable
    )
    nearest, eighth, stable_distance = (np.full(len(rows), np.inf) for _ in range(3))
    if params["sparse_filter"] and len(rows):
        distance, _ = tree.query(cloud.points[rows], k=8, workers=1)
        nearest, eighth = distance[:, 1], distance[:, 7]
        if stable.any():
            stable_distance = cKDTree(cloud.points[stable]).query(cloud.points[rows], workers=1)[0]
        selected = (
            (nearest > params["sparse_nearest_distance"])
            & (eighth > params["sparse_eighth_distance"])
            & (stable_distance > params["sparse_stable_distance"])
        )
        proposed[rows[selected]] = True
    return proposed, rows, nearest, eighth, stable_distance


# 两遍证据统计与可回溯输出
def temporal_filter(input_pcd, frames_csv, output_dir, **parameters):
    """按帧质量、结构保护与 YAML 参数统计候选，返回可回溯的筛选报告。"""
    params = complete_parameters("Pipeline", "TemporalFilter", parameters)
    mode, ground_z = params["mode"], params["ground_z"]
    protected_bounds, protect_horizontal = params["protected_bounds"], params["protect_horizontal"]
    source, poses, out = (
        Path(p).expanduser().resolve() for p in (input_pcd, frames_csv, output_dir)
    )
    if mode not in ("review", "apply") or not np.isfinite(ground_z):
        raise ValueError("mode必须为review/apply，ground_z必须有限")
    if out.exists():
        raise FileExistsError(f"输出目录须为新目录：{out}")
    cloud = read_cloud(source)
    if not len(cloud) or not np.isfinite(cloud.points).all():
        raise ValueError("当前PCD为空或含无效坐标，请先预处理")
    if len(np.unique(cloud.scan_id)) > 1 or "fusion_input_id" in cloud.extras:
        raise ValueError("不能用单次录制的帧筛选多源融合云；请在融合前逐份执行")
    if any(k.startswith("temporal_") and k != "temporal_protected" for k in cloud.extras):
        raise ValueError("当前PCD已有时序结果；请从未筛选的输入重新运行，避免重复剔除")
    frames, rejected_frames = _frames(poses, params)
    inputs = {str(p): sha256(p) for p in [source, poses]}
    if sidecar_path(source).exists():
        inputs[str(sidecar_path(source))] = sha256(sidecar_path(source))
    bounds_transform_file = params["protected_bounds_transform_file"]
    if bounds_transform_file is not None:
        path = Path(bounds_transform_file).expanduser().resolve()
        transform = validate_transform(np.load(path, allow_pickle=False))
        inputs[str(path)] = sha256(path)
        transformed_bounds = []
        for box in protected_bounds:
            corners = np.array(list(product(*zip(box[::2], box[1::2]))))
            corners = corners @ transform[:3, :3].T + transform[:3, 3]
            transformed_bounds.append(
                np.column_stack([corners.min(0), corners.max(0)]).ravel().tolist()
            )
        protected_bounds = transformed_bounds
    for row in frames:
        if row["left_transform_file"] is not None:
            inputs[str(row["left_transform_file"])] = sha256(row["left_transform_file"])
    stamps = np.array([r["stamp"] for r in frames])
    stamps -= stamps[0]
    origins = np.array([r["matrix"][:3, 3] for r in frames])
    tree = cKDTree(cloud.points)
    protected, guards, boxes = _protected(
        cloud, tree, ground_z, protected_bounds, protect_horizontal, params
    )
    hit_frames = np.zeros(len(cloud), dtype=np.uint32)
    hit_views = np.zeros(len(cloud), dtype=np.uint8)
    hit_witness = np.full((len(cloud), params["independent_views"]), -1, dtype=np.int32)
    start = time.monotonic()
    for frame, row in enumerate(frames):
        inputs[str(row["file"])] = sha256(row["file"])
        if sidecar_path(row["file"]).exists():
            inputs[str(sidecar_path(row["file"]))] = sha256(sidecar_path(row["file"]))
        points, _ = _read_frame(row, params)
        world = points @ row["matrix"][:3, :3].T + origins[frame]
        distance, matched = tree.query(world, distance_upper_bound=params["hit_radius"], workers=1)
        ids = np.unique(matched[np.isfinite(distance)])
        hit_frames[ids] += 1
        _vote(ids, hit_views, hit_witness, frame, origins, stamps, params)
        if frame % 50 == 0:
            print(f"时序命中：{frame + 1}/{len(frames)}", flush=True)
    stable = hit_views >= params["independent_views"]
    candidates = np.flatnonzero(~protected & ~stable & (hit_frames <= params["ray_max_hit_frames"]))
    free_views = np.zeros(len(candidates), dtype=np.uint8)
    free_witness = np.full((len(candidates), params["independent_views"]), -1, dtype=np.int32)
    endpoint_witness = np.full_like(free_witness, -1)
    blocked_votes = 0
    for frame, row in enumerate(frames):
        if not row["free_ok"]:
            continue
        active = np.flatnonzero(free_views < params["independent_views"])
        delta = cloud.points[candidates[active]] - origins[frame]
        target_range = np.linalg.norm(delta, axis=1)
        valid = (target_range > params["query_range"][0]) & (
            target_range < params["query_range"][1]
        )
        active, delta, target_range = active[valid], delta[valid], target_range[valid]
        if not len(active):
            continue
        points, point_rows = _read_frame(row, params)
        if not len(points):
            continue
        rays = points @ row["matrix"][:3, :3].T
        free, nearest = _ray_query(delta, target_range, rays, params)
        selected, endpoints = active[free], point_rows[nearest[free]]
        blocked = _blocked(cloud.points[candidates[selected]], origins[frame], boxes)
        blocked_votes += int(blocked.sum())
        _vote(
            selected[~blocked],
            free_views,
            free_witness,
            frame,
            origins,
            stamps,
            params,
            endpoint_rows=endpoints[~blocked],
            endpoints=endpoint_witness,
        )
        if frame % 50 == 0:
            print(
                f"时序射线：{frame + 1}/{len(frames)}，独立视角候选{int((free_views == params['independent_views']).sum())}",
                flush=True,
            )
    ray_proposed = np.zeros(len(cloud), bool)
    ray_proposed[candidates[free_views == params["independent_views"]]] = True
    sparse_proposed, sparse_rows, near_distance, eighth_distance, stable_distance = (
        _sparse_proposal(cloud, tree, hit_frames, protected, stable, params)
    )
    proposed = ray_proposed | sparse_proposed
    removed = proposed if mode == "apply" else np.zeros(len(cloud), bool)
    assert not np.any(proposed & (protected | stable))
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise RuntimeError(f"输入在处理过程中变化，拒绝输出：{path}")
    out.mkdir(parents=True, exist_ok=False)
    cloud.extras.update(
        temporal_hit_frames=hit_frames,
        temporal_hit_views=hit_views,
        temporal_guarded=protected,
        temporal_remove_candidate=proposed,
        temporal_ray_candidate=ray_proposed,
        temporal_sparse_candidate=sparse_proposed,
    )
    write_cloud(out / "cleaned.pcd", cloud.subset(~removed))
    write_cloud(out / "rejected_review.pcd", cloud.subset(proposed))
    np.savez_compressed(
        out / "temporal_evidence.npz",
        input_rows=np.arange(len(cloud)),
        input_sha256=np.array(inputs[str(source)]),
        kept_rows=np.flatnonzero(~removed),
        removed_rows=np.flatnonzero(removed),
        proposed_rows=np.flatnonzero(proposed),
        ray_proposed_rows=np.flatnonzero(ray_proposed),
        sparse_proposed_rows=np.flatnonzero(sparse_proposed),
        sparse_query_rows=sparse_rows,
        sparse_nearest_distance=near_distance,
        sparse_eighth_distance=eighth_distance,
        sparse_stable_distance=stable_distance,
        hit_frames=hit_frames,
        hit_views=hit_views,
        hit_witness_frames=hit_witness,
        protected=protected,
        stable=stable,
        query_source_rows=candidates,
        free_views=free_views,
        free_witness_frames=free_witness,
        free_witness_endpoint_rows=endpoint_witness,
        frame_stamps_relative_s=stamps,
        frame_origins=origins,
        frame_matrices=np.array([r["matrix"] for r in frames]),
        frame_files=np.array([str(r["file"]) for r in frames]),
        frame_csv_rows=np.array([r["csv_row"] for r in frames]),
        frame_free_qualified=np.array([r["free_ok"] for r in frames]),
    )
    report = {
        "module": "TemporalFilter",
        "implementation_sha256": sha256(__file__),
        "mode": mode,
        "input": str(source),
        "input_sha256": inputs[str(source)],
        "frames_csv": str(poses),
        "output_sidecar_sha256": sha256(sidecar_path(out / "cleaned.pcd")),
        "output_pcd": str(out / "cleaned.pcd"),
        "output_sha256": sha256(out / "cleaned.pcd"),
        "review_pcd": str(out / "rejected_review.pcd"),
        "evidence": str(out / "temporal_evidence.npz"),
        "evidence_sha256": sha256(out / "temporal_evidence.npz"),
        "input_points": len(cloud),
        "output_points": int((~removed).sum()),
        "proposed_points": int(proposed.sum()),
        "removed_points": int(removed.sum()),
        "ray_proposed_points": int(ray_proposed.sum()),
        "sparse_proposed_points": int(sparse_proposed.sum()),
        "statistical_geometry_filter_performed": bool(params["sparse_filter"]),
        "sparse_exclusions_are_verified_motion": False,
        "protected_points": int(protected.sum()),
        "strong_multiview_points": int(stable.sum()),
        "protected_removed": int((removed & protected).sum()),
        "guard_counts": {k: int(v.sum()) for k, v in guards.items()},
        "protected_bounds_current": boxes.tolist(),
        "geometry_frames": len(frames),
        "rejected_geometry_frames": rejected_frames,
        "strict_free_frames": sum(r["free_ok"] for r in frames),
        "blocked_region_votes": blocked_votes,
        "parameters": params,
        "input_hashes": inputs,
        "inputs_unchanged": True,
        "downsampling_performed": False,
        "statistical_filter_performed": False,
        "coordinates_modified": False,
        "source_attributes_preserved": True,
        "safe_free_cells_generated": False,
        "glass_identity_detected": False,
        "dufo_used": False,
        "limitations": [
            "输出为有保护约束的时序不一致筛选，不证明运动/材质或绝对精度。",
            "玻璃疑区需提供保护字段或有限保护框，不能自动保证未标记玻璃安全。",
            "射线起点使用每帧位姿，不是逐点IMU轨迹；帧内运动带来的起点误差仍需评估。",
            "未采用历史DUFO候选生成，不宣称与v07逐点选择相同。",
        ],
        "elapsed_seconds": time.monotonic() - start,
    }
    write_report(out / "report.json", report)
    print(
        f"时序筛选完成：{len(cloud):,} → {report['output_points']:,}，剔除{report['removed_points']:,}",
        flush=True,
    )
    return report


# 直接运行
def main(config=None):
    config = config or load_config("Pipeline")
    settings = config.section("TemporalFilter")
    return temporal_filter(
        settings["input_pcd"],
        settings["frames_csv"],
        settings["output_dir"],
        **settings["parameters"],
    )


if __name__ == "__main__":
    main(cli_config("Pipeline"))
