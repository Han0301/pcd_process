"""
提取可信地面，按地面平面做一次刚体调平，并保留原始地形起伏。

输入：PCD，单位须为米，Z 轴基本竖直。
输出：完整 leveled.pcd、ground.pcd、完整余集 non_ground.pcd、obstacle_band.pcd。

主地面用“水平法向 + 每个空间桶的低位点 + 高度分层 + 鲁棒平面”估计。
桌面不是因为水平就被认作地面。缺少有效低位支持时明确报错，需要裁出可靠地面后重试。
局部地面仅用于高度分类，不逐点压平，不把全部非地面点都当作障碍。
这是一层室内近水平地面的初步提取方法；楼梯、多楼层、陡坡不适合直接套用默认参数。
"""

import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parents[1])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from Common.Configuration import cli_config, configured_defaults, load_config
from Common.PointCloudIO import (
    read_cloud,
    sha256,
    transform_cloud,
    validate_transform,
    write_cloud,
    write_report,
)

# 法向和低位地面候选


def _normal_candidates(
    points,
    raw_count,
    sample_stride,
    max_normal_samples,
    normal_neighbors,
    min_normal_z,
    max_curvature,
    min_raw_count,
    min_seed_points,
    random_seed,
):
    n = len(points)
    # 小点云不能因默认步长造成“只剩几颗种子”；抽样只是计算加速。
    stride = min(sample_stride, max(1, n // (min_seed_points * 8)))
    ids = np.arange(0, n, stride, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    if len(ids) > max_normal_samples:
        ids = np.sort(rng.choice(ids, max_normal_samples, replace=False))
    ids = ids[raw_count[ids] >= min_raw_count]
    if len(ids) < min_seed_points:
        raise RuntimeError("满足原始点数阈值的法向样本不足，请检查 raw_count 或降低 MIN_RAW_COUNT")
    tree = cKDTree(points)
    good, normals, curvature = [], [], []
    k = min(normal_neighbors, n)
    for start in range(0, len(ids), 16_384):
        block = ids[start : start + 16_384]
        _, neighbors = tree.query(points[block], k=k, workers=1)
        samples = points[neighbors]
        centered = samples - np.mean(samples, axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / k
        values, vectors = np.linalg.eigh(cov)
        values = np.maximum(values, 0.0)
        normal = vectors[:, :, 0]
        curve = values[:, 0] / np.maximum(values.sum(axis=1), 1e-18)
        # 第二特征值排除近共线/重复点产生的假平面。
        mask = (
            (np.abs(normal[:, 2]) >= min_normal_z)
            & (curve <= max_curvature)
            & (values[:, 1] > 1e-10)
        )
        good.append(block[mask])
        normals.append(normal[mask])
        curvature.append(curve[mask])
    return (
        np.concatenate(good),
        np.concatenate(normals),
        np.concatenate(curvature),
        stride,
        len(ids),
    )


def _low_candidates(points, bucket_size, quantile, tolerance):
    cells = np.floor(points[:, :2] / bucket_size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.lexsort((points[:, 2], inverse))
    groups = inverse[order]
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    positions = starts + np.floor((sizes - 1) * quantile).astype(int)
    threshold = points[order[positions], 2]
    return points[:, 2] <= threshold[inverse] + tolerance


def _fit_z_plane(points):
    # z = a*x + b*y + c；先居中可降低大地坐标造成的条件数问题。
    center = np.mean(points, axis=0)
    xy = points[:, :2] - center[:2]
    if np.linalg.matrix_rank(xy) < 2:
        raise ValueError("地面种子近共线，无法拟合平面")
    slope = np.linalg.lstsq(xy, points[:, 2] - center[2], rcond=None)[0]
    return np.array([slope[0], slope[1], center[2] - slope @ center[:2]])


def _plane_height(points, coefficients):
    return points[:, :2] @ coefficients[:2] + coefficients[2]


# 分层鲁棒拟合与刚体调平


def _estimate_plane(
    points,
    normals,
    histogram_bin_size,
    low_bucket_size,
    low_quantile,
    low_fit_quantile,
    plane_fit_tolerance,
    ransac_iterations,
    min_seed_points,
    min_seed_cells,
    min_seed_span,
    min_normal_z,
    plane_normal_angle,
    min_relative_plane_support,
    random_seed,
):
    low_mask = _low_candidates(points, low_bucket_size, low_quantile, plane_fit_tolerance)
    low, low_normals = points[low_mask], normals[low_mask]
    if len(low) < min_seed_points:
        raise RuntimeError("低位水平地面候选不足，不能可靠识别主地面")
    bins = np.floor(low[:, 2] / histogram_bin_size).astype(np.int64)
    labels, counts = np.unique(bins, return_counts=True)
    # 忽略孤立的低位回波；保留低 65% 高度，降低高密度桌面对主平面的影响。
    min_bin_count = max(3, min_seed_points // 6)
    significant = labels[counts >= min_bin_count]
    low_limit = np.quantile(low[:, 2], low_fit_quantile) + histogram_bin_size
    fit_mask = np.isin(bins, significant) & (low[:, 2] <= low_limit)
    fit, fit_normals = low[fit_mask], low_normals[fit_mask]
    if len(fit) < min_seed_points:
        raise RuntimeError("低位高度层没有足够连续支持，不能拟合地面")
    rng = np.random.default_rng(random_seed)
    # 限制假设评估开销；最终拟合、支持门控仍回到全部水平候选。
    if len(fit) > 30_000:
        selection = np.sort(rng.choice(len(fit), 30_000, replace=False))
        fit, fit_normals = fit[selection], fit_normals[selection]
    cells = np.floor(fit[:, :2] / low_bucket_size).astype(np.int64)
    _, group, group_counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    weight = 1.0 / group_counts[group]
    anchor = np.median(fit[:, :2], axis=0)
    hypotheses = []

    def consider(coefficients):
        normal_z = 1.0 / np.sqrt(1 + np.dot(coefficients[:2], coefficients[:2]))
        if normal_z < min_normal_z:
            return
        residual = fit[:, 2] - _plane_height(fit, coefficients)
        normal = np.array([-coefficients[0], -coefficients[1], 1.0]) * normal_z
        # 交叉穿过几个真实平面的“假平面”不能仅因位置接近就获得地面票数。
        agreement = np.abs(fit_normals @ normal) >= np.cos(np.deg2rad(plane_normal_angle))
        inlier = (np.abs(residual) <= plane_fit_tolerance) & agreement
        if int(inlier.sum()) < min_seed_points:
            return
        area = float(weight[inlier].sum())
        if np.unique(group[inlier]).size < min_seed_cells:
            return
        if np.any(np.ptp(fit[inlier, :2], axis=0) < min_seed_span):
            return
        below = float(weight[residual < -2 * plane_fit_tolerance].sum())
        # 有大量更低水平点的面通常是桌面；它不能作为主地面。
        if below > max(0.20 * float(weight.sum()), 0.50 * area):
            return
        height = float(anchor @ coefficients[:2] + coefficients[2])
        hypotheses.append((height, area, coefficients.copy()))

    # 每个有支持的低位高度层先给一个拟合假设；斜地面的窄条也可提供正确斜率。
    fit_bins = np.floor(fit[:, 2] / histogram_bin_size).astype(np.int64)
    for label in np.unique(fit_bins):
        layer = fit[fit_bins == label]
        if len(layer) >= min_seed_points:
            try:
                consider(_fit_z_plane(layer))
            except (ValueError, np.linalg.LinAlgError):
                pass
    for _ in range(ransac_iterations):
        sample = fit[rng.choice(len(fit), 3, replace=False)]
        try:
            consider(_fit_z_plane(sample))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not hypotheses:
        raise RuntimeError("主地面支持不足：请检查 Z 轴方向、地面覆盖及 MIN_SEED_* 参数")
    # 先排除支持面积极小的偶然平面，再在最低合理层选择最大面积假设。
    max_area = max(row[1] for row in hypotheses)
    hypotheses = [
        row
        for row in hypotheses
        if row[1] >= max(min_seed_cells * 0.5, min_relative_plane_support * max_area)
    ]
    lowest_height = min(row[0] for row in hypotheses)
    viable = [row for row in hypotheses if row[0] <= lowest_height + 2 * plane_fit_tolerance]
    coefficients = max(viable, key=lambda row: row[1])[2]
    for _ in range(5):
        normal = np.array([-coefficients[0], -coefficients[1], 1.0])
        normal /= np.linalg.norm(normal)
        agreement = np.abs(normals @ normal) >= np.cos(np.deg2rad(plane_normal_angle))
        mask = (
            np.abs(points[:, 2] - _plane_height(points, coefficients)) <= plane_fit_tolerance
        ) & agreement
        if int(mask.sum()) < min_seed_points:
            raise RuntimeError("鲁棒拟合后地面种子数量不足")
        updated = _fit_z_plane(points[mask])
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    mask = (
        np.abs(points[:, 2] - _plane_height(points, coefficients)) <= plane_fit_tolerance
    ) & agreement
    seeds = points[mask]
    if 1.0 / np.sqrt(1.0 + np.dot(coefficients[:2], coefficients[:2])) < min_normal_z:
        raise RuntimeError("最终拟合平面超出允许地面倾角")
    seed_cells = np.unique(np.floor(seeds[:, :2] / low_bucket_size).astype(np.int64), axis=0)
    if (
        len(seeds) < min_seed_points
        or len(seed_cells) < min_seed_cells
        or np.any(np.ptp(seeds[:, :2], axis=0) < min_seed_span)
    ):
        raise RuntimeError("最终地面种子未通过数量、空间桶和跨度门控")
    return (
        coefficients,
        mask,
        {
            "histogram_bin_index": labels,
            "histogram_count": counts,
            "fit_sample_count": len(fit),
            "seed_cells": len(seed_cells),
            "candidate_hypotheses": len(hypotheses),
        },
    )


def _level_transform(coefficients):
    normal = np.array([-coefficients[0], -coefficients[1], 1.0])
    normal /= np.linalg.norm(normal)
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(normal, target)
    cosine = float(normal @ target)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    rotation = np.eye(3) + skew + (skew @ skew) / (1 + cosine)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    # 仅把平面的法向距离归零；不移动 XY 原点，不逐点改 Z。
    matrix[2, 3] = -coefficients[2] * normal[2]
    return validate_transform(matrix)


# 局部地面支持与分类


def _local_model(points, candidate_ids, plane, grid_size, min_seeds, max_deviation):
    candidates = points[candidate_ids]
    residual = candidates[:, 2] - _plane_height(candidates, plane)
    mask = np.abs(residual) <= max_deviation
    candidates, residual = candidates[mask], residual[mask]
    cells = np.floor(candidates[:, :2] / grid_size).astype(np.int64)
    unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    valid_groups = np.flatnonzero(counts >= min_seeds)
    if len(valid_groups) == 0:
        raise RuntimeError("没有通过局部支持门控的地面桶，请减小 SAMPLE_STRIDE 或检查地面密度")
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts[:-1])]
    offsets = np.array(
        [np.median(residual[order[starts[i] : starts[i] + counts[i]]]) for i in valid_groups]
    )
    # 用实际支持点 XY 均值，而非空桶中心：边界距离更符合真实证据位置。
    centers = np.column_stack(
        [
            np.bincount(inverse, weights=candidates[:, j], minlength=len(unique)) / counts
            for j in range(2)
        ]
    )[valid_groups]
    return unique[valid_groups], centers, offsets, counts[valid_groups]


@configured_defaults("Pipeline", "ExtractGround")
def extract_ground(
    input_pcd,
    output_dir,
    apply_leveling=None,
    ground_tolerance=None,
    min_obstacle_height=None,
    max_obstacle_height=None,
    min_raw_count=None,
    adapt_unit_raw_count=None,
    sample_stride=None,
    max_normal_samples=None,
    normal_neighbors=None,
    min_normal_z=None,
    max_curvature=None,
    histogram_bin_size=None,
    low_bucket_size=None,
    low_quantile=None,
    low_fit_quantile=None,
    plane_fit_tolerance=None,
    plane_normal_angle=None,
    min_relative_plane_support=None,
    ransac_iterations=None,
    min_seed_points=None,
    min_seed_cells=None,
    min_seed_span=None,
    local_grid_size=None,
    local_min_seeds=None,
    local_max_distance=None,
    local_max_plane_deviation=None,
    random_seed=None,
):
    """输出完整调平点云；函数所有可调参数均在签名和顶部配置区中列出。"""
    params = dict(locals())
    positive = (
        ground_tolerance,
        max_obstacle_height,
        min_raw_count,
        sample_stride,
        max_normal_samples,
        normal_neighbors,
        histogram_bin_size,
        low_bucket_size,
        plane_fit_tolerance,
        ransac_iterations,
        min_seed_points,
        min_seed_cells,
        min_seed_span,
        local_grid_size,
        local_min_seeds,
        local_max_distance,
        local_max_plane_deviation,
    )
    if not all(np.isfinite(value) and value > 0 for value in positive):
        raise ValueError("距离、数量、步长参数必须为正有限数")
    integer_names = (
        "min_raw_count",
        "sample_stride",
        "max_normal_samples",
        "normal_neighbors",
        "ransac_iterations",
        "min_seed_points",
        "min_seed_cells",
        "local_min_seeds",
        "random_seed",
    )
    if any(int(params[name]) != params[name] for name in integer_names):
        raise ValueError("数量、步长和随机种子必须为整数")
    if not (
        0 < min_normal_z <= 1
        and 0 < max_curvature < 1
        and 0 < low_quantile < 1
        and 0 < low_fit_quantile <= 1
        and 0 < min_relative_plane_support <= 1
        and 0 < plane_normal_angle < 90
    ):
        raise ValueError("法向、曲率和分位数参数范围错误")
    if (
        min_obstacle_height < 0
        or max_obstacle_height <= min_obstacle_height
        or normal_neighbors < 3
        or min_seed_points < 3
    ):
        raise ValueError("障碍高度范围或最小邻域/种子数量无效")
    cloud = read_cloud(input_pcd)
    if len(cloud) < max(normal_neighbors, min_seed_points) or not np.isfinite(cloud.points).all():
        raise ValueError("输入点数不足或含 NaN/Inf，请先执行预处理")
    # 混合不同采样密度的 PCD 时必须按来源单独适配，不能要求整个融合云计数全为 1。
    # gate_count 仅用于候选/分类门控，不写回 cloud.raw_count，也不伪造聚合计数。
    gate_count = cloud.raw_count.astype(np.float64)
    adapted_scan_ids, adapted_point_count, count_thresholds = [], 0, {}
    for scan_id in np.unique(cloud.scan_id):
        group = cloud.scan_id == scan_id
        adapt_group = bool(
            adapt_unit_raw_count and min_raw_count > 1 and np.all(cloud.raw_count[group] == 1)
        )
        count_thresholds[str(int(scan_id))] = 1 if adapt_group else min_raw_count
        if adapt_group:
            gate_count[group] = min_raw_count
            adapted_scan_ids.append(int(scan_id))
            adapted_point_count += int(group.sum())
    count_adapted = bool(adapted_scan_ids)
    effective_count = 1 if adapted_point_count == len(cloud) else min_raw_count
    candidate_ids, normals, curvature, actual_stride, sampled = _normal_candidates(
        cloud.points,
        gate_count,
        sample_stride,
        max_normal_samples,
        normal_neighbors,
        min_normal_z,
        max_curvature,
        min_raw_count,
        min_seed_points,
        random_seed,
    )
    if len(candidate_ids) < min_seed_points:
        raise RuntimeError("没有足够的水平低曲率候选，请检查 Z 轴和地面可见性")
    plane, seed_mask, fit_info = _estimate_plane(
        cloud.points[candidate_ids],
        normals,
        histogram_bin_size,
        low_bucket_size,
        low_quantile,
        low_fit_quantile,
        plane_fit_tolerance,
        ransac_iterations,
        min_seed_points,
        min_seed_cells,
        min_seed_span,
        min_normal_z,
        plane_normal_angle,
        min_relative_plane_support,
        random_seed,
    )
    matrix = _level_transform(plane) if apply_leveling else np.eye(4)
    result = transform_cloud(cloud, matrix)
    output_plane = np.zeros(3) if apply_leveling else plane
    cells, centers, offsets, local_counts = _local_model(
        result.points,
        candidate_ids,
        output_plane,
        local_grid_size,
        local_min_seeds,
        local_max_plane_deviation,
    )
    distances, nearest = cKDTree(centers).query(result.points[:, :2], k=1, workers=1)
    supported = distances <= local_max_distance
    ground_height = np.full(len(result), np.nan, dtype=np.float64)
    ground_height[supported] = (
        _plane_height(result.points[supported], output_plane) + offsets[nearest[supported]]
    )
    relative_height = result.points[:, 2] - ground_height
    supported_count = gate_count >= min_raw_count
    ground_mask = supported & (np.abs(relative_height) <= ground_tolerance) & supported_count
    obstacle_mask = (
        supported
        & (relative_height >= min_obstacle_height)
        & (relative_height <= max_obstacle_height)
        & supported_count
    )
    if int(ground_mask.sum()) < min_seed_points:
        raise RuntimeError("分类后有效地面数量不足，未写出点云")
    result.extras.update(
        ground_mask=ground_mask,
        obstacle_mask=obstacle_mask,
        ground_supported=supported,
        ground_height=ground_height,
    )

    # 保存点云、模型、矩阵和报告
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    output_pcd = out / "leveled.pcd"
    paths = {
        "output_pcd": str(output_pcd),
        "ground_pcd": str(out / "ground.pcd"),
        "non_ground_pcd": str(out / "non_ground.pcd"),
        "obstacle_pcd": str(out / "obstacle_band.pcd"),
        "ground_model": str(out / "ground_model.npz"),
        "transform": str(out / "T_input_to_level.npy"),
    }
    write_cloud(output_pcd, result)
    write_cloud(paths["ground_pcd"], result.subset(ground_mask))
    write_cloud(paths["non_ground_pcd"], result.subset(~ground_mask))
    write_cloud(paths["obstacle_pcd"], result.subset(obstacle_mask))
    np.save(paths["transform"], matrix, allow_pickle=False)
    np.savez_compressed(
        paths["ground_model"],
        plane_input_z_abc=plane,
        plane_output_z_abc=output_plane,
        T_input_to_level=matrix,
        seed_input_index=candidate_ids[seed_mask],
        seed_source_index=cloud.source_index[candidate_ids[seed_mask]],
        seed_scan_id=cloud.scan_id[candidate_ids[seed_mask]],
        seed_xyz_input=cloud.points[candidate_ids[seed_mask]],
        normal_candidate_input_index=candidate_ids,
        candidate_normals=normals,
        candidate_curvature=curvature,
        local_cells=cells,
        local_centers_xy=centers,
        local_offsets=offsets,
        local_seed_counts=local_counts,
        local_grid_size=local_grid_size,
        local_max_distance=local_max_distance,
        histogram_bin_index=fit_info["histogram_bin_index"],
        histogram_count=fit_info["histogram_count"],
    )
    report = {
        "module": "ExtractGround",
        "input": str(Path(input_pcd).expanduser().resolve()),
        "input_sha256": sha256(input_pcd),
        "parameters": params,
        **paths,
        "input_points": len(cloud),
        "output_points": len(result),
        "has_colors": result.colors is not None,
        "ground_points": int(ground_mask.sum()),
        "non_ground_points": int((~ground_mask).sum()),
        "obstacle_band_points": int(obstacle_mask.sum()),
        "unsupported_points": int((~supported).sum()),
        "normal_samples": sampled,
        "normal_candidates": len(candidate_ids),
        "actual_sample_stride": actual_stride,
        "global_seed_points": int(seed_mask.sum()),
        "global_seed_cells": fit_info["seed_cells"],
        "local_support_cells": len(cells),
        "plane_input_z_abc": plane,
        "T_input_to_level": matrix,
        "effective_min_raw_count": effective_count,
        "raw_count_adapted_for_unit_count": count_adapted,
        "effective_min_raw_count_by_scan": count_thresholds,
        "adapted_scan_ids": adapted_scan_ids,
        "adapted_point_count": adapted_point_count,
        "raw_count_adaptation_policy": "按 scan_id 检查：该组全部 raw_count=1 时适配阈值为1；其他组保持 min_raw_count；仅修改临时门控，不改真实计数",
        "raw_count_note": "部分或全部来源的计数全为1，按源适配阈值；未改写 raw_count 或伪造重复观测"
        if count_adapted
        else "使用配置的原始点数阈值",
        "limitations": [
            "仅对单层、Z 轴近竖直、具有足够可见地面的点云估计主地面；只有桌面且看不到地面时无法仅凭几何证明地面身份",
            "局部支持外的点保留在完整点云和 non_ground 中，不声称是障碍或自由空间",
            "只有一次全局刚体调平；局部地形起伏完整保留；apply_leveling=False 保持输入共同坐标",
        ],
        "sha256": {name: sha256(path) for name, path in paths.items()},
    }
    write_report(out / "ground_report.json", report)
    print(
        f"主地面种子：{int(seed_mask.sum())}；地面：{int(ground_mask.sum())}；缺少局部支持：{int((~supported).sum())}"
    )
    print(f"完整点云：{output_pcd}")
    return report


# 程序入口


def main(config=None):
    config = config or load_config("Pipeline")
    settings = config.section("ExtractGround")
    result = extract_ground(settings["input_pcd"], settings["output_dir"], **settings["parameters"])
    print(f"模块完成：{result['output_pcd']}")
    return result


if __name__ == "__main__":
    main(cli_config("Pipeline"))
