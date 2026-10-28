"""两份点云的刚体配准：全局 FPFH/RANSAC → 多尺度点面 ICP → 独立留出验收。

只估计 SE(3)，不缩放、不改形。accepted 表示本模块的一致性门槛通过，
不代表有独立真值的绝对精度。失败仍保存诊断，但不能作为融合授权。
"""

import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parents[1])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import hashlib
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# 参数配置
from Common.Configuration import cli_config, complete_parameters, load_config, validate_parameters
from Common.PointCloudIO import (
    read_cloud,
    sha256,
    transform_cloud,
    validate_transform,
    write_cloud,
    write_report,
)

# 采样与几何辅助


def geometry_sha256(points):
    """行序与 float64 XYZ 的指纹；仅加分类属性不会改变该指纹。"""
    array = np.ascontiguousarray(points, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sample(points, maximum, rng):
    if len(points) <= maximum:
        return points
    return points[np.sort(rng.choice(len(points), maximum, replace=False))]


def _fit_evidence(cloud, minimum_raw, mode="classified"):
    if mode not in ("classified", "all_finite"):
        raise ValueError("fit_mode 只能为 classified 或 all_finite")
    if mode == "all_finite":
        mask = np.isfinite(cloud.points).all(axis=1) & (cloud.raw_count >= minimum_raw)
        return (
            cloud.points[mask],
            {
                "rule": "全部有限几何训练点；仍独立执行完整几何留出验收",
                "input_points": len(cloud),
                "fit_evidence_points": int(mask.sum()),
                "full_source_output_preserved": True,
            },
            np.flatnonzero(mask),
        )
    required = ("ground_mask", "obstacle_mask", "ground_supported")
    available = all(key in cloud.extras for key in required)
    if not available:
        return (
            cloud.points,
            {
                "rule": "未提供完整地面证据，使用全部有限坐标；不推断分类",
                "input_points": len(cloud),
                "fit_evidence_points": len(cloud),
            },
            np.arange(len(cloud)),
        )
    for key in required:
        if cloud.extras[key].shape != (len(cloud),) or cloud.extras[key].dtype.kind != "b":
            raise ValueError(f"配准证据字段必须为逐点布尔数组：{key}")
    mask = (cloud.extras["ground_mask"] | cloud.extras["obstacle_mask"]) & cloud.extras[
        "ground_supported"
    ]
    mask &= cloud.raw_count >= minimum_raw
    return (
        cloud.points[mask],
        {
            "rule": "(ground_mask|obstacle_mask)&ground_supported&(raw_count>=min)",
            "minimum_raw_count": minimum_raw,
            "input_points": len(cloud),
            "fit_evidence_points": int(mask.sum()),
            "excluded_unknown_or_unsupported_points": int((~mask).sum()),
            "full_source_output_preserved": True,
        },
        np.flatnonzero(mask),
    )


def _pcd(points, voxel=None, normals=False, config=None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if voxel:
        cloud = cloud.voxel_down_sample(float(voxel))
    if normals:
        config = config or load_config("Register").parameters("RegisterPointClouds")
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(2.5 * float(voxel or 0.05), config["normal_radius"]),
                max_nn=int(config["normal_max_neighbors"]),
            )
        )
        cloud.normalize_normals()
    return cloud


def _split_source(points, block_size, seed):
    # 整个空间小块只进入训练或留出；留出点完全不参与粗配准和 ICP。
    cells = np.floor(points / block_size).astype(np.int64)
    hashed = (
        (cells[:, 0] * 73856093)
        ^ (cells[:, 1] * 19349663)
        ^ (cells[:, 2] * 83492791)
        ^ np.int64(seed)
    )
    held = np.mod(hashed, 5) == 0
    if held.sum() < 30 or (~held).sum() < 50:
        raise ValueError("独立空间留出点不足；增加点数或缩小 holdout_block")
    return ~held, held


def _delta(first, second):
    relative = np.linalg.inv(first) @ second
    return {
        "translation_m": float(np.linalg.norm(relative[:3, 3])),
        "rotation_deg": float(
            np.degrees(np.linalg.norm(Rotation.from_matrix(relative[:3, :3]).as_rotvec()))
        ),
    }


def _icp(source, target, initial, voxels, distances, config, iterations=None):
    iterations = int(config["icp_iterations"]) if iterations is None else iterations
    transform = initial.copy()
    stages = []
    for voxel, distance in zip(voxels, distances):
        src = _pcd(source, voxel, normals=True, config=config)
        dst = _pcd(target, voxel, normals=True, config=config)
        if min(len(src.points), len(dst.points)) < 20:
            raise ValueError("ICP 下采样后点数不足")
        result = o3d.pipelines.registration.registration_icp(
            src,
            dst,
            distance,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.HuberLoss(k=config["icp_huber_delta"])
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
        )
        transform = validate_transform(result.transformation)
        stages.append(
            {
                "voxel_m": voxel,
                "correspondence_distance_m": distance,
                "fitness_training": float(result.fitness),
                "inlier_rmse_training_m": float(result.inlier_rmse),
            }
        )
    return transform, stages


def _fpfh_candidate(source, target, voxel, iterations, max_features, rng, config):
    src = _pcd(source, voxel, normals=True, config=config)
    dst = _pcd(target, voxel, normals=True, config=config)
    # 粗层特征再限量，控制 FPFH 最近邻及 RANSAC 的耗时。
    for cloud in (src, dst):
        if len(cloud.points) > max_features:
            ids = np.sort(rng.choice(len(cloud.points), max_features, replace=False))
            if cloud is src:
                src = cloud.select_by_index(ids)
            else:
                dst = cloud.select_by_index(ids)
    if min(len(src.points), len(dst.points)) < 30:
        raise ValueError("全局粗配准特征点不足")

    def feature(cloud):
        return o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100)
        )

    distance = 1.5 * voxel
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src,
        dst,
        feature(src),
        feature(dst),
        True,
        distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(iterations, 0.999),
    )
    return validate_transform(result.transformation), {
        "method": "FPFH_RANSAC",
        "voxel_m": voxel,
        "source_features": len(src.points),
        "target_features": len(dst.points),
        "fitness_training": float(result.fitness),
        "inlier_rmse_training_m": float(result.inlier_rmse),
        "max_iterations": iterations,
    }


def _structure(points, low=0.10, high=2.0):
    return points[(points[:, 2] >= low) & (points[:, 2] <= high)]


def _layer_grid(points, resolution):
    xy = np.floor(points[:, :2] / resolution).astype(np.int64)
    low = xy.min(axis=0)
    xy -= low
    shape = (int(xy[:, 1].max()) + 1, int(xy[:, 0].max()) + 1)
    if max(shape) > 3000 or np.prod(shape) > 4_000_000:
        raise ValueError("yaw栅格范围过大；先检查单位/离群点，不能扩大包围盒自动配准")
    levels = np.searchsorted([0.60, 1.20], points[:, 2])
    grid = np.zeros((3,) + shape, dtype=np.float32)
    grid[levels, xy[:, 1], xy[:, 0]] = 1.0
    return grid, low


def _yaw_fft_candidates(source, target, resolution, step):
    src, dst = _structure(source), _structure(target)
    if min(len(src), len(dst)) < 100:
        return [], {"available": False, "reason": "地面归零后0.1~2m结构点不足"}
    dst_grid, dst_low = _layer_grid(dst, resolution)
    proposals = []
    # 全0~360度均匀搜索，不使用旧工程矩阵或历史yaw。
    for yaw in np.arange(0.0, 360.0, step):
        rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        rotated = src @ rotation.T
        src_grid, src_low = _layer_grid(rotated, resolution)
        correlation = sum(
            fftconvolve(dst_grid[k], src_grid[k, ::-1, ::-1], mode="full") for k in range(3)
        )
        denominator = max(float(src_grid.sum() + dst_grid.sum()), 1.0)
        suppress = max(1, int(np.ceil(0.9 / resolution)))
        for _ in range(8):
            row, col = np.unravel_index(np.argmax(correlation), correlation.shape)
            value = float(correlation[row, col])
            if value <= 0:
                break
            lag = np.array([col - src_grid.shape[2] + 1, row - src_grid.shape[1] + 1])
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:2, 3] = (dst_low - src_low + lag) * resolution
            proposals.append(
                {
                    "T": transform,
                    "method": "height_layer_yaw_XY_FFT",
                    "yaw_deg": float(yaw),
                    "fft_score": 2 * value / denominator,
                }
            )
            correlation[
                max(0, row - suppress) : row + suppress + 1,
                max(0, col - suppress) : col + suppress + 1,
            ] = -1
    proposals.sort(key=lambda value: value["fft_score"], reverse=True)
    return proposals, {
        "available": True,
        "search_yaw_deg": [0, 360],
        "yaw_step_deg": step,
        "resolution_m": resolution,
        "height_range_m": [0.1, 2.0],
        "height_split_m": [0.6, 1.2],
        "source_structure_points": len(src),
        "target_structure_points": len(dst),
        "note": "要求输入已调平且地面归零；三高度层二值端点互相关；没有自由空间或历史矩阵先验",
    }


def _symmetric_training_score(source, target, source_tree, target_tree, transform, gate=0.30):
    moved = source @ transform[:3, :3].T + transform[:3, 3]
    reverse = (target - transform[:3, 3]) @ transform[:3, :3]
    source_distance, _ = target_tree.query(moved, workers=2)
    target_distance, _ = source_tree.query(reverse, workers=2)
    overlap_source = float((source_distance <= gate).mean())
    overlap_target = float((target_distance <= gate).mean())

    # 双向各取最近50%，同时保留全域重叠惩罚，防止很小匹配区独占评分。
    def trimmed(values):
        count = max(1, len(values) // 2)
        return np.partition(values, count - 1)[:count]

    trimmed_distance = np.r_[trimmed(source_distance), trimmed(target_distance)]
    rmse = float(np.sqrt(np.mean(np.minimum(trimmed_distance, gate) ** 2)))
    overlap = float(np.sqrt(overlap_source * overlap_target))
    score = overlap / (1.0 + 2 * rmse / gate)
    return {
        "score": score,
        "symmetric_overlap": overlap,
        "source_overlap": overlap_source,
        "target_overlap": overlap_target,
        "symmetric_trim50_rmse_m": rmse,
        "distance_gate_m": gate,
    }


def _global(source, target, voxel, iterations, max_features, rng, config):
    fpfh_T, fpfh_report = _fpfh_candidate(
        source, target, voxel, iterations, max_features, rng, config
    )
    proposals = []
    # fitness=0不是有效假设，不能把identity继续交给ICP当全局成功。
    if fpfh_report["fitness_training"] > 0.02:
        proposals.append({"T": fpfh_T, "method": "FPFH_RANSAC"})
    fft_report = {"available": False, "reason": "调用者关闭yaw搜索"}
    if config["global_yaw_search"]:
        fft_candidates, fft_report = _yaw_fft_candidates(
            source, target, config["yaw_grid_resolution"], config["yaw_step_deg"]
        )
        proposals.extend(fft_candidates)
    if not proposals:
        raise ValueError("全局配准没有有效候选：FPFH无对应且结构yaw搜索不可用")
    # 只用训练结构点评分；独立留出点不用于候选选择或调整门槛。
    src, dst = _structure(source), _structure(target)
    if min(len(src), len(dst)) < 100:
        src, dst = source, target
    src = _sample(np.asarray(_pcd(src, 0.05).points), 15_000, rng)
    dst = _sample(np.asarray(_pcd(dst, 0.05).points), 15_000, rng)
    source_tree, target_tree = cKDTree(src), cKDTree(dst)
    for proposal in proposals:
        proposal["training_score"] = _symmetric_training_score(
            src, dst, source_tree, target_tree, proposal["T"], 0.45
        )
    proposals.sort(key=lambda value: value["training_score"]["score"], reverse=True)
    # 先保留各30度方向区间的最好候选，避免重复工位占满全局候选预算。
    directional = {}
    for proposal in proposals:
        yaw = Rotation.from_matrix(proposal["T"][:3, :3]).as_euler("xyz", degrees=True)[2] % 360
        directional.setdefault(int(yaw // 30), proposal)
    shortlist = list(directional.values()) + proposals
    distinct = []
    for proposal in shortlist:
        if any(
            _delta(proposal["T"], old["T"])["translation_m"] < 0.75
            and _delta(proposal["T"], old["T"])["rotation_deg"] < 5
            for old in distinct
        ):
            continue
        distinct.append(proposal)
        if len(distinct) >= config["global_icp_candidates"]:
            break
    refined = []
    for proposal in distinct:
        try:
            transform, stages = _icp(
                source,
                target,
                proposal["T"],
                tuple(config["icp_voxels"]),
                tuple(config["icp_distances"]),
                config,
            )
            score = _symmetric_training_score(src, dst, source_tree, target_tree, transform)
            refined.append(
                {
                    "T": transform,
                    "method": proposal["method"],
                    "initial_T": proposal["T"],
                    "training_score": score,
                    "icp_stages": stages,
                }
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
    if not refined:
        raise ValueError("全部全局候选的ICP均失败")
    refined.sort(key=lambda value: value["training_score"]["score"], reverse=True)
    best = refined[0]
    alternatives = [
        entry
        for entry in refined[1:]
        if _delta(best["T"], entry["T"])["translation_m"] > 0.30
        or _delta(best["T"], entry["T"])["rotation_deg"] > 1.0
    ]
    second = alternatives[0] if alternatives else None
    gap = (
        (1.0 - second["training_score"]["score"] / max(best["training_score"]["score"], 1e-12))
        if second
        else 1.0
    )
    return best["T"], {
        "method": "FPFH_and_levelled_yaw_FFT_multicandidate_ICP",
        "fpfh": fpfh_report,
        "yaw_fft": fft_report,
        "proposed_candidates": len(proposals),
        "refined_candidates": refined,
        "selected_method": best["method"],
        "selected_training_score": best["training_score"],
        "distinct_second_relative_score_gap": gap,
        "unambiguous": bool(gap >= config["min_global_score_gap"]),
        "ambiguity_requirement": config["min_global_score_gap"],
        "note": "候选与选择只用训练集；留出验收、面残差和稳定性阈值保持独立",
    }


# 独立留出残差、可观测性和扰动


def _validate_holdout(source, target, transform, gate, config):
    aligned = source @ transform[:3, :3].T + transform[:3, 3]
    tree = cKDTree(target)
    distances, indices = tree.query(aligned, workers=-1)
    matched = distances <= gate
    result = {
        "source_holdout_points": len(source),
        "target_holdout_points": len(target),
        "matched_points": int(matched.sum()),
        "source_overlap": float(matched.mean()),
        "nn_rmse_m": None,
        "nn_p95_m": None,
        "point_plane_rmse_m": None,
        "observable_rank": 0,
        "normal_information_condition": None,
        "normal_information_eigenvalues": [],
    }
    if matched.sum() < 30:
        return result
    distance = distances[matched]
    result.update(
        nn_rmse_m=float(np.sqrt(np.mean(distance**2))), nn_p95_m=float(np.quantile(distance, 0.95))
    )
    # 目标留出测量邻域单独估法向，不使用训练点的法向。
    query = aligned[matched]
    _, neighbor = tree.query(target[indices[matched]], k=min(16, len(target)), workers=-1)
    neighbors = target[neighbor]
    center = neighbors.mean(axis=1)
    centered = neighbors - center[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbors.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    curvature = eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-12)
    broad = (eigenvalues[:, 1] > 1e-8) & (curvature < 0.10)
    broad_residual = np.einsum("ij,ij->i", query - center, normals)
    result["broad_normal_point_plane_rmse_m"] = (
        float(np.sqrt(np.mean(broad_residual[broad] ** 2))) if broad.any() else None
    )
    result["broad_normal_pairs"] = int(broad.sum())
    valid = (eigenvalues[:, 1] > 1e-8) & (curvature < config["max_holdout_curvature"])
    result["normal_planarity_max_curvature"] = config["max_holdout_curvature"]
    result["valid_plane_pairs"] = int(valid.sum())
    if valid.sum() < 30:
        return result
    query, center, normals = query[valid], center[valid], normals[valid]
    residual = np.einsum("ij,ij->i", query - center, normals)
    result["point_plane_rmse_m"] = float(np.sqrt(np.mean(residual**2)))
    centered_query = query - query.mean(axis=0)
    scale = max(float(np.sqrt(np.mean(np.sum(centered_query**2, axis=1)))), 0.1)
    jacobian = np.column_stack([np.cross(centered_query / scale, normals), normals])
    eigen = np.linalg.eigvalsh(jacobian.T @ jacobian / len(jacobian))
    threshold = max(float(eigen[-1]) * 1e-5, 1e-9)
    result["observable_rank"] = int((eigen > threshold).sum())
    result["normal_information_eigenvalues"] = eigen.tolist()
    if eigen[0] > 0:
        result["normal_information_condition"] = float(eigen[-1] / eigen[0])
    return result


def _stability(source, target, transform, voxel, distance, shift, angle, config):
    outcomes = []
    for axis in range(6):
        perturb = np.eye(4)
        if axis < 3:
            perturb[axis, 3] = shift
        else:
            vector = np.zeros(3)
            vector[axis - 3] = np.radians(angle)
            perturb[:3, :3] = Rotation.from_rotvec(vector).as_matrix()
        # 围绕已配准源的中心转动，避免地图离原点较远时放大扰动。
        center = source.mean(axis=0) @ transform[:3, :3].T + transform[:3, 3]
        perturb[:3, 3] += center - perturb[:3, :3] @ center
        candidate, _ = _icp(
            source,
            target,
            perturb @ transform,
            (voxel,),
            (distance,),
            config,
            iterations=int(config["stability_iterations"]),
        )
        outcomes.append({"axis": axis, **_delta(transform, candidate)})
    return outcomes


# 配准主接口


def register_clouds(source_pcd, target_pcd, output_dir, initial_transform=None, **params):
    """返回 accepted/output_pcd/transform_file/report_file；失败 output_pcd=None。"""
    validate_parameters("RegisterPointClouds", params, partial=True)
    config = complete_parameters("Register", "RegisterPointClouds", params)
    if isinstance(initial_transform, (str, Path)):
        path = Path(initial_transform).expanduser().resolve()
        initial_transform = (
            np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else np.loadtxt(path)
        )
    voxels = tuple(float(x) for x in config["icp_voxels"])
    distances = tuple(float(x) for x in config["icp_distances"])
    if not voxels or len(voxels) != len(distances) or min(voxels + distances) <= 0:
        raise ValueError("ICP 尺度和距离须为等长的正数序列")
    if not 0 < config["min_overlap"] <= 1 or config["holdout_block"] <= 0:
        raise ValueError("min_overlap 需在 (0,1]，holdout_block 须大于0")
    started = time.monotonic()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_path, target_path = Path(source_pcd).resolve(), Path(target_pcd).resolve()
    source, target = read_cloud(source_path), read_cloud(target_path)
    input_hashes = {str(p): sha256(p) for p in (source_path, target_path)}
    sidecar_hashes = {
        str(p) + ".npz": sha256(str(p) + ".npz")
        for p in (source_path, target_path)
        if Path(str(p) + ".npz").is_file()
    }
    report = {
        "accepted": False,
        "source_pcd": str(source_path),
        "target_pcd": str(target_path),
        "input_sha256": input_hashes,
        "input_sidecar_sha256": sidecar_hashes,
        "source_geometry_sha256": geometry_sha256(source.points),
        "target_geometry_sha256": geometry_sha256(target.points),
        "parameters": config,
        "normal_estimation": {
            "radius_m": config["normal_radius"],
            "max_neighbors": config["normal_max_neighbors"],
        },
        "icp_loss": {"type": "Huber", "delta_m": config["icp_huber_delta"], "points_deleted": 0},
        "output_pcd": None,
        "transform_file": None,
        "report_file": str(out / "registration_report.json"),
        "absolute_accuracy_validated": False,
        "validation_scope": "内部几何一致性；不是外部控制点/机器人通行精度",
        "rejection_reasons": [],
    }
    transform = None
    try:
        if min(len(source), len(target)) < 100:
            raise ValueError("配准至少需要每源100点")
        if not np.isfinite(source.points).all() or not np.isfinite(target.points).all():
            raise ValueError("输入含 NaN/Inf，请先完成预处理")
        rng = np.random.default_rng(config["seed"])
        o3d.utility.random.seed(int(config["seed"]))
        fit_source, source_selection, _ = _fit_evidence(
            source, config["min_fit_raw_count"], config["fit_mode"]
        )
        fit_target, target_selection, target_input_rows = _fit_evidence(
            target, config["min_fit_raw_count"], config["fit_mode"]
        )
        report["fit_selection"] = {"source": source_selection, "target": target_selection}
        if min(len(fit_source), len(fit_target)) < 100:
            raise ValueError("可信地面/障碍配准证据不足100点，不回退未支持点")
        train_mask, held_mask = _split_source(fit_source, config["holdout_block"], config["seed"])
        # 目标每5点预留1点；与其余训练点无行号交集。
        # 使用原输入行号，保证分类留出与全源留出都不出现在目标训练集。
        target_held = target_input_rows % 5 == 0
        source_train = _sample(fit_source[train_mask], int(config["max_fit_points"]), rng)
        target_train = _sample(fit_target[~target_held], int(config["max_fit_points"]), rng)
        source_test = _sample(fit_source[held_mask], int(config["max_holdout_points"]), rng)
        target_test = _sample(
            fit_target[target_held], int(config["max_target_holdout_points"]), rng
        )
        report["holdout_policy"] = "源整空间块留出20%；目标独立行留出20%；全部不参与拟合"
        if initial_transform is None:
            transform, report["coarse"] = _global(
                source_train,
                target_train,
                config["coarse_voxel"],
                int(config["ransac_iterations"]),
                int(config["max_feature_points"]),
                rng,
                config,
            )
        else:
            transform = validate_transform(initial_transform).copy()
            report["coarse"] = {"method": "explicit_initial_transform"}
        report["initial_transform"] = transform.tolist()
        transform, report["icp_stages"] = _icp(
            source_train, target_train, transform, voxels, distances, config
        )
        metrics = _validate_holdout(source_test, target_test, transform, distances[-1], config)
        report["holdout_metrics"] = metrics
        # 分类证据筛选不能隐藏高墙/天花等完整源几何上的错位；保留原全源留出复核。
        _, full_source_held = _split_source(source.points, config["holdout_block"], config["seed"])
        full_target_held = np.arange(len(target)) % 5 == 0
        full_metrics = _validate_holdout(
            _sample(source.points[full_source_held], int(config["max_holdout_points"]), rng),
            _sample(target.points[full_target_held], int(config["max_target_holdout_points"]), rng),
            transform,
            distances[-1],
            config,
        )
        report["full_geometry_holdout_metrics"] = full_metrics
        gates = {
            "global_unambiguous": report["coarse"].get("unambiguous", True),
            "enough_holdout_pairs": metrics["matched_points"] >= 30,
            "overlap": metrics["source_overlap"] >= config["min_overlap"],
            "nn_rmse": metrics["nn_rmse_m"] is not None
            and metrics["nn_rmse_m"] <= config["max_nn_rmse"],
            "nn_p95": metrics["nn_p95_m"] is not None
            and metrics["nn_p95_m"] <= config["max_nn_p95"],
            "plane_rmse": metrics["point_plane_rmse_m"] is not None
            and metrics["point_plane_rmse_m"] <= config["max_plane_rmse"],
            "observable_rank": metrics["observable_rank"] >= config["min_observable_rank"],
            "information_condition": metrics["normal_information_condition"] is not None
            and metrics["normal_information_condition"] <= config["max_information_condition"],
            "full_geometry_overlap": full_metrics["source_overlap"] >= config["min_overlap"],
            "full_geometry_nn_rmse": full_metrics["nn_rmse_m"] is not None
            and full_metrics["nn_rmse_m"] <= config["max_nn_rmse"],
            "full_geometry_nn_p95": full_metrics["nn_p95_m"] is not None
            and full_metrics["nn_p95_m"] <= config["max_nn_p95"],
            "full_geometry_plane_rmse": full_metrics["point_plane_rmse_m"] is not None
            and full_metrics["point_plane_rmse_m"] <= config["max_plane_rmse"],
        }
        if all(gates.values()) and config["check_stability"]:
            stability = _stability(
                source_train,
                target_train,
                transform,
                voxels[-1],
                distances[-1],
                config["perturb_translation"],
                config["perturb_rotation_deg"],
                config,
            )
            report["perturbation_trials"] = stability
            gates["perturbation_stability"] = all(
                x["translation_m"] <= config["max_stability_translation"]
                and x["rotation_deg"] <= config["max_stability_rotation_deg"]
                for x in stability
            )
        elif not config["check_stability"]:
            # 关闭稳定性检查仅用于诊断，不能产生可融合的 accepted。
            gates["perturbation_stability"] = False
            report["stability_note"] = "稳定性验收被关闭，仅生成诊断，禁止融合"
        report["gates"] = gates
        report["rejection_reasons"] = [key for key, passed in gates.items() if not passed]
        report["accepted"] = bool(all(gates.values()))
        if report["accepted"]:
            aligned = transform_cloud(source, transform)
            destination = out / "aligned.pcd"
            write_cloud(destination, aligned)
            exported = read_cloud(destination)
            if not np.array_equal(exported.points, aligned.points):
                raise RuntimeError("写出回读XYZ不一致；须同时使用PointCloudIO精确坐标旁文件")
            if aligned.colors is not None and not np.array_equal(exported.colors, aligned.colors):
                raise RuntimeError("写出回读RGB不一致")
            report.update(
                output_pcd=str(destination),
                output_sha256=sha256(destination),
                output_sidecar_sha256=sha256(str(destination) + ".npz"),
                aligned_geometry_sha256=geometry_sha256(exported.points),
                aligned_point_count=len(aligned),
                full_source_preserved=True,
            )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        report["accepted"] = False
        report["output_pcd"] = None
        report["rejection_reasons"].append(str(error))
    if transform is not None:
        np.save(out / "T_source_to_target.npy", transform, allow_pickle=False)
        np.savetxt(out / "T_source_to_target.txt", transform, fmt="%.17g")
        report["transform_file"] = str(out / "T_source_to_target.npy")
        report["T_source_to_target"] = transform.tolist()
    report["inputs_unchanged"] = all(
        sha256(path) == digest for path, digest in {**input_hashes, **sidecar_hashes}.items()
    )
    if not report["inputs_unchanged"]:
        report["accepted"] = False
        report["rejection_reasons"].append("运行中输入文件发生变化")
    report["elapsed_seconds"] = time.monotonic() - started
    write_report(out / "registration_report.json", report)
    return report


# 独立运行


def main(config=None):
    config = config or load_config("Register")
    settings = config.section("RegisterPointClouds")
    result = register_clouds(
        settings["source_pcd"],
        settings["target_pcd"],
        settings["output_dir"],
        settings["initial_transform"],
        **settings["parameters"],
    )
    print("配准通过" if result["accepted"] else "配准拒绝，停止融合")
    print(result["report_file"])
    return result


if __name__ == "__main__":
    main(cli_config("Register"))
