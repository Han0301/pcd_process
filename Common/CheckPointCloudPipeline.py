"""独立检查两类 PCD 流程：文件来源、刚体矩阵、选点与融合回滚。"""

# 允许直接运行子目录脚本，也允许通过上层包导入。
import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parents[1])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from Common.Configuration import cli_config, load_config
from Common.PointCloudIO import (
    binary_pcd,
    pcd_rows,
    read_cloud,
    sha256,
    validate_transform,
    write_report,
)

# 基础断言和文件哈希


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _bits_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(
            np.ascontiguousarray(left).view(np.uint8), np.ascontiguousarray(right).view(np.uint8)
        )
    )


def _geometry_sha(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype="<f8").tobytes()).hexdigest()


def _file_hash(path, expected, label):
    _require(Path(path).is_file(), f"{label} 文件不存在：{path}")
    _require(sha256(path) == expected, f"{label} SHA256 不匹配：{path}")


def _hash_report(report, label):
    digest = report.get("input_sha256")
    if isinstance(digest, str):
        source = report.get("input", report.get("input_pcd"))
        _require(source is not None, f"{label} 缺少输入路径")
        _file_hash(source, digest, label + ".input")
    elif isinstance(digest, dict):
        for path, value in digest.items():
            _file_hash(path, value, label + ".input")
    sidecars = report.get("input_sidecar_sha256", {})
    if isinstance(sidecars, str):
        source = report.get("input", report.get("input_pcd"))
        sidecars = {source + ".npz": sidecars}
    for path, value in sidecars.items():
        _file_hash(path, value, label + ".sidecar")
    if report.get("output_sha256"):
        _file_hash(report["output_pcd"], report["output_sha256"], label + ".output")
    if report.get("output_sidecar_sha256"):
        _file_hash(
            report["output_pcd"] + ".npz",
            report["output_sidecar_sha256"],
            label + ".output_sidecar",
        )
    for key, value in report.get("sha256", {}).items():
        _file_hash(report[key], value, label + "." + key)
    if report.get("output_pcd"):
        # read_cloud 还会验证旁文件绑定的本体 SHA、长度，并恢复精确坐标。
        cloud = read_cloud(report["output_pcd"])
        _require(np.isfinite(cloud.points).all(), f"{label} 输出含 NaN/Inf")
        for key in ("output_points", "output_point_count", "aligned_point_count"):
            if key in report:
                _require(len(cloud) == report[key], f"{label} 输出点数与报告不一致")
        del cloud


def _identity_attributes(source, target, rows, label, ignore=(), raw_count=True):
    rows = np.asarray(rows, dtype=np.int64)
    _require(len(rows) == len(target), f"{label} 来源行数不一致")
    for name in ("source_index", "scan_id") + (("raw_count",) if raw_count else ()):
        _require(
            _bits_equal(getattr(source, name)[rows], getattr(target, name)),
            f"{label} {name} 被改写",
        )
    _require((source.colors is None) == (target.colors is None), f"{label} 颜色存在状态改变")
    if source.colors is not None:
        _require(_bits_equal(source.colors[rows], target.colors), f"{label} RGB 与来源不一致")
    for key, values in source.extras.items():
        if key not in ignore:
            _require(
                key in target.extras and _bits_equal(values[rows], target.extras[key]),
                f"{label} 属性 {key} 被改写",
            )


def _subset_rows(source, target, label):
    # 每次录制的 source_index 可能相同，必须连同 scan_id 匹配。
    dtype = np.dtype([("scan", "<i4"), ("source", "<i8")])
    a, b = np.empty(len(source), dtype=dtype), np.empty(len(target), dtype=dtype)
    a["scan"], a["source"] = source.scan_id, source.source_index
    b["scan"], b["source"] = target.scan_id, target.source_index
    order = np.argsort(a, kind="stable")
    _require(not np.any(a[order][1:] == a[order][:-1]), f"{label} 输入来源键重复，不能唯一追溯")
    positions = np.searchsorted(a[order], b)
    _require(np.all(positions < len(a)), f"{label} 来源行越界")
    rows = order[positions]
    _require(np.array_equal(a[rows], b), f"{label} 输出来源不属于输入")
    return rows


# 选点和刚体变换验证


def _check_subset(report, kind):
    target = read_cloud(report["output_pcd"])
    if kind == "crop" and report.get("skipped"):
        _require(report["mode"] == "none" and report["removed_points"] == 0, "跳过裁剪记录不合法")
        _require(
            Path(report["input"]).resolve() == Path(report["output_pcd"]).resolve(),
            "跳过裁剪未复用上游",
        )
        _require(
            len(target) == report["input_points"] == report["output_points"], "跳过裁剪点数不一致"
        )
        return
    if kind == "preprocess" and report.get("streaming"):
        records = binary_pcd(report["input"])
        source = pcd_rows(records, target.source_index)
        _require(len(records) == report["input_points"], "分块输入点数不一致")
        if not report["statistical_enabled"]:
            _require(
                int(target.raw_count.astype(np.uint64).sum()) == len(records),
                "分块体素原始计数未守恒",
            )
    else:
        source = read_cloud(report["input"])
    rows = _subset_rows(source, target, kind)
    _require(_bits_equal(source.points[rows], target.points), f"{kind} 选点改变 XYZ")
    _identity_attributes(
        source, target, rows, kind, raw_count=kind != "preprocess" or report["voxel_size_m"] == 0
    )
    folder = Path(report["output_pcd"]).parent
    if kind == "crop":
        with np.load(folder / "crop_selection.npz", allow_pickle=False) as data:
            _require(np.array_equal(rows, data["kept_input_rows"]), "裁剪来源行号不一致")
            partition = np.r_[data["kept_input_rows"], data["excluded_input_rows"]]
            _require(
                np.array_equal(np.sort(partition), np.arange(len(source))),
                "裁剪保留/删除未构成完整分割",
            )
        if report["mode"] == "none":
            _require(len(source) == len(target), "none 裁剪模式删除了点")
        elif report["mode"] in ("bounds", "quantile_xy"):
            bounds = np.asarray(report["bounds"])
            dims = bounds.shape[1]
            mask = (
                (source.points[:, :dims] >= bounds[0]) & (source.points[:, :dims] < bounds[1])
            ).all(axis=1)
            _require(np.array_equal(rows, np.flatnonzero(mask)), "裁剪与记录范围不一致")


def _check_transform(report, ground=False, transform_atol=None):
    if transform_atol is None:
        transform_atol = load_config("Pipeline").section("CheckPointCloudPipeline")[
            "transform_atol"
        ]
    source_path = report["input"] if ground else report["source_pcd"]
    source, target = read_cloud(source_path), read_cloud(report["output_pcd"])
    transform_file = report["transform"] if ground else report["transform_file"]
    matrix = validate_transform(np.load(transform_file, allow_pickle=False))
    expected = source.points @ matrix[:3, :3].T + matrix[:3, 3]
    _require(
        np.allclose(target.points, expected, rtol=0, atol=transform_atol),
        "刚体后 XYZ 与保存矩阵不一致",
    )
    ignore = (
        ("ground_mask", "obstacle_mask", "ground_supported", "ground_height")
        if ground
        else ("ground_height",)
    )
    _identity_attributes(
        source, target, np.arange(len(source)), "地面" if ground else "配准", ignore=ignore
    )
    if ground:
        params = report["parameters"]
        if params["apply_leveling"] is False:
            _require(
                np.array_equal(matrix, np.eye(4)) and _bits_equal(source.points, target.points),
                "关闭调平时共同坐标仍被修改",
            )
        h = target.extras["ground_height"]
        support = target.extras["ground_supported"].astype(bool)
        relative = target.points[:, 2] - h
        # 从实际来源计数独立复算每组门槛，不能将融合报告的单个统计值用于所有来源。
        # 原生PCD全1和已有聚合计数可能共存；适配不应改写真实raw_count。
        threshold = np.full(len(target), params["min_raw_count"], dtype=np.uint32)
        adapted_ids = []
        for scan_id in np.unique(target.scan_id):
            group = target.scan_id == scan_id
            if (
                params["adapt_unit_raw_count"]
                and params["min_raw_count"] > 1
                and np.all(target.raw_count[group] == 1)
            ):
                threshold[group] = 1
                adapted_ids.append(int(scan_id))
        _require(
            adapted_ids == report.get("adapted_scan_ids", adapted_ids),
            "按来源适配的原始计数记录不一致",
        )
        count = target.raw_count >= threshold
        mask = support & (np.abs(relative) <= params["ground_tolerance"]) & count
        obstacles = (
            support
            & (relative >= params["min_obstacle_height"])
            & (relative <= params["max_obstacle_height"])
            & count
        )
        _require(
            np.array_equal(mask, target.extras["ground_mask"]), "地面标签与高度/支持/计数不一致"
        )
        _require(
            np.array_equal(obstacles, target.extras["obstacle_mask"]),
            "障碍标签与高度/支持/计数不一致",
        )
        _require(not np.any(support & ~np.isfinite(h)), "地面支持位置高度无效")
    else:
        _require(
            report.get("accepted") is True and all(report["gates"].values()), "配准未通过全部门控"
        )
        _require(
            report["aligned_geometry_sha256"] == _geometry_sha(target.points),
            "配准输出几何指纹不一致",
        )
        if "ground_height" in source.extras:
            valid = np.isfinite(source.extras["ground_height"])
            if "ground_supported" in source.extras:
                valid &= source.extras["ground_supported"].astype(bool)
            bases = source.points[valid].copy()
            bases[:, 2] = source.extras["ground_height"][valid]
            z = (bases @ matrix[:3, :3].T + matrix[:3, 3])[:, 2]
            _require(
                np.allclose(target.extras["ground_height"][valid], z, rtol=0, atol=transform_atol),
                "配准后地面高度未同步变换",
            )


# 融合回滚与最终栅格输入


def _check_fusion(report):
    _require(report.get("accepted") is True, "融合未通过")
    fused = read_cloud(report["output_pcd"])
    _require(
        len(report["input_order"]) == len(report["removed_archives"]),
        "融合输入与删除归档数量不一致",
    )
    _require(
        np.all(np.isin(fused.extras["fusion_input_id"], np.arange(len(report["input_order"])))),
        "融合含未知输入来源",
    )
    _require(_geometry_sha(fused.points) == report["output_geometry_sha256"], "融合几何指纹不一致")
    rollback = []
    for item, archive in zip(report["input_order"], report["removed_archives"]):
        index = item["input_id"]
        _require(
            index == len(rollback) and archive["input_id"] == index, "融合来源/删除归档编号乱序"
        )
        _file_hash(item["path"], item["sha256"], "融合输入")
        if item.get("sidecar_sha256"):
            _file_hash(item["path"] + ".npz", item["sidecar_sha256"], "融合输入旁文件")
        _file_hash(archive["file"], archive["sha256"], "融合删除归档")
        original = read_cloud(item["path"])
        selected = np.flatnonzero(fused.extras["fusion_input_id"] == index)
        kept_rows = fused.extras["fusion_input_row"][selected]
        with np.load(archive["file"], allow_pickle=False) as removed:
            removed_rows = removed["input_row"]
            _require(
                np.array_equal(np.sort(np.r_[kept_rows, removed_rows]), np.arange(len(original))),
                "融合保留/删除行未构成完整无重复分割",
            )

            def rebuild(values, deleted, dtype):
                result = np.empty((len(original),) + values.shape[1:], dtype=dtype)
                result[kept_rows], result[removed_rows] = values, deleted
                return result

            for name in ("points", "source_index", "raw_count", "scan_id"):
                expected = getattr(original, name)
                reconstructed = rebuild(
                    getattr(fused, name)[selected], removed[name], expected.dtype
                )
                _require(
                    _bits_equal(reconstructed, expected), f"融合回滚 {index} 的 {name} 逐字节不一致"
                )
            if original.colors is not None:
                reconstructed = rebuild(
                    fused.colors[selected], removed["colors"], original.colors.dtype
                )
                _require(
                    _bits_equal(reconstructed, original.colors), f"融合回滚 {index} 的 RGB 不一致"
                )
            for key, expected in original.extras.items():
                storage = item.get("extra_storage_names", {}).get(key, key)
                reconstructed = rebuild(
                    fused.extras[storage][selected], removed["extra__" + storage], expected.dtype
                )
                _require(
                    _bits_equal(reconstructed, expected),
                    f"融合回滚 {index} 属性 {key} 逐字节不一致",
                )
        rollback.append(
            {
                "input_id": index,
                "points": len(original),
                "xyz_rgb_source_extras_bitwise_exact": True,
            }
        )
    return rollback


def _check_grid_input(report):
    source, target = read_cloud(report["input"]), read_cloud(report["output_pcd"])
    ground, obstacle = source.extras["ground_mask"], source.extras["obstacle_mask"]
    h = source.extras["ground_height"]
    mode = report.get("selection_mode", "local_ground")
    matrix = validate_transform(report.get("coordinate_transform", np.eye(4)))
    mapped = (
        source.points
        if mode == "local_ground"
        else source.points @ matrix[:3, :3].T + matrix[:3, 3]
    )
    keep = (
        (source.extras["ground_supported"] & np.isfinite(h) & (ground | obstacle))
        if mode == "local_ground"
        else np.ones(len(source), bool)
    )
    if report.get("bounds") is not None:
        limits = np.asarray(report["bounds"])
        keep &= ((mapped >= limits[0]) & (mapped < limits[1])).all(axis=1)
    rows = np.flatnonzero(keep)
    _identity_attributes(source, target, rows, "最终导出", ignore=("ground_height",))
    if mode == "global_height":
        _require(not report["z_is_relative_to_local_ground"], "全局高度模式混入局部归一化")
        _require(
            np.allclose(mapped[rows], target.points, atol=1e-12, rtol=0), "共同坐标刚体投影不一致"
        )
        expected = np.full(len(rows), np.nan)
        valid = source.extras["ground_supported"][rows] & np.isfinite(h[rows])
        base = source.points[rows[valid]].copy()
        base[:, 2] = h[rows[valid]]
        expected[valid] = (base @ matrix[:3, :3].T + matrix[:3, 3])[:, 2]
        _require(
            np.allclose(
                expected, target.extras["ground_height"], equal_nan=True, atol=1e-12, rtol=0
            ),
            "共同坐标地面高度不一致",
        )
    else:
        _require(np.array_equal(source.points[rows, :2], target.points[:, :2]), "栅格导出改变 XY")
        if report["z_is_relative_to_local_ground"]:
            _require(
                np.array_equal(source.points[rows, 2] - h[rows], target.points[:, 2]),
                "相对 Z 不等于真实 Z 减局部地面 Z",
            )
            _require(
                np.array_equal(target.extras["original_ground_height"], h[rows]),
                "地面高度归档不一致",
            )
            _require(np.all(target.extras["ground_height"] == 0), "相对高程 ground_height 不为0")
        else:
            _require(np.array_equal(source.points[rows], target.points), "不归一化时 XYZ 被修改")
            _require(
                _bits_equal(h[rows], target.extras["ground_height"]), "不归一化时地面高度被修改"
            )
    _require(
        np.array_equal(target.extras["original_z"], source.points[rows, 2]), "原始 Z 归档不一致"
    )
    if "original_xyz" in target.extras:
        _require(
            np.array_equal(target.extras["original_xyz"], source.points[rows]),
            "原始 XYZ 归档不一致",
        )
    _require(report.get("unknown_as_free") is False, "派生点云导出启用了 unknown_as_free")


def _check_temporal(report):
    for path, expected in report["input_hashes"].items():
        _file_hash(path, expected, "时序输入")
    _file_hash(report["evidence"], report["evidence_sha256"], "时序证据")
    source, output = read_cloud(report["input"]), read_cloud(report["output_pcd"])
    with np.load(report["evidence"], allow_pickle=False) as evidence:
        kept, removed = evidence["kept_rows"], evidence["removed_rows"]
        proposed = evidence["proposed_rows"]
        if "sparse_proposed_rows" in evidence:
            sparse = evidence["sparse_proposed_rows"]
            ray = evidence["ray_proposed_rows"]
            _require(
                np.array_equal(np.union1d(sparse, ray), proposed), "稀疏/射线候选与总候选不一致"
            )
            params = report["parameters"]
            _require(not len(sparse) or params["sparse_filter"], "未启用稀疏排除却发生删除")
            _require(
                np.all(source.raw_count[sparse] <= params["sparse_raw_count_max"]),
                "删除了高计数稀疏候选",
            )
            _require(
                np.all(evidence["hit_frames"][sparse] <= params["sparse_max_hit_frames"]),
                "删除了多次命中稀疏候选",
            )
            queries = evidence["sparse_query_rows"]
            if params["sparse_filter"]:
                select = (
                    (evidence["sparse_nearest_distance"] > params["sparse_nearest_distance"])
                    & (evidence["sparse_eighth_distance"] > params["sparse_eighth_distance"])
                    & (evidence["sparse_stable_distance"] > params["sparse_stable_distance"])
                )
                _require(np.array_equal(queries[select], sparse), "稀疏统计门槛与候选不一致")
            _require(
                np.all(evidence["hit_frames"][ray] <= params["ray_max_hit_frames"]),
                "射线候选超过命中上限",
            )

        _require(
            np.array_equal(np.sort(np.r_[kept, removed]), np.arange(len(source))),
            "时序保留/删除未完整分割输入",
        )
        _require(np.array_equal(source.points[kept], output.points), "时序筛选修改了坐标")
        _identity_attributes(source, output, kept, "时序来源")
        _require(
            not evidence["protected"][proposed].any() and not evidence["stable"][proposed].any(),
            "时序候选越过保护或多帧稳定域",
        )
        if report["mode"] == "review":
            _require(len(removed) == 0 and len(output) == len(source), "review 模式发生了删点")
        else:
            _require(np.array_equal(removed, proposed), "apply 删除与候选行号不一致")
        _require(
            np.array_equal(output.extras["temporal_remove_candidate"], np.isin(kept, proposed)),
            "输出时序候选标签不一致",
        )
        _require(
            not np.intersect1d(removed, np.flatnonzero(evidence["protected"])).size,
            "删除了受保护点",
        )
    _require(
        report["inputs_unchanged"] and not report["coordinates_modified"],
        "时序输入或坐标一致性检查未通过",
    )


def check_pipeline(report_dir, validation_name=None, transform_atol=None):
    if validation_name is None or transform_atol is None:
        settings = load_config("Pipeline").section("CheckPointCloudPipeline")
        validation_name = (
            settings["validation_name"] if validation_name is None else validation_name
        )
        transform_atol = settings["transform_atol"] if transform_atol is None else transform_atol
    folder = Path(report_dir).expanduser().resolve()
    started = time.monotonic()
    result = {
        "accepted": False,
        "pipeline_report": str(folder / "pipeline_report.json"),
        "checks": [],
        "absolute_accuracy_validated": False,
        "scope": "文件一致性、来源、矩阵、分类导出与融合回滚；不证明绝对精度、镜像身份或机器人可通行性",
    }

    def passed(label):
        result["checks"].append({"name": label, "passed": True})
        print(f"通过：{label}", flush=True)

    try:
        pipeline_path = folder / "pipeline_report.json"
        if not pipeline_path.is_file():
            pipeline_path = folder / "registration_pipeline_report.json"
        result["pipeline_report"] = str(pipeline_path)
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        _require(pipeline.get("status") == "complete", f"处理链未完成：{pipeline.get('status')}")
        _require(pipeline.get("inputs_unchanged") is True, "处理链源文件保持不变检查未通过")
        if pipeline.get("mode") == "single_point_cloud":
            _file_hash(pipeline["input"], pipeline["input_sha256"], "单份原始输入")
            if pipeline.get("input_sidecar_sha256"):
                _file_hash(
                    str(pipeline["input"]) + ".npz", pipeline["input_sidecar_sha256"], "输入旁文件"
                )
            previous = pipeline["input"]
            for key in ("preprocess", "ground", "crop"):
                stage = pipeline["stages"][key]
                _require(stage["input"] == previous, f"单份 {key} 输入断链")
                _hash_report(stage, key)
                previous = stage["output_pcd"]
            if "raw_temporal" in pipeline["stages"]:
                raw = pipeline["stages"]["raw_temporal"]
                _require(
                    raw.get("status") == "complete" and raw.get("input_data_unchanged"),
                    "原始帧准备未完成",
                )
                _require(raw.get("uses_historical_outputs") is False, "原始帧准备不可依赖历史输出")
                for path, digest in raw["input_hashes"].items():
                    _file_hash(path, digest, "原始帧来源")
                _file_hash(raw["frames_csv"], raw["frames_csv_sha256"], "新生成的时序帧 CSV")
                passed("METABAG 重算、原始索引与新生成时序输入")
            if pipeline.get("enable_temporal_filter"):
                temporal = pipeline["stages"]["temporal"]
                _require(temporal["input"] == previous, "时序输入断链")
                _hash_report(temporal, "temporal")
                _check_temporal(temporal)
                previous = temporal["output_pcd"]
                passed("时序来源、坐标、保护域、证据与保留删除分割")
            _check_subset(pipeline["stages"]["preprocess"], "preprocess")
            _check_transform(
                pipeline["stages"]["ground"], ground=True, transform_atol=transform_atol
            )
            _check_subset(pipeline["stages"]["crop"], "crop")
            _require(
                previous == pipeline.get("stage_output_pcd", pipeline["output_pcd"]),
                "完整处理云输出断链",
            )
            _file_hash(pipeline["output_pcd"], pipeline["output_sha256"], "单云固定出口")
            _file_hash(
                pipeline["output_pcd"] + ".npz",
                pipeline["output_sidecar_sha256"],
                "单云固定出口旁文件",
            )
            _require(sha256(previous) == pipeline["output_sha256"], "固定出口与末阶段PCD不一致")
            _require(
                sha256(previous + ".npz") == pipeline["output_sidecar_sha256"],
                "固定出口旁文件不一致",
            )
            matrix = np.load(pipeline["stages"]["ground"]["transform"], allow_pickle=False)
            _require(
                np.array_equal(matrix, np.load(pipeline["transform_file"], allow_pickle=False)),
                "单份完整矩阵不一致",
            )
            _require(
                not pipeline["registration_performed"] and not pipeline["fusion_performed"],
                "单份入口调用了配准/融合",
            )
            passed("单份源文件、三步选点/地面/矩阵与完整云")
            if "grid_input" in pipeline:
                _require(
                    pipeline["grid_input"]["input"] == pipeline["output_pcd"], "派生栅格输入断链"
                )
                _hash_report(pipeline["grid_input"], "grid_input")
                _check_grid_input(pipeline["grid_input"])
                passed("派生输入的局部相对高程")
            result["accepted"] = True
            return result
        if pipeline.get("mode") == "two_processed_clouds":
            _require(pipeline["preprocessing_performed"] is False, "配准入口重复执行了预处理")
            _hash_report(pipeline, "配准输入")
            _hash_report(pipeline["registration"], "registration")
            _check_transform(pipeline["registration"], transform_atol=transform_atol)
            passed("两份输入SHA、完整源点、刚体矩阵与配准验收")
            if "fusion" in pipeline:
                _hash_report(pipeline["fusion"], "fusion")
                result["fusion_rollback"] = _check_fusion(pipeline["fusion"])
                passed("融合来源与完整回滚")
            result["accepted"] = True
            return result
        raise ValueError("仅支持当前单份 PCD 和两份 PCD 配准报告；旧混合流程已移入备份")
    except Exception as error:
        result["error_type"], result["error"] = type(error).__name__, str(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_report(folder / validation_name, result)


def main(config=None):
    config = config or load_config("Pipeline")
    settings = config.section("CheckPointCloudPipeline")
    result = check_pipeline(
        settings["report_dir"], settings["validation_name"], settings["transform_atol"]
    )
    print(f"完整检查通过；报告：{settings['report_dir'] / settings['validation_name']}")
    return result


if __name__ == "__main__":
    main(cli_config("Pipeline"))
