"""共同坐标点云融合：由真实分类端点产生覆盖域，保留未覆盖的旧扫描。

本模块不估计配准，也不从包围盒猜覆盖。最新源优先，输出点均来自输入行，
不平均坐标、不二次体素化；被替换旧点另存，便于追溯和恢复。
"""

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
from scipy.spatial import cKDTree

from Common.Configuration import cli_config, configured_defaults, load_config
from Common.OutputLayout import publish_cloud
from Common.PointCloudIO import (
    Cloud,
    concatenate_clouds,
    read_cloud,
    sha256,
    write_cloud,
    write_report,
)

# 来源与共同坐标校验


def _geometry_sha(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype="<f8").tobytes()).hexdigest()


def _registration_checks(paths, clouds, reports, already_aligned):
    if len(paths) <= 1:
        return []
    if already_aligned:
        return [
            {
                "input_id": i,
                "accepted": True,
                "authorization": "explicit_already_aligned",
                "note": "调用者明确确认共同坐标；本模块未重新验证绝对配准精度",
            }
            for i in range(1, len(paths))
        ]
    if reports is None:
        raise ValueError(
            "多源融合需要 accepted 配准报告；人工确认共同坐标可显式 already_aligned=True"
        )
    if isinstance(reports, dict):
        ordered = [None] + [reports.get(str(p), reports.get(p.name)) for p in paths[1:]]
    else:
        ordered = list(reports)
        if len(ordered) == len(paths) - 1:
            ordered = [None] + ordered
        if len(ordered) != len(paths):
            raise ValueError("registration_reports 长度须为输入数或输入数减1")
    accepted_geometry = {_geometry_sha(clouds[0].points)}
    checks = []
    for i in range(1, len(paths)):
        entry = ordered[i]
        if isinstance(entry, (str, Path)):
            report_path = Path(entry).expanduser().resolve()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_origin = str(report_path)
        elif isinstance(entry, dict):
            report, report_origin = entry, "provided_report_dict"
        else:
            raise ValueError(f"第{i}号源缺少配准报告")
        if report.get("accepted") is not True:
            raise ValueError(f"第{i}号源配准未通过，禁止融合")
        expected = report.get("aligned_geometry_sha256")
        actual = _geometry_sha(clouds[i].points)
        if not expected or expected != actual:
            raise ValueError(f"第{i}号源与通过验收的对齐XYZ/行序不一致")
        if report.get("target_geometry_sha256") not in accepted_geometry:
            raise ValueError(f"第{i}号源配准目标不在此前已确认的共同坐标源中")
        accepted_geometry.add(actual)
        checks.append(
            {
                "input_id": i,
                "accepted": True,
                "report": report_origin,
                "aligned_geometry_sha256": actual,
                "T_source_to_target": report.get("T_source_to_target"),
                "target_geometry_sha256": report.get("target_geometry_sha256"),
                "geometry_verified_after_optional_attribute_only_classification": True,
            }
        )
    return checks


def _harmonize(clouds, fill_missing_color):
    """保留 extras 并显式标识缺失字段；不猜颜色、不静默丢弃属性。"""
    colored = [cloud.colors is not None for cloud in clouds]
    filled_inputs = []
    if fill_missing_color is not None:
        color = np.asarray(fill_missing_color)
        if (
            color.shape != (3,)
            or not np.isfinite(color).all()
            or np.any(color < 0)
            or np.any(color > 255)
            or np.any(color != np.floor(color))
        ):
            raise ValueError("fill_missing_color 须为3个0~255整数")
        for i, cloud in enumerate(clouds):
            if cloud.colors is None:
                cloud.colors = np.tile(color.astype(np.uint8), (len(cloud), 1))
                filled_inputs.append(i)
    elif any(colored) and not all(colored):
        raise ValueError("混合彩色/无色源须显式指定 fill_missing_color；不猜颜色")
    # 上一轮来源层换到独立字段，报告保存映射；本轮仍使用固定的 fusion_input_*。
    reserved = {"fusion_input_id", "fusion_input_row"}
    storage_maps = [{} for _ in clouds]
    existing = set().union(*(cloud.extras for cloud in clouds))
    for key in reserved:
        storage = "prior__" + key
        while storage in existing:
            storage = "prior__" + storage
        for i, cloud in enumerate(clouds):
            if key in cloud.extras:
                cloud.extras[storage] = cloud.extras.pop(key)
                storage_maps[i][key] = storage
    keys = set().union(*(cloud.extras for cloud in clouds))
    for key in keys:
        values = [cloud.extras[key] for cloud in clouds if key in cloud.extras]
        shape = values[0].shape[1:]
        if any(value.shape[1:] != shape for value in values):
            raise ValueError(f"同名属性形状不同：{key}")
        dtype = np.result_type(*(value.dtype for value in values))
        if any(not np.can_cast(value.dtype, dtype, casting="safe") for value in values):
            raise ValueError(f"同名属性无法无损统一dtype：{key}")
        if dtype.kind in "fc":
            for value in values:
                if value.dtype.kind in "iu":
                    with np.errstate(invalid="ignore", over="ignore"):
                        restored = value.astype(dtype).astype(value.dtype)
                    if not np.array_equal(restored, value):
                        raise ValueError(f"属性 {key} 的整数转浮点会丢失精度，拒绝融合")
        missing = len(values) != len(clouds)
        presence_name = "attribute_present__" + key
        for cloud in clouds:
            present = key in cloud.extras
            if present:
                cloud.extras[key] = cloud.extras[key].astype(dtype, copy=False)
            else:
                fill = np.nan if dtype.kind in "fc" else ("" if dtype.kind in "US" else 0)
                cloud.extras[key] = np.full((len(cloud),) + shape, fill, dtype=dtype)
            if missing and presence_name not in keys and not key.startswith("attribute_present__"):
                cloud.extras[presence_name] = np.full(len(cloud), present, dtype=bool)
    for i, cloud in enumerate(clouds):
        cloud.extras["fusion_input_id"] = np.full(len(cloud), i, dtype=np.int32)
        cloud.extras["fusion_input_row"] = np.arange(len(cloud), dtype=np.int64)
    return filled_inputs, storage_maps


# 实际端点覆盖与逐点保留


def _endpoints(cloud):
    for name in ("ground_mask", "obstacle_mask"):
        if name not in cloud.extras or cloud.extras[name].shape != (len(cloud),):
            raise ValueError(f"覆盖源缺少逐点 {name}；先完成地面/障碍分类，禁止包围盒回退")
        if cloud.extras[name].dtype.kind != "b":
            raise ValueError(f"{name} 必须为布尔数组，不能静默转换标签")
    return cloud.extras["ground_mask"] | cloud.extras["obstacle_mask"]


def _remove_by_new(old_points, endpoint_points, resolution, halo, dedup):
    if not len(old_points) or not len(endpoint_points):
        empty = np.zeros(len(old_points), dtype=bool)
        return empty, empty.copy()
    # 全局固定0为落格基准；负坐标使用 floor，不依赖任何输入包围盒。
    endpoint_cells = np.unique(
        np.floor(endpoint_points[:, :2] / resolution).astype(np.int64), axis=0
    )
    old_cells = np.floor(old_points[:, :2] / resolution).astype(np.int64)
    cell_distance, _ = cKDTree(endpoint_cells).query(old_cells, workers=-1)
    covered = cell_distance * resolution <= halo + 1e-10
    near = np.zeros(len(old_points), dtype=bool)
    if dedup > 0:
        distances, _ = cKDTree(endpoint_points).query(old_points, workers=-1)
        near = distances <= dedup
    return covered, near


def _save_removed(path, cloud, rows, reasons):
    removed = cloud.subset(rows)
    payload = {
        "points": removed.points,
        "source_index": removed.source_index,
        "raw_count": removed.raw_count,
        "scan_id": removed.scan_id,
        "input_row": rows,
        "removal_reason_bits": reasons[rows],
    }
    if removed.colors is not None:
        payload["colors"] = removed.colors
    payload.update({"extra__" + key: value for key, value in removed.extras.items()})
    np.savez_compressed(path, **payload)


def _assert_origin(fused, clouds, keep_masks):
    for input_id, (cloud, keep) in enumerate(zip(clouds, keep_masks)):
        output_rows = np.flatnonzero(fused.extras["fusion_input_id"] == input_id)
        input_rows = fused.extras["fusion_input_row"][output_rows]
        if not np.array_equal(input_rows, np.flatnonzero(keep)):
            raise AssertionError("融合输出来源行号不一致")
        if not np.array_equal(fused.points[output_rows], cloud.points[input_rows]):
            raise AssertionError("融合不得创造或平均 XYZ")
        if fused.colors is not None and not np.array_equal(
            fused.colors[output_rows], cloud.colors[input_rows]
        ):
            raise AssertionError("融合不得重算 RGB")


# 融合主接口


@configured_defaults("Register", "FusePointClouds")
def fuse_clouds(
    input_pcds,
    output_dir,
    priority=None,
    resolution=None,
    coverage_halo=None,
    dedup_distance=None,
    registration_reports=None,
    already_aligned=None,
    fill_missing_color=None,
):
    """按输入时序进行覆盖融合；priority=last 为最后输入源优先。"""
    if priority not in ("first", "last"):
        raise ValueError("priority 只能为 first 或 last")
    if (
        not np.isfinite([resolution, coverage_halo, dedup_distance]).all()
        or resolution <= 0
        or min(coverage_halo, dedup_distance) < 0
    ):
        raise ValueError("resolution须大于0，其余距离须为非负有限数")
    paths = [Path(p).expanduser().resolve() for p in input_pcds]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("须提供至少一个输入，且不能重复同一文件")
    started = time.monotonic()
    clouds = [read_cloud(path) for path in paths]
    if any(not np.isfinite(cloud.points).all() for cloud in clouds):
        raise ValueError("输入含 NaN/Inf，请先完成预处理")
    input_hashes = [sha256(p) for p in paths]
    input_sidecar_hashes = [
        sha256(str(p) + ".npz") if Path(str(p) + ".npz").exists() else None for p in paths
    ]
    original_attributes = [
        {
            key: {"dtype": value.dtype.str, "shape_tail": list(value.shape[1:])}
            for key, value in cloud.extras.items()
        }
        for cloud in clouds
    ]
    original_has_colors = [cloud.colors is not None for cloud in clouds]
    checks = _registration_checks(paths, clouds, registration_reports, already_aligned)
    # 多源全部需有明确分类，即使某源本轮没有覆盖其他源也不能假定有效端点。
    endpoint_masks = (
        [_endpoints(cloud) for cloud in clouds]
        if len(clouds) > 1
        else [np.ones(len(clouds[0]), bool)]
    )
    filled_inputs, storage_maps = _harmonize(clouds, fill_missing_color)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    keep = [np.ones(len(cloud), dtype=bool) for cloud in clouds]
    reasons = [np.zeros(len(cloud), dtype=np.uint8) for cloud in clouds]
    order = list(range(len(clouds))) if priority == "last" else list(reversed(range(len(clouds))))
    stages = []
    previous = []
    for new_id in order:
        endpoint_points = clouds[new_id].points[endpoint_masks[new_id]]
        stage = {
            "priority_input_id": new_id,
            "valid_endpoint_count": len(endpoint_points),
            "replaced_sources": [],
        }
        for old_id in previous:
            active_rows = np.flatnonzero(keep[old_id])
            covered, near = _remove_by_new(
                clouds[old_id].points[active_rows],
                endpoint_points,
                resolution,
                coverage_halo,
                dedup_distance,
            )
            removed = covered | near
            bits = covered.astype(np.uint8) | (near.astype(np.uint8) << 1)
            reasons[old_id][active_rows] = bits
            keep[old_id][active_rows[removed]] = False
            if not keep[old_id][active_rows[~removed]].all():
                raise AssertionError("覆盖与近点范围外的旧点必须全部保留")
            stage["replaced_sources"].append(
                {
                    "input_id": old_id,
                    "coverage_removed": int(covered.sum()),
                    "near_removed": int(near.sum()),
                    "removed_union": int(removed.sum()),
                    "outside_both_retained": int((~removed).sum()),
                }
            )
        previous.append(new_id)
        stages.append(stage)
    fused = concatenate_clouds([cloud.subset(mask) for cloud, mask in zip(clouds, keep)])
    _assert_origin(fused, clouds, keep)
    output_pcd = out / "fused.pcd"
    write_cloud(output_pcd, fused)
    exported = read_cloud(output_pcd)
    _assert_origin(exported, clouds, keep)
    # 从输出+各输入被删行即可重建融合前每源；不重复保存整个参考云。
    archives = []
    for i, cloud in enumerate(clouds):
        removed_rows = np.flatnonzero(~keep[i])
        archive = out / f"removed_input_{i:03d}.npz"
        _save_removed(archive, cloud, removed_rows, reasons[i])
        original_order = np.sort(np.r_[np.flatnonzero(keep[i]), removed_rows])
        if not np.array_equal(original_order, np.arange(len(cloud))):
            raise AssertionError("保留与删除索引未能完整恢复输入行号")
        archives.append(
            {
                "input_id": i,
                "file": str(archive),
                "sha256": sha256(archive),
                "removed_points": len(removed_rows),
            }
        )
    unchanged = all(sha256(path) == digest for path, digest in zip(paths, input_hashes))
    sidecars_unchanged = all(
        (sha256(str(path) + ".npz") if Path(str(path) + ".npz").exists() else None) == digest
        for path, digest in zip(paths, input_sidecar_hashes)
    )
    if not unchanged or not sidecars_unchanged:
        raise RuntimeError("运行中输入点云或旁文件发生变化，输出不可接受")
    report = {
        "accepted": True,
        "output_pcd": str(output_pcd),
        "output_sha256": sha256(output_pcd),
        "output_sidecar_sha256": sha256(str(output_pcd) + ".npz"),
        "output_point_count": len(fused),
        "output_geometry_sha256": _geometry_sha(fused.points),
        "priority": priority,
        "resolution_m": resolution,
        "coverage_halo_m": coverage_halo,
        "dedup_distance_m": dedup_distance,
        "explicit_missing_color_fill": None
        if fill_missing_color is None
        else list(fill_missing_color),
        "inputs_with_filled_color": filled_inputs,
        "coverage_rule": "实际 ground|obstacle 端点落XY格，再按格中心距离圆形膨胀；不是bbox",
        "coverage_is_xy_not_visibility_raycast": True,
        "dedup_rule": "旧点到新源合格端点三维欧氏距离",
        "input_order": [
            {
                "input_id": i,
                "path": str(path),
                "sha256": input_hashes[i],
                "sidecar_sha256": input_sidecar_hashes[i],
                "points": len(clouds[i]),
                "kept_points": int(keep[i].sum()),
                "removed_points": int((~keep[i]).sum()),
                "original_had_colors": original_has_colors[i],
                "original_extra_attributes": original_attributes[i],
                "extra_storage_names": storage_maps[i],
                "geometry_sha256": _geometry_sha(clouds[i].points),
            }
            for i, path in enumerate(paths)
        ],
        "registration_checks": checks,
        "stages": stages,
        "removed_archives": archives,
        "checks": {
            "xyzrgb_are_exact_input_rows": True,
            "no_postfusion_voxelization": True,
            "old_points_outside_coverage_and_dedup_preserved": True,
            "inputs_and_sidecars_unchanged": True,
            "retained_plus_removed_reconstruct_each_input": True,
        },
        "recovery": "fused.pcd.npz 中 fusion_input_id/row 定位保留行；removed_input_XXX.npz 保存删除行；按input_row还原",
        "absolute_accuracy_validated": False,
        "elapsed_seconds": time.monotonic() - started,
        "report_file": str(out / "fusion_report.json"),
    }
    write_report(out / "fusion_report.json", report)
    return report


# 从保留点与删除归档恢复融合前输入


def restore_input_from_fusion(fusion_report, input_id, output_pcd):
    """不需要原输入仍存在；用 fused+removed 恢复指定源的点序和原属性。"""
    report = (
        json.loads(Path(fusion_report).read_text(encoding="utf-8"))
        if not isinstance(fusion_report, dict)
        else fusion_report
    )
    if not report.get("accepted") or input_id not in range(len(report["input_order"])):
        raise ValueError("融合报告未通过或 input_id 越界")
    item = report["input_order"][input_id]
    fused_path = Path(report["output_pcd"])
    if (
        sha256(fused_path) != report["output_sha256"]
        or sha256(str(fused_path) + ".npz") != report["output_sidecar_sha256"]
    ):
        raise ValueError("融合点云或旁文件已改变")
    archive = report["removed_archives"][input_id]
    if sha256(archive["file"]) != archive["sha256"]:
        raise ValueError("删除点归档已改变")
    fused = read_cloud(fused_path)
    chosen = fused.extras["fusion_input_id"] == input_id
    kept = fused.subset(chosen)
    keep_rows = kept.extras["fusion_input_row"]
    with np.load(archive["file"], allow_pickle=False) as packed:
        removed_rows = packed["input_row"]
        rows = np.r_[keep_rows, removed_rows]
        if not np.array_equal(np.sort(rows), np.arange(item["points"])):
            raise ValueError("保留/删除来源索引不完整或重复")

        def restore(values, removed_values, dtype=None):
            values = np.asarray(values)
            result = np.empty((item["points"],) + values.shape[1:], dtype=dtype or values.dtype)
            result[keep_rows] = values
            result[removed_rows] = removed_values
            return result

        points = restore(kept.points, packed["points"])
        colors = restore(kept.colors, packed["colors"]) if item["original_had_colors"] else None
        names = item.get("extra_storage_names", {})
        extras = {
            key: restore(
                kept.extras[names.get(key, key)],
                packed["extra__" + names.get(key, key)],
                np.dtype(spec["dtype"]),
            )
            for key, spec in item["original_extra_attributes"].items()
        }
        cloud = Cloud(
            points,
            colors,
            restore(kept.source_index, packed["source_index"]),
            restore(kept.raw_count, packed["raw_count"]),
            restore(kept.scan_id, packed["scan_id"]),
            extras,
        )
    if _geometry_sha(cloud.points) != item["geometry_sha256"]:
        raise ValueError("恢复XYZ与融合前输入指纹不一致")
    write_cloud(output_pcd, cloud)
    return {
        "output_pcd": str(Path(output_pcd).resolve()),
        "input_id": input_id,
        "point_count": len(cloud),
        "geometry_exact": True,
    }


# 独立运行


def main(config=None):
    config = config or load_config("Register")
    settings = config.section("FusePointClouds")
    result = fuse_clouds(
        settings["input_pcds"],
        settings["output_dir"],
        registration_reports=settings["registration_reports"],
        **settings["parameters"],
    )
    result["published_pcd"] = publish_cloud(
        result["output_pcd"], config.layout.result_dir / "cloud/fused.pcd"
    )
    print(f"融合完成：{result['published_pcd']}")
    return result


if __name__ == "__main__":
    main(cli_config("Register"))
