"""从有地面分类的完整点云派生二维投影输入，保留原坐标与来源记录。

local_ground：使用局部地面支持，可将 Z 转为相对地面高度。
global_height：保留共同坐标系中的真实高度，可变换坐标并裁剪投影范围；
不因缺少局部地面支持而丢弃墙体。两种模式均不修改输入参考云。
"""

import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parents[1])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from pathlib import Path

import numpy as np

from Common.Configuration import cli_config, configured_defaults, load_config
from Common.PointCloudIO import read_cloud, sha256, validate_transform, write_cloud, write_report

# 输入检查与相对高度导出


@configured_defaults("Pipeline", "PrepareGridInput")
def prepare_grid_input(
    input_pcd,
    output_dir,
    normalize_local_ground=None,
    selection_mode=None,
    coordinate_transform=None,
    bounds=None,
):
    """按选择模式派生投影云，保存原坐标、选点行号及坐标变换。"""
    cloud = read_cloud(input_pcd)
    required = {"ground_mask", "obstacle_mask", "ground_supported", "ground_height"}
    if not required <= cloud.extras.keys():
        raise ValueError("缺少地面证据旁文件，请先 ExtractGround，复制PCD时保留 .pcd.npz")
    if not len(cloud) or not np.isfinite(cloud.points).all():
        raise ValueError("投影输入须非空且坐标有限")
    for key in ("ground_mask", "obstacle_mask", "ground_supported"):
        values = cloud.extras[key]
        if values.shape != (len(cloud),) or values.dtype.kind != "b":
            raise ValueError(f"{key} 必须是长度 N 的布尔数组")
    if cloud.extras["ground_height"].shape != (len(cloud),):
        raise ValueError("ground_height 必须是长度 N 的高度数组")
    ground = cloud.extras["ground_mask"]
    obstacle = cloud.extras["obstacle_mask"]
    supported = cloud.extras["ground_supported"]
    height = np.asarray(cloud.extras["ground_height"], float)
    if np.any(supported & ~np.isfinite(height)):
        raise ValueError("有地面支持的位置缺少有限地面高度")
    matrix = validate_transform(np.eye(4) if coordinate_transform is None else coordinate_transform)
    if selection_mode not in ("local_ground", "global_height"):
        raise ValueError("selection_mode 必须为 local_ground 或 global_height")
    if selection_mode == "local_ground" and not np.array_equal(matrix, np.eye(4)):
        raise ValueError("局部地面投影使用当前坐标；共同坐标变换请选 global_height")
    if selection_mode == "global_height" and normalize_local_ground:
        raise ValueError("global_height 不得同时归一化局部地面高度")
    mapped = (
        cloud.points
        if selection_mode == "local_ground"
        else cloud.points @ matrix[:3, :3].T + matrix[:3, 3]
    )
    keep = (
        (supported & np.isfinite(height) & (ground | obstacle))
        if selection_mode == "local_ground"
        else np.ones(len(cloud), bool)
    )
    if bounds is not None:
        limit = np.asarray(bounds, float)
        if limit.shape != (2, 3) or not np.isfinite(limit).all() or not np.all(limit[0] < limit[1]):
            raise ValueError("bounds 必须为有限2×3范围，且下界小于上界")
        keep &= ((mapped >= limit[0]) & (mapped < limit[1])).all(axis=1)
    if not keep.any() or (selection_mode == "local_ground" and not (keep & ground).any()):
        raise RuntimeError("没有可导出的投影输入，请检查范围和地面支持")
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{out}")
    selected = cloud.subset(keep)
    original = cloud.points[keep]
    selected.points = mapped[keep].copy()
    selected.extras["original_xyz"] = original.copy()
    selected.extras["original_z"] = original[:, 2].copy()
    if selection_mode == "global_height":
        heights = np.full(len(selected), np.nan)
        valid = supported[keep] & np.isfinite(height[keep])
        bases = original[valid].copy()
        bases[:, 2] = height[keep][valid]
        heights[valid] = (bases @ matrix[:3, :3].T + matrix[:3, 3])[:, 2]
        selected.extras["ground_height"] = heights
    elif normalize_local_ground:
        selected.extras["original_ground_height"] = height[keep].copy()
        selected.points[:, 2] -= height[keep]
        selected.extras["ground_height"] = np.zeros(len(selected), dtype=float)
    out.mkdir(parents=True, exist_ok=False)
    write_cloud(out / "grid_input.pcd", selected)
    if (~keep).any():
        write_cloud(out / "not_projected.pcd", cloud.subset(~keep))
    np.savez_compressed(
        out / "export_selection.npz",
        input_rows=np.flatnonzero(keep),
        excluded_input_rows=np.flatnonzero(~keep),
        original_xyz=original,
        source_index=selected.source_index,
        scan_id=selected.scan_id,
        coordinate_transform=matrix,
    )
    report = {
        "input": str(Path(input_pcd).resolve()),
        "input_sha256": sha256(input_pcd),
        "output_pcd": str((out / "grid_input.pcd").resolve()),
        "input_points": len(cloud),
        "output_points": len(selected),
        "ground_points": int((ground & keep).sum()),
        "obstacle_points": int((obstacle & keep).sum()),
        "not_projected_points": int((~keep).sum()),
        "selection_mode": selection_mode,
        "coordinate_transform": matrix.tolist(),
        "bounds": bounds,
        "unsupported_points_retained": int((keep & ~supported).sum()),
        "z_is_relative_to_local_ground": bool(normalize_local_ground),
        "physical_reference_cloud_modified": False,
        "recommended_ground_z": 0.0 if normalize_local_ground else None,
        "unknown_as_free": False,
    }
    write_report(out / "report.json", report)
    return report


def main(config=None):
    config = config or load_config("Pipeline")
    settings = config.section("PrepareGridInput")
    result = prepare_grid_input(
        settings["input_pcd"], settings["output_dir"], **settings["parameters"]
    )
    print(f"模块完成：{result['output_pcd']}")
    return result


if __name__ == "__main__":
    main(cli_config("Pipeline"))
