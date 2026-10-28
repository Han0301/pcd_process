"""
显式范围裁剪，或从实际地面支持估计XY分位范围。
统计范围只限制建图范围，不代表范围外点已被识别为玻璃镜像。
"""

import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parents[1])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)
from pathlib import Path

import numpy as np

from Common.Configuration import cli_config, configured_defaults, load_config
from Common.OutputLayout import publish_cloud
from Common.PointCloudIO import read_cloud, sha256, write_cloud, write_report


# 计算有限范围并保留裁剪来源
@configured_defaults("Pipeline", "CropPointCloud")
def crop_cloud(
    input_pcd,
    output_dir,
    mode=None,
    bounds=None,
    xy_quantiles=None,
    xy_padding=None,
    resolution=None,
):
    """按配置选取输入点并记录来源；none 模式仅记录报告，复用输入云。"""
    cloud = read_cloud(input_pcd)
    if not len(cloud) or not np.isfinite(cloud.points).all():
        raise ValueError("输入必须非空且坐标有限")
    if mode not in ("none", "bounds", "quantile_xy"):
        raise ValueError("mode 必须为 none、bounds、quantile_xy")
    keep = np.ones(len(cloud), bool)
    actual_bounds = None
    if mode == "bounds":
        limit = np.asarray(bounds, float)
        if limit.shape != (2, 3) or not np.isfinite(limit).all() or not np.all(limit[0] < limit[1]):
            raise ValueError("BOUNDS必须为有限的2×3数组，且下限小于上限")
        keep = ((cloud.points >= limit[0]) & (cloud.points < limit[1])).all(1)
        actual_bounds = limit.tolist()
    if mode == "quantile_xy":
        if "ground_mask" not in cloud.extras:
            raise ValueError("分位裁剪需要实际地面支持，请先执行 ExtractGround")
        floor = cloud.points[cloud.extras["ground_mask"].astype(bool)]
        a, b = xy_quantiles
        if len(floor) < 30 or not (0 <= a < b <= 1) or xy_padding < 0 or resolution <= 0:
            raise ValueError("地面支持不足，或分位/留边/分辨率不合法")
        lo = np.floor((np.quantile(floor[:, :2], a, axis=0) - xy_padding) / resolution) * resolution
        hi = np.ceil((np.quantile(floor[:, :2], b, axis=0) + xy_padding) / resolution) * resolution
        keep = ((cloud.points[:, :2] >= lo) & (cloud.points[:, :2] < hi)).all(1)
        actual_bounds = [lo.tolist(), hi.tolist()]
    if not keep.any():
        raise RuntimeError("范围内没有点，请检查坐标系和裁剪参数")
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    output_pcd = (
        str(Path(input_pcd).resolve()) if mode == "none" else str((out / "cropped.pcd").resolve())
    )
    if mode != "none":
        write_cloud(output_pcd, cloud.subset(keep))
    if (~keep).any():
        write_cloud(out / "outside_roi.pcd", cloud.subset(~keep))
    if mode != "none":
        np.savez_compressed(
            out / "crop_selection.npz",
            kept_input_rows=np.flatnonzero(keep),
            excluded_input_rows=np.flatnonzero(~keep),
            source_index=cloud.source_index,
            scan_id=cloud.scan_id,
        )
    report = {
        "input": str(Path(input_pcd).resolve()),
        "input_sha256": sha256(input_pcd),
        "output_pcd": output_pcd,
        "mode": mode,
        "skipped": mode == "none",
        "bounds": actual_bounds,
        "input_points": len(cloud),
        "output_points": int(keep.sum()),
        "removed_points": int((~keep).sum()),
        "outside_points_semantically_identified_as_reflection": False,
    }
    write_report(out / "report.json", report)
    return report


def main(config=None):
    config = config or load_config("Pipeline")
    settings = config.section("CropPointCloud")
    result = crop_cloud(settings["input_pcd"], settings["output_dir"], **settings["parameters"])
    output = config.section("RunPointCloudPipeline")["output_dir"] / "processed.pcd"
    result["processed_pcd"] = publish_cloud(result["output_pcd"], output)
    print(f"模块完成：{output}")
    return result


if __name__ == "__main__":
    main(cli_config("Pipeline"))
