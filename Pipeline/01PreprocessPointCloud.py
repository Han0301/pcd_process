"""
基础点云预处理：无效坐标 → 可选体素 → 可选统计离群滤波。

不自动用分位数收缩场地、不默认删除稀疏墙面或玻璃周边。
输出 cleaned.pcd；被剔除点按原因分别保存，来源字段跟随旁文件保存。
体素合并代表点不是错误点，单独列出 merged_voxel_representatives.pcd。
"""

import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parents[1])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

from Common.Configuration import cli_config, configured_defaults, load_config
from Common.PointCloudIO import (
    binary_pcd,
    concatenate_clouds,
    pcd_rows,
    read_cloud,
    sha256,
    voxel_reduce,
    write_cloud,
    write_report,
)

# 无效点审计


def _save_invalid(path, cloud):
    # 无效 XYZ 不能作为正常 PCD 写出，用 NPZ 保留原值和来源，便于逐行复查。
    payload = {
        "points": cloud.points,
        "source_index": cloud.source_index,
        "raw_count": cloud.raw_count,
        "scan_id": cloud.scan_id,
        "reason_code": np.full(len(cloud), 1, dtype=np.uint8),
        **{"extra__" + k: v for k, v in cloud.extras.items()},
    }
    if cloud.colors is not None:
        payload["colors"] = cloud.colors
    np.savez_compressed(path, **payload)


# 基础过滤


def _stream_voxels(source, output, voxel_size):
    """相同体素进入同一磁盘桶，复用 voxel_reduce，避免把完整原始云载入内存。"""
    records = binary_pcd(source)
    pieces, invalid_count = [], 0
    with tempfile.TemporaryDirectory(prefix=".pcd_buckets_", dir=output) as temporary:
        paths = [Path(temporary) / f"{i:02d}.bin" for i in range(64)]
        streams = [p.open("wb") for p in paths]
        try:
            for start in range(0, len(records), 500_000):
                rows = np.arange(start, min(start + 500_000, len(records)))
                cloud = pcd_rows(records, rows)
                finite = np.isfinite(cloud.points).all(axis=1)
                invalid_count += int((~finite).sum())
                cells = np.floor(cloud.points[finite] / voxel_size)
                if np.any(np.abs(cells) > np.iinfo(np.int64).max / 100_000_000):
                    raise ValueError("坐标/体素比例超出分块哈希安全范围")
                cells = cells.astype(np.int64)
                buckets = (
                    (cells[:, 0] * 73856093) ^ (cells[:, 1] * 19349663) ^ (cells[:, 2] * 83492791)
                ) % 64
                valid_rows = rows[finite]
                for i, stream in enumerate(streams):
                    valid_rows[buckets == i].tofile(stream)
        finally:
            for stream in streams:
                stream.close()
        if invalid_count:
            raise ValueError("分块输入含无效坐标；请先单独审计，拒绝静默跳过")
        for i, path in enumerate(paths):
            rows = np.fromfile(path, dtype=np.int64)
            if len(rows):
                pieces.append(voxel_reduce(pcd_rows(records, rows), voxel_size))
            if (i + 1) % 16 == 0:
                print(f"大 PCD 下采样：{i + 1}/64 桶", flush=True)
    return concatenate_clouds(pieces), len(records)


@configured_defaults("Pipeline", "PreprocessPointCloud")
def preprocess(
    input_pcd,
    output_dir,
    voxel_size=None,
    statistical=None,
    nb_neighbors=None,
    std_ratio=None,
    save_removed=None,
    streaming=None,
):
    """从米制 PCD 生成处理云与来源报告；体素代表点保留实际坐标。"""
    source = Path(input_pcd).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    target = output / "cleaned.pcd"
    if source == target:
        raise ValueError("拒绝覆盖原始文件")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output}")
    if not np.isfinite(voxel_size) or voxel_size < 0:
        raise ValueError("voxel_size 必须为非负有限数")
    if statistical and (nb_neighbors < 2 or std_ratio <= 0 or not np.isfinite(std_ratio)):
        raise ValueError("统计滤波需要 nb_neighbors>=2 且 std_ratio>0")
    if streaming:
        if voxel_size <= 0 or save_removed:
            raise ValueError(
                "分块处理需要 voxel_size>0 且 save_removed=False；原始 PCD 保留所有合并点"
            )
        output.mkdir(parents=True, exist_ok=True)
        cloud, input_count = _stream_voxels(source, output, voxel_size)
        voxel_merged_count = input_count - len(cloud)
    else:
        cloud = read_cloud(source)
        input_count = len(cloud)
        voxel_merged_count = 0
    output.mkdir(parents=True, exist_ok=True)
    removed_counts, removed_files = {}, {}
    finite = np.isfinite(cloud.points).all(axis=1)
    invalid = cloud.subset(~finite)
    removed_counts["invalid_xyz"] = len(invalid)
    if save_removed and len(invalid):
        invalid_file = output / "removed_invalid_xyz.npz"
        _save_invalid(invalid_file, invalid)
        removed_files["invalid_xyz"] = str(invalid_file)
    cloud = cloud.subset(finite)
    if len(cloud) == 0:
        raise RuntimeError("没有有效坐标；审计数据已保留")
    before_voxel = cloud
    if not streaming:
        cloud = voxel_reduce(cloud, voxel_size)
        voxel_merged_count = len(before_voxel) - len(cloud)
    if save_removed and voxel_merged_count:
        # 同一 scan_id 的 source_index 唯一。多扫描场景使用两列键避免误匹配。
        pairs = np.rec.fromarrays(
            [before_voxel.scan_id, before_voxel.source_index], names="scan,source"
        )
        kept_pairs = np.rec.fromarrays([cloud.scan_id, cloud.source_index], names="scan,source")
        merged = before_voxel.subset(~np.isin(pairs, kept_pairs))
        path = output / "merged_voxel_representatives.pcd"
        write_cloud(path, merged)
        removed_files["voxel_merged_representatives"] = str(path)
    removed_counts["statistical_outlier"] = 0
    effective_neighbors = None
    if statistical:
        if len(cloud) < 3:
            raise RuntimeError("点数少于 3，不能可靠执行统计滤波")
        effective_neighbors = min(int(nb_neighbors), len(cloud) - 1)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud.points))
        _, inliers = pcd.remove_statistical_outlier(
            nb_neighbors=effective_neighbors, std_ratio=float(std_ratio)
        )
        mask = np.zeros(len(cloud), dtype=bool)
        mask[np.asarray(inliers, dtype=np.int64)] = True
        removed = cloud.subset(~mask)
        removed_counts["statistical_outlier"] = len(removed)
        if save_removed and len(removed):
            path = output / "removed_statistical_outlier.pcd"
            write_cloud(path, removed)
            removed_files["statistical_outlier"] = str(path)
        cloud = cloud.subset(mask)
    if len(cloud) == 0:
        raise RuntimeError("预处理后点云为空，拒绝生成空地图输入")
    write_cloud(target, cloud)
    report = {
        "module": "PreprocessPointCloud",
        "input": str(source),
        "input_sha256": sha256(source),
        "output_pcd": str(target),
        "output_sha256": sha256(target),
        "input_points": input_count,
        "output_points": len(cloud),
        "represented_raw_points": int(cloud.raw_count.astype(np.uint64).sum()),
        "voxel_size_m": voxel_size,
        "voxel_merged_representatives": voxel_merged_count,
        "statistical_enabled": bool(statistical),
        "nb_neighbors_requested": int(nb_neighbors),
        "nb_neighbors_effective": effective_neighbors,
        "std_ratio": std_ratio,
        "removed_by_reason": removed_counts,
        "removed_files": removed_files,
        "save_removed": bool(save_removed),
        "streaming": bool(streaming),
        "notes": [
            "cropping is handled only by CropPointCloud",
            "voxel merged representatives remain represented by accumulated raw_count",
            "statistical filtering may remove real sparse structures; inspect removed points before accepting",
        ],
    }
    write_report(output / "preprocess_report.json", report)
    return report


# 主程序


def main(config=None):
    config = config or load_config("Pipeline")
    settings = config.section("PreprocessPointCloud")
    result = preprocess(settings["input_pcd"], settings["output_dir"], **settings["parameters"])
    print(f"模块完成：{result['output_pcd']}")
    return result


if __name__ == "__main__":
    main(cli_config("Pipeline"))
