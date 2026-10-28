"""将 LAS 转为米制 PCD，保留可用的 RGB、intensity 和 classification。

路径与保存格式读取 YAML 的 LastoPcd 节；本步骤不降采样或变换坐标。
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


import os

import laspy
import numpy as np

from Common.Configuration import cli_config, load_config

# 读取 LAS 点云


def main(config=None):
    config = config or load_config("Pipeline")
    settings = config.section("LastoPcd")
    input_las = settings["input_las"]
    output_pcd = settings["output_pcd"]
    pcd_data_format = settings["pcd_data_format"]
    output_pcd = Path(output_pcd)
    if output_pcd.exists() or Path(str(output_pcd) + ".npz").exists():
        raise FileExistsError(f"拒绝覆盖已有 PCD 或旁文件：{output_pcd}")
    if output_pcd.suffix.lower() != ".pcd":
        raise ValueError("output_pcd 必须以 .pcd 结尾")
    if not os.path.isfile(input_las):
        raise FileNotFoundError(f"输入文件不存在：{input_las}")

    las = laspy.read(input_las)

    # 获取经过 LAS scale/offset 还原的实际坐标
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float32)

    if len(points) == 0:
        raise RuntimeError("输入 LAS 文件中没有点")

    # 过滤无效坐标
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    if len(points) == 0:
        raise RuntimeError("过滤无效点后，点云为空")

    dimensions = {str(name).lower() for name in las.point_format.dimension_names}

    print(f"输入点数：{len(points)}")
    print(f"LAS 字段：{dimensions}")

    # 构建 PCD 字段

    fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
    ]

    pcd_fields = ["x", "y", "z"]
    pcd_sizes = [4, 4, 4]
    pcd_types = ["F", "F", "F"]

    if "intensity" in dimensions:
        fields.append(("intensity", "<f4"))
        pcd_fields.append("intensity")
        pcd_sizes.append(4)
        pcd_types.append("F")

    has_rgb = all(name in dimensions for name in ("red", "green", "blue"))

    if has_rgb:
        fields.append(("rgb", "<f4"))
        pcd_fields.append("rgb")
        pcd_sizes.append(4)
        pcd_types.append("F")

    if "classification" in dimensions:
        fields.append(("classification", "u1"))
        pcd_fields.append("classification")
        pcd_sizes.append(1)
        pcd_types.append("U")

    # 填充点云数据

    data = np.zeros(len(points), dtype=np.dtype(fields, align=False))

    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]

    # 强度
    if "intensity" in dimensions:
        intensity = np.asarray(las.intensity, dtype=np.float32)[valid]

        data["intensity"] = intensity

    # RGB 颜色
    if has_rgb:

        def to_rgb8(channel):
            channel = np.asarray(channel, dtype=np.uint32)[valid]

            # LAS 颜色通常为 16 位，转换到 8 位
            if channel.max() > 255:
                channel = np.rint(channel.astype(np.float64) * 255.0 / 65535.0).astype(np.uint32)

            return channel

        red = to_rgb8(las.red)
        green = to_rgb8(las.green)
        blue = to_rgb8(las.blue)

        # 打包为 0x00RRGGBB
        rgb_packed = ((red << 16) | (green << 8) | blue).astype("<u4")

        # 按 PCD/PCL 约定写入 RGB 位模式
        data["rgb"] = rgb_packed.view("<f4")

    # 分类信息
    if "classification" in dimensions:
        classification = np.asarray(las.classification, dtype=np.uint8)[valid]

        data["classification"] = classification

    # 写入 PCD 文件

    output_dir = os.path.dirname(os.path.abspath(output_pcd))

    os.makedirs(output_dir, exist_ok=True)

    if pcd_data_format not in ("binary", "ascii"):
        raise ValueError("pcd_data_format 只能是 binary 或 ascii")

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(pcd_fields)}\n"
        f"SIZE {' '.join(map(str, pcd_sizes))}\n"
        f"TYPE {' '.join(pcd_types)}\n"
        f"COUNT {' '.join(['1'] * len(pcd_fields))}\n"
        f"WIDTH {len(data)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(data)}\n"
        f"DATA {pcd_data_format}\n"
    )

    with open(output_pcd, "wb") as f:
        f.write(header.encode("ascii"))

        if pcd_data_format == "binary":
            data.tofile(f)
        else:
            # ASCII 格式适合检查数据，但文件通常较大
            matrix = np.column_stack([data[name] for name in pcd_fields])

            np.savetxt(f, matrix, fmt="%.9g")

    # 输出统计信息

    print("\nLAS 转 PCD 完成")
    print(f"输出文件：{output_pcd}")
    print(f"点云数量：{len(data)}")
    print(f"PCD 字段：{pcd_fields}")
    print(f"输出格式：{pcd_data_format}")
    print(f"X 范围：{points[:, 0].min():.3f} ~ {points[:, 0].max():.3f}")
    print(f"Y 范围：{points[:, 1].min():.3f} ~ {points[:, 1].max():.3f}")
    print(f"Z 范围：{points[:, 2].min():.3f} ~ {points[:, 2].max():.3f}")


if __name__ == "__main__":
    main(config=cli_config("Pipeline"))
