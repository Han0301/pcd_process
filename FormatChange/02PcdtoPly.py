"""将 PCD 导出为 PLY 预览，保留 Open3D 支持的坐标、颜色和法向。

完整来源属性仍以原 PCD 及其 .pcd.npz 旁文件为准。
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os

import open3d as o3d

from Common.Configuration import cli_config, load_config

# 读取点云


def main(config=None):
    config = config or load_config("Pipeline")
    settings = config.section("PcdtoPly")
    input_pcd = settings["input_pcd"]
    output_dir = settings["output_dir"]
    output_ply_name = settings["output_ply_name"]
    write_ascii = settings["write_ascii"]
    compressed = settings["compressed"]

    if not os.path.isfile(input_pcd):
        raise FileNotFoundError(f"输入文件不存在：{input_pcd}")

    if Path(output_dir).exists():
        raise FileExistsError(output_dir)

    pcd = o3d.io.read_point_cloud(str(input_pcd))

    if len(pcd.points) == 0:
        raise RuntimeError(f"点云为空或读取失败：{input_pcd}")

    print(f"输入文件：{input_pcd}")
    print(f"点云数量：{len(pcd.points)}")

    # 检查点云属性

    print("\n========== 点云属性 ==========")

    print(f"是否包含颜色：{pcd.has_colors()}")
    print(f"是否包含法向量：{pcd.has_normals()}")

    if pcd.has_colors():
        print("颜色格式：RGB")

    if pcd.has_normals():
        print("包含法向量数据")

    # 设置输出路径

    output_path = os.path.join(output_dir, output_ply_name)

    # 写入 PLY 文件

    os.makedirs(output_dir, exist_ok=False)

    success = o3d.io.write_point_cloud(
        output_path, pcd, write_ascii=write_ascii, compressed=compressed
    )

    if not success:
        raise RuntimeError(f"PLY 文件写入失败：{output_path}")

    # 验证输出文件

    if not os.path.isfile(output_path):
        raise RuntimeError("输出文件不存在")

    file_size_mb = os.path.getsize(output_path) / (1024**2)

    print("\n========== 转换完成 ==========")
    print(f"输入文件：{input_pcd}")
    print(f"输出文件：{output_path}")
    print(f"点云数量：{len(pcd.points)}")
    print(f"是否包含颜色：{pcd.has_colors()}")
    print(f"是否包含法向量：{pcd.has_normals()}")
    print(f"输出文件大小：{file_size_mb:.2f} MB")


if __name__ == "__main__":
    main(config=cli_config("Pipeline"))
