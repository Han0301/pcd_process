"""按高度范围和栅格参数将投影云导出为 PGM、PNG 与 YAML。

读取 YAML 的 PcdtoGridMap 节；可选统计滤波只作用于本次地图的输入。
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
import os

import numpy as np
import open3d as o3d
from PIL import Image

from Common.Configuration import cli_config, load_config

# 地面高度估计


def estimate_ground_z(z, bin_size=None):
    """
    在较低的 Z 高度范围内，通过高度直方图估计地面高度。
    适用于存在较多地面点、Z 轴基本竖直的点云。
    """

    if bin_size is None:
        bin_size = load_config("Pipeline").parameters("PcdtoGridMap")["ground_histogram_bin_size"]
    p1, p25 = np.percentile(z, [1, 25])
    low = z[(z >= p1) & (z <= p25)]

    if len(low) < 100:
        low = z

    z_min = np.floor(low.min() / bin_size) * bin_size
    z_max = np.ceil(low.max() / bin_size) * bin_size + bin_size

    edges = np.arange(z_min, z_max + bin_size * 0.5, bin_size)

    if len(edges) < 2:
        return float(np.median(low))

    hist, edges = np.histogram(low, bins=edges)
    idx = int(np.argmax(hist))

    return float((edges[idx] + edges[idx + 1]) / 2)


# 读取点云


def main(input_pcd=None, output_dir=None, config=None):
    config = config or load_config("Pipeline")
    settings = config.section("PcdtoGridMap")
    parameters = settings["parameters"]
    source = str(settings["input_pcd"] if input_pcd is None else input_pcd)
    destination = str(settings["output_dir"] if output_dir is None else output_dir)
    if Path(destination).exists():
        raise FileExistsError(destination)

    pcd = o3d.io.read_point_cloud(source)
    points = np.asarray(pcd.points)

    if points.size == 0:
        raise RuntimeError(f"点云为空或读取失败：{source}")

    # 去除 NaN、Inf
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) == 0:
        raise RuntimeError("过滤无效点后，点云为空")

    print(f"有效点数（统计滤波前）：{len(points)}")

    # 统计离群点滤波

    if parameters["use_statistical_filter"]:
        # 构建仅包含有效坐标的点云
        clean_pcd = o3d.geometry.PointCloud()
        clean_pcd.points = o3d.utility.Vector3dVector(points)

        # 统计滤波：根据邻域平均距离识别离群点
        filtered_pcd, inlier_indices = clean_pcd.remove_statistical_outlier(
            nb_neighbors=parameters["statistical_nb_neighbors"],
            std_ratio=parameters["statistical_std_ratio"],
        )

        points = np.asarray(filtered_pcd.points)

        if len(points) == 0:
            raise RuntimeError("统计滤波后点云为空")

        removed_count = len(inlier_indices)
        outlier_count = len(clean_pcd.points) - removed_count

        print("统计滤波已启用")
        print(f"邻域点数量：{parameters['statistical_nb_neighbors']}")
        print(f"标准差倍数：{parameters['statistical_std_ratio']}")
        print(f"滤除离群点数：{outlier_count}")
        print(f"统计滤波后点数：{len(points)}")

    else:
        print("统计滤波未启用")

    # 地面高度估计

    print("Z 高度分位数：", np.percentile(points[:, 2], [1, 5, 25, 50, 95, 99]))

    ground_z = (
        estimate_ground_z(points[:, 2], bin_size=parameters["ground_histogram_bin_size"])
        if parameters["ground_z"] is None
        else float(parameters["ground_z"])
    )

    print(f"采用的地面高度：{ground_z:.3f} m")

    # 确定地图边界

    res = parameters["resolution"]

    x_min = math.floor(points[:, 0].min() / res) * res
    y_min = math.floor(points[:, 1].min() / res) * res

    x_max = math.ceil(points[:, 0].max() / res) * res
    y_max = math.ceil(points[:, 1].max() / res) * res

    nx = int(round((x_max - x_min) / res)) + 1
    ny = int(round((y_max - y_min) / res)) + 1

    print(f"地图尺寸：{nx} × {ny}")
    print(f"地图分辨率：{res} m")
    print(f"地图原点：({x_min:.3f}, {y_min:.3f})")

    # 三维点映射到二维栅格

    # grid[y, x]：
    # -1 = 未知
    #  0 = 空闲
    # 100 = 占据

    grid = np.full((ny, nx), -1, dtype=np.int8)

    ix = np.floor((points[:, 0] - x_min) / res).astype(np.int64)

    iy = np.floor((points[:, 1] - y_min) / res).astype(np.int64)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    points = points[valid]
    ix = ix[valid]
    iy = iy[valid]

    flat_idx = iy * nx + ix

    # 地面点：标记空闲

    z = points[:, 2]

    ground_mask = np.abs(z - ground_z) <= parameters["ground_tolerance"]

    ground_counts = np.bincount(flat_idx[ground_mask], minlength=nx * ny)

    grid_flat = grid.ravel()

    # 至少有一个地面点，就作为空闲区域
    grid_flat[ground_counts >= 1] = 0

    # 障碍物点：标记占据

    obstacle_mask = (z >= ground_z + parameters["min_obstacle_height"]) & (
        z <= ground_z + parameters["max_obstacle_height"]
    )

    obstacle_counts = np.bincount(flat_idx[obstacle_mask], minlength=nx * ny)

    # 占据优先级高于空闲
    grid_flat[obstacle_counts >= parameters["min_obstacle_points"]] = 100

    # 可选障碍物膨胀

    if parameters["inflation_radius"] > 0:
        from scipy.ndimage import binary_dilation

        radius_cells = int(math.ceil(parameters["inflation_radius"] / res))

        yy, xx = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]

        structure = xx**2 + yy**2 <= radius_cells**2

        occupied = grid == 100

        inflated = binary_dilation(occupied, structure=structure)

        # 膨胀范围内的未知区域也会变成占据
        grid[inflated] = 100

    # 可选：将全部未知区域视为空闲
    if parameters["unknown_as_free"]:
        grid[grid == -1] = 0

    # 输出 PGM 地图

    # ROS map_server 图像约定：
    # 黑色 = 占据
    # 白色 = 空闲
    # 灰色 = 未知

    image = np.full((ny, nx), 205, dtype=np.uint8)

    image[grid == 0] = 254
    image[grid == 100] = 0

    # NumPy 中 Y 轴向上，图像行号向下，需要翻转
    image = np.flipud(image)

    pgm_path = os.path.join(destination, f"{parameters['map_name']}.pgm")

    png_path = os.path.join(destination, f"{parameters['map_name']}.png")

    os.makedirs(destination, exist_ok=False)
    Image.fromarray(image).save(pgm_path)
    Image.fromarray(image).save(png_path)

    # 输出 YAML 配置
    yaml_path = os.path.join(destination, f"{parameters['map_name']}.yaml")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"image: {os.path.basename(pgm_path)}\n")
        f.write("mode: trinary\n")
        f.write(f"resolution: {res}\n")
        f.write(f"origin: [{x_min}, {y_min}, 0.0]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")

    # 输出统计信息

    total = grid.size

    print("\n地图生成完成")
    print(f"PGM 文件：{pgm_path}")
    print(f"PNG 文件：{png_path}")
    print(f"YAML 文件：{yaml_path}")
    print(f"空闲栅格：{np.count_nonzero(grid == 0)}")
    print(f"占据栅格：{np.count_nonzero(grid == 100)}")
    print(f"未知栅格：{np.count_nonzero(grid == -1)}")
    print(f"总栅格数：{total}")


if __name__ == "__main__":
    main(config=cli_config("Pipeline"))
