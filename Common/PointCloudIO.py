"""PCD 读写与来源维护。XYZ/RGB 写入 PCD，精确坐标及附加属性写入 .pcd.npz。"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

# 统一数据结构


@dataclass
class Cloud:
    points: np.ndarray
    colors: np.ndarray | None = None
    source_index: np.ndarray | None = None
    raw_count: np.ndarray | None = None
    scan_id: np.ndarray | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points 必须为 N×3")
        n = len(self.points)
        if self.colors is not None:
            colors = np.asarray(self.colors)
            if colors.shape != (n, 3):
                raise ValueError("colors 必须为 N×3")
            if not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 255):
                raise ValueError("Cloud.colors 必须为 0~255")
            self.colors = np.rint(colors).astype(np.uint8)
        self.source_index = self._attribute(
            self.source_index, np.int64, np.arange(n, dtype=np.int64), n
        )
        self.raw_count = self._attribute(self.raw_count, np.uint32, np.ones(n, dtype=np.uint32), n)
        self.scan_id = self._attribute(self.scan_id, np.int32, np.zeros(n, dtype=np.int32), n)
        if np.any(self.source_index < 0) or np.any(self.raw_count == 0):
            raise ValueError("source_index 必须非负，raw_count 必须大于 0")
        self.extras = {str(k): np.asarray(v) for k, v in self.extras.items()}
        for name, values in self.extras.items():
            if not name or values.ndim == 0 or len(values) != n or values.dtype.hasobject:
                raise ValueError(f"extras 字段必须有 N 行且不能含 Python 对象：{name}")

    @staticmethod
    def _attribute(values, dtype, default, n):
        if values is None:
            return default
        values = np.asarray(values)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError("来源属性必须为长度 N 的有限数组")
        limits = np.iinfo(dtype)
        if (
            np.any(values < limits.min)
            or np.any(values > limits.max)
            or np.any(values != np.floor(values))
        ):
            raise ValueError(f"来源属性超出 {np.dtype(dtype)} 的整数范围")
        return values.astype(dtype)

    def subset(self, ids):
        return Cloud(
            self.points[ids],
            None if self.colors is None else self.colors[ids],
            self.source_index[ids],
            self.raw_count[ids],
            self.scan_id[ids],
            {k: v[ids] for k, v in self.extras.items()},
        )

    def __len__(self):
        return len(self.points)


# 哈希、矩阵与报告


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(path):
    return Path(str(path) + ".npz")


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=encode, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_transform(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("刚体变换必须为有限的 4×4 矩阵")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("刚体变换末行必须为 0,0,0,1")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-6
    ):
        raise ValueError("旋转部分必须正交，且行列式为 +1；不接受缩放和镜像")
    return matrix


def transform_cloud(cloud, matrix):
    matrix = validate_transform(matrix)
    result = cloud.subset(np.arange(len(cloud)))
    result.points = cloud.points @ matrix[:3, :3].T + matrix[:3, 3]
    if "ground_height" in cloud.extras:
        heights = np.asarray(cloud.extras["ground_height"], dtype=np.float64)
        if heights.shape != (len(cloud),):
            raise ValueError("ground_height 必须为每点一个地面 Z 值")
        supported = np.isfinite(heights) & np.isfinite(cloud.points).all(axis=1)
        if "ground_supported" in cloud.extras:
            ground_supported = np.asarray(cloud.extras["ground_supported"])
            if ground_supported.shape != (len(cloud),):
                raise ValueError("ground_supported 必须为每点一个布尔值")
            supported &= ground_supported.astype(bool)
        # 地面高度是坐标系相关属性。变换对应基底点 (x,y,ground_z)，
        # 只改变云的 XYZ 而保留旧 ground_z 会让 Z 平移/俯仰后的高度错误。
        transformed_heights = np.full(len(cloud), np.nan, dtype=np.float64)
        bases = cloud.points[supported].copy()
        bases[:, 2] = heights[supported]
        transformed_heights[supported] = (bases @ matrix[:3, :3].T + matrix[:3, 3])[:, 2]
        result.extras["ground_height"] = transformed_heights
    return result


def concatenate_clouds(clouds):
    clouds = list(clouds)
    if not clouds:
        return Cloud(np.empty((0, 3)))
    keys = set(clouds[0].extras)
    if any(set(c.extras) != keys for c in clouds):
        raise ValueError("合并前 extras 字段必须相同，避免静默丢字段")
    has_colors = [c.colors is not None for c in clouds]
    if any(has_colors) and not all(has_colors):
        raise ValueError("彩色与无色点云不能静默混合，请先明确统一颜色")
    return Cloud(
        np.concatenate([c.points for c in clouds]),
        np.concatenate([c.colors for c in clouds]) if all(has_colors) else None,
        np.concatenate([c.source_index for c in clouds]),
        np.concatenate([c.raw_count for c in clouds]),
        np.concatenate([c.scan_id for c in clouds]),
        {k: np.concatenate([c.extras[k] for c in clouds]) for k in keys},
    )


def voxel_reduce(cloud, voxel_size):
    """每体素选 source_index 最小的原始点，保留实际 XYZ/RGB，raw_count 求和。"""
    if not np.isfinite(voxel_size) or voxel_size < 0:
        raise ValueError("voxel_size 必须为非负有限数")
    if voxel_size == 0 or len(cloud) == 0:
        return cloud.subset(np.arange(len(cloud)))
    if not np.isfinite(cloud.points).all():
        raise ValueError("体素处理前须先剔除非有限坐标")
    scaled = np.floor(cloud.points / voxel_size)
    if np.any(np.abs(scaled) > np.iinfo(np.int64).max / 2):
        raise ValueError("坐标/体素比例过大")
    cells = scaled.astype(np.int64)
    order = np.lexsort((cloud.source_index, cells[:, 2], cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    starts = np.r_[0, np.flatnonzero(np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)) + 1]
    selected = order[starts]
    counts = np.add.reduceat(cloud.raw_count[order].astype(np.uint64), starts)
    if np.any(counts > np.iinfo(np.uint32).max):
        raise OverflowError("单个体素的原始点计数超出 uint32")
    result = cloud.subset(selected)
    result.raw_count = counts.astype(np.uint32)
    return result


# PCD 读取与旁文件


def pcd_path(path):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".pcd":
        raise ValueError("仅支持 .pcd 点云；格式转换请使用 FormatChange 中的原脚本")
    return path


def binary_pcd(path):
    """映射原生 binary PCD；仅用于无旁文件的大文件分块处理。"""
    path = pcd_path(path)
    header = {}
    with path.open("rb") as stream:
        for _ in range(100):
            line = stream.readline().decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                offset = stream.tell()
                break
        else:
            raise ValueError("PCD 头不完整")
    if header["DATA"] != ["binary"] or sidecar_path(path).exists():
        raise ValueError("分块路径要求原生 DATA binary PCD 且无旁文件；其他 PCD 使用普通读取")
    sizes, types = header["SIZE"], header["TYPE"]
    fields = header["FIELDS"]
    counts = header.get("COUNT", ["1"] * len(fields))
    kinds = {"F": "f", "I": "i", "U": "u"}
    dtype = np.dtype(
        [
            (name, "<" + kinds[t] + size, (int(n),))
            if int(n) != 1
            else (name, "<" + kinds[t] + size)
            for name, size, t, n in zip(fields, sizes, types, counts)
        ],
        align=False,
    )
    count = int(header["POINTS"][0])
    if path.stat().st_size != offset + count * dtype.itemsize:
        raise ValueError("PCD 数据长度与头信息不一致")
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))


def pcd_rows(records, rows):
    """原生 PCD 的指定行转为统一结构；source_index 保留输入 PCD 行号。"""
    rows = np.asarray(rows, dtype=np.int64)
    data = records[rows]
    names = set(data.dtype.names)
    colors = None
    color_key = "rgb" if "rgb" in names else "rgba" if "rgba" in names else None
    if color_key:
        packed = data[color_key].copy()
        packed = packed.view("<u4") if packed.dtype.kind == "f" else packed.astype(np.uint32)
        colors = np.column_stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255]).astype(
            np.uint8
        )
    extras = {
        k: data[k].reshape(len(rows), -1).copy()
        for k in data.dtype.names
        if k not in ("x", "y", "z", "rgb", "rgba")
    }
    return Cloud(
        np.column_stack([data[k] for k in ("x", "y", "z")]),
        colors,
        source_index=rows,
        extras=extras,
    )


def read_cloud(path):
    path = pcd_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    pcd = o3d.t.io.read_point_cloud(
        str(path), remove_nan_points=False, remove_infinite_points=False
    )
    points = pcd.point.positions.numpy().copy() if "positions" in pcd.point else np.empty((0, 3))
    colors = pcd.point.colors.numpy().copy() if "colors" in pcd.point else None
    if colors is not None and np.issubdtype(colors.dtype, np.floating):
        colors = np.rint(colors * 255)
    extras = {
        ("normal" if k == "normals" else k): pcd.point[k].numpy().copy()
        for k in pcd.point
        if k not in ("positions", "colors")
    }
    cloud = Cloud(points, colors, extras=extras)
    sidecar = sidecar_path(path)
    if sidecar.exists():
        with np.load(sidecar, allow_pickle=False) as data:
            if str(data["cloud_sha256"].item()) != sha256(path):
                raise ValueError(f"旁文件 SHA256 不匹配，点云可能已被替换：{path}")
            exact_points = data["exact_points"] if "exact_points" in data else cloud.points
            exact_colors = data["exact_colors"] if "exact_colors" in data else cloud.colors
            if np.asarray(exact_points).shape != cloud.points.shape:
                raise ValueError(f"旁文件精确坐标长度与点云不一致：{path}")
            cloud = Cloud(
                exact_points,
                exact_colors,
                data["source_index"],
                data["raw_count"],
                data["scan_id"],
                {k[7:]: data[k] for k in data.files if k.startswith("extra__")},
            )
    return cloud


# 保存 PCD 与来源旁文件


def write_cloud(path, cloud, overwrite=False):
    path = pcd_path(path)
    if (path.exists() or sidecar_path(path).exists()) and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有点云或旁文件：{path}")
    if not np.isfinite(cloud.points).all():
        raise ValueError("写出点云含 NaN/Inf，请先运行预处理；无效点写审计 NPZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing_", suffix=".pcd", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    temp_sidecar = Path(str(temporary) + ".npz")
    try:
        payload = {
            "source_index": cloud.source_index,
            "raw_count": cloud.raw_count,
            "scan_id": cloud.scan_id,
            **{"extra__" + k: v for k, v in cloud.extras.items()},
        }
        if len(cloud) == 0:
            temporary.write_text(
                "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 0\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 0\nDATA ascii\n",
                encoding="ascii",
            )
        else:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud.points))
            if cloud.colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(cloud.colors.astype(np.float64) / 255)
            if not o3d.io.write_point_cloud(
                str(temporary), pcd, write_ascii=False, compressed=False
            ):
                raise IOError(f"点云写出失败：{path}")
        np.savez_compressed(
            temp_sidecar,
            cloud_sha256=np.array(sha256(temporary)),
            color_encoding=np.array("uint8_0_255"),
            exact_points=cloud.points,
            **({"exact_colors": cloud.colors} if cloud.colors is not None else {}),
            **payload,
        )
        os.replace(temporary, path)
        os.replace(temp_sidecar, sidecar_path(path))
    finally:
        temporary.unlink(missing_ok=True)
        temp_sidecar.unlink(missing_ok=True)
    return str(path)
