"""读取两类 YAML、检查字段与类型，并解析共享目录和模块间路径引用。"""

import argparse
import functools
import hashlib
import inspect
import math
import re
from copy import deepcopy
from pathlib import Path

import yaml

from Common.ConfigSchema import PARAMETER_TYPES, SECTION_FIELDS
from Common.OutputLayout import OutputLayout


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError(f"YAML 键必须是字符串，第 {key_node.start_mark.line + 1} 行")
        if key in result:
            raise ValueError(f"YAML 重复键 {key!r}，第 {key_node.start_mark.line + 1} 行")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
_PATH_FIELDS = {
    "input_pcd",
    "input_las",
    "output_pcd",
    "output_dir",
    "report_dir",
    "source_pcd",
    "target_pcd",
    "source",
    "target",
    "initial_transform",
    "frames_csv",
    "source_dir",
    "left_transform_file",
    "register_config",
}
_LIST_PATH_FIELDS = {"input_pcds", "registration_reports"}
_BOOL_FIELDS = {
    "enabled",
    "allow_free_space",
    "export_grid_input",
    "enable_temporal_filter",
    "fuse_after_registration",
    "export_map",
    "write_ascii",
    "compressed",
}


def validate_parameters(module, parameters, partial=False):
    schema = PARAMETER_TYPES[module]
    if not isinstance(parameters, dict):
        raise ValueError(f"{module}.parameters 必须是键值表")
    unknown = set(parameters) - set(schema)
    missing = set(schema) - set(parameters) if not partial else set()
    if unknown or missing:
        raise ValueError(
            f"{module}.parameters 未知字段 {sorted(unknown)}，缺少字段 {sorted(missing)}"
        )
    for key, value in parameters.items():
        kind = schema[key]
        valid = (
            (kind == "bool" and type(value) is bool)
            or (kind == "int" and type(value) is int)
            or (
                kind in ("number", "nullable_number")
                and type(value) in (int, float)
                and math.isfinite(value)
            )
            or (kind in ("str", "nullable_str") and isinstance(value, str))
            or (kind in ("list", "nullable_list") and isinstance(value, list))
            or (kind.startswith("nullable_") and value is None)
        )
        if not valid:
            raise ValueError(f"{module}.{key} 类型应为 {kind}，收到 {value!r}")

    def positive(keys, allow_zero=False):
        for key in keys:
            if key in parameters and (parameters[key] < 0 if allow_zero else parameters[key] <= 0):
                raise ValueError(f"{module}.{key} 必须为{'非负' if allow_zero else '正'}数")

    if module == "PrepareTemporalFrames":
        positive(["frame_period", "build_jobs"])
        positive(["duration"], allow_zero=True)
    if module == "PreprocessPointCloud":
        positive(["voxel_size"], allow_zero=True)
        positive(["std_ratio"])
        if "nb_neighbors" in parameters and parameters["nb_neighbors"] < 2:
            raise ValueError("nb_neighbors 必须至少为 2")
        if (
            not partial
            and parameters["streaming"]
            and (parameters["voxel_size"] <= 0 or parameters["save_removed"])
        ):
            raise ValueError("分块处理要求 voxel_size>0 且 save_removed=false")
    if module in ("PcdtoGridMap", "FusePointClouds", "CropPointCloud"):
        positive(["resolution"])
    if module == "PcdtoGridMap":
        positive(
            [
                "ground_tolerance",
                "min_obstacle_points",
                "statistical_std_ratio",
                "ground_histogram_bin_size",
            ]
        )
        positive(["inflation_radius"], allow_zero=True)
        if "map_name" in parameters:
            _filename(parameters["map_name"], "map_name")
    if module == "TemporalFilter":
        positive(
            [
                "horizontal_guard_dilation",
                "ray_max_hit_frames",
                "sparse_raw_count_max",
                "sparse_max_hit_frames",
                "sparse_nearest_distance",
                "sparse_eighth_distance",
                "sparse_stable_distance",
            ],
            allow_zero=True,
        )
        positive(
            [
                "hit_radius",
                "min_origin_distance",
                "min_time_distance",
                "ray_distance",
                "endpoint_margin",
                "normal_radius",
            ]
        )
        positive(["max_plane_p95", "max_nn_p95", "ground_guard_height"], allow_zero=True)
        if "independent_views" in parameters and not 1 <= parameters["independent_views"] <= 255:
            raise ValueError("independent_views 必须在 1 到 255 之间")
        if "normal_neighbors" in parameters and parameters["normal_neighbors"] < 3:
            raise ValueError("normal_neighbors 必须至少为 3")
        for key in ["frame_range", "query_range"]:
            if key in parameters:
                values = parameters[key]
                if (
                    len(values) != 2
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
                    or not 0 <= values[0] < values[1]
                ):
                    raise ValueError(f"{key} 须为递增的非负有限距离对")
        if "protected_fields" in parameters and not all(
            isinstance(x, str) and x for x in parameters["protected_fields"]
        ):
            raise ValueError("protected_fields 必须为非空属性名列表")
    enum = {
        ("CropPointCloud", "mode"): {"none", "bounds", "quantile_xy"},
        ("PrepareGridInput", "selection_mode"): {"local_ground", "global_height"},
        ("TemporalFilter", "mode"): {"review", "apply"},
        ("RegisterPointClouds", "fit_mode"): {"classified", "all_finite"},
        ("FusePointClouds", "priority"): {"first", "last"},
    }
    for (name, key), allowed in enum.items():
        if module == name and key in parameters and parameters[key] not in allowed:
            raise ValueError(f"{module}.{key} 必须为 {sorted(allowed)}")
    return deepcopy(parameters)


def _filename(value, label):
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or Path(value).name != value
        or "\\" in value
    ):
        raise ValueError(f"{label} 必须是单个文件名")


def complete_parameters(kind, module, supplied):
    validate_parameters(module, supplied, partial=True)
    if set(supplied) == set(PARAMETER_TYPES[module]):
        return validate_parameters(module, supplied)
    return validate_parameters(module, {**load_config(kind).parameters(module), **supplied})


def configured_defaults(kind, module):
    """仅在调用者省略参数时读取默认 YAML，导入模块不读取默认配置。"""

    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            supplied = {
                key: value
                for key, value in bound.arguments.items()
                if key in PARAMETER_TYPES[module]
            }
            parameters = complete_parameters(kind, module, supplied)
            for key, value in parameters.items():
                if key not in bound.arguments:
                    bound.arguments[key] = value
            return function(*bound.args, **bound.kwargs)

        return wrapped

    return decorate


class Configuration:
    def __init__(self, kind, path=None):
        if kind not in SECTION_FIELDS:
            raise ValueError(f"未知配置类别：{kind}")
        self.kind = kind
        self.path = (
            Path(path).expanduser().resolve()
            if path is not None
            else Path(__file__).resolve().parents[1] / "config" / f"{kind}.yaml"
        )
        try:
            content = self.path.read_bytes()
            self.config_sha256 = hashlib.sha256(content).hexdigest()
            self.raw = yaml.load(content.decode("utf-8"), Loader=_UniqueLoader)
        except (yaml.YAMLError, ValueError) as error:
            raise ValueError(f"配置文件 {self.path}：{error}") from error
        if not isinstance(self.raw, dict):
            raise ValueError(f"配置文件必须是键值表：{self.path}")
        expected = {"ROOT_DIR", *SECTION_FIELDS[kind]}
        optional = {"PrepareTemporalFrames", "RunDataset"} if kind == "Pipeline" else set()
        if set(self.raw) - expected or expected - set(self.raw) - optional:
            raise ValueError(
                f"{self.path} 顶层字段不符；未知 {sorted(set(self.raw) - expected)}，缺少 {sorted(expected - set(self.raw))}"
            )
        root = self.raw["ROOT_DIR"]
        if not isinstance(root, str) or not root:
            raise ValueError("ROOT_DIR 必须是非空路径字符串")
        root = Path(root).expanduser()
        self.root_dir = (root if root.is_absolute() else self.path.parent / root).resolve()
        for name, fields in SECTION_FIELDS[kind].items():
            if name not in self.raw:
                continue
            section = self.raw[name]
            if not isinstance(section, dict) or set(section) != set(fields):
                raise ValueError(f"{name} 字段必须为 {fields}")
            if "parameters" in section:
                validate_parameters(name, section["parameters"])
            for key, value in section.items():
                if key in _BOOL_FIELDS and type(value) is not bool:
                    raise ValueError(f"{name}.{key} 必须为 true 或 false")
                if key in _PATH_FIELDS:
                    valid = (
                        (value is None or isinstance(value, (str, list)))
                        if key == "initial_transform"
                        else isinstance(value, str) and bool(value)
                    )
                    if not valid:
                        raise ValueError(f"{name}.{key} 路径类型不合法")
                if key == "auto_stream_threshold_bytes" and (type(value) is not int or value <= 0):
                    raise ValueError("auto_stream_threshold_bytes 必须为正整数")
                if key in _LIST_PATH_FIELDS:
                    if not isinstance(value, list) or any(
                        not isinstance(v, str) and not (key == "registration_reports" and v is None)
                        for v in value
                    ):
                        raise ValueError(f"{name}.{key} 必须是路径列表")
                if key in ("output_ply_name", "validation_name"):
                    _filename(value, key)
                if key == "pcd_data_format" and value not in ("binary", "ascii"):
                    raise ValueError("pcd_data_format 必须为 binary 或 ascii")
                if key == "transform_atol" and (
                    type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                ):
                    raise ValueError("transform_atol 必须为正有限数")
        run = self.raw["run"]
        if not all(isinstance(v, str) for v in run.values()):
            raise ValueError("run 中的批次、试验和扫描名称必须是字符串")
        self.layout = OutputLayout(self.root_dir, run)
        self.variables = self.layout.variables()
        if kind == "Register":
            jobs = self.raw["RunRegistration"]["jobs"]
            required = {
                "name",
                "source",
                "target",
                "output_dir",
                "initial_transform",
                "parameters",
                "fusion_parameters",
            }
            if not isinstance(jobs, list) or not jobs:
                raise ValueError("RunRegistration.jobs 必须是非空列表")
            for index, job in enumerate(jobs):
                if not isinstance(job, dict) or set(job) != required:
                    raise ValueError(f"jobs[{index}] 字段必须为 {sorted(required)}")
                for key in ["name", "source", "target", "output_dir"]:
                    if not isinstance(job[key], str) or not job[key]:
                        raise ValueError(f"jobs[{index}].{key} 必须为非空字符串")
                if (
                    job["source"] == "previous"
                    or job["output_dir"] == "previous"
                    or (index == 0 and job["target"] == "previous")
                ):
                    raise ValueError("previous 只能用作第二轮及以后的 target")
                validate_parameters("RegisterPointClouds", job["parameters"], partial=True)
                validate_parameters("FusePointClouds", job["fusion_parameters"], partial=True)
        # Resolve all paths before algorithms create output directories.
        for name in SECTION_FIELDS[kind]:
            if name in self.raw:
                self.section(name)

    def with_context(self, **variables):
        clone = object.__new__(type(self))
        clone.__dict__ = self.__dict__.copy()
        clone.variables = {**self.variables, **variables}
        return clone

    def _reference(self, key, chain):
        if key in chain:
            raise ValueError(f"配置路径循环引用：{' -> '.join((*chain, key))}")
        if key in self.variables:
            return str(self.variables[key])
        node = self.raw
        try:
            for part in key.split("."):
                node = node[int(part)] if isinstance(node, list) else node[part]
        except (KeyError, ValueError, IndexError, TypeError) as error:
            raise ValueError(f"未知路径占位符 {{{key}}}") from error
        if not isinstance(node, (str, int, float)):
            raise ValueError(f"路径引用 {{{key}}} 必须指向字符串或数字")
        return self._expand(str(node), (*chain, key))

    def _expand(self, value, chain=()):
        return re.sub(r"\{([^{}]+)\}", lambda match: self._reference(match.group(1), chain), value)

    def _resolve(self, value, key=""):
        if isinstance(value, dict):
            return {k: self._resolve(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [
                self._resolve(v, "input_pcd" if key in _LIST_PATH_FIELDS else "") for v in value
            ]
        if isinstance(value, str):
            value = self._expand(value)
            if key in _PATH_FIELDS and not (key == "target" and value == "previous"):
                path = Path(value).expanduser()
                return (path if path.is_absolute() else self.root_dir / path).resolve()
        return value

    def section(self, name):
        if name not in self.raw:
            raise ValueError(f"{self.kind}.yaml 不包含模块 {name}")
        return self._resolve(deepcopy(self.raw[name]))

    def parameters(self, name):
        return self.section(name)["parameters"]

    def provenance(self):
        return {
            "config_file": str(self.path),
            "config_sha256": self.config_sha256,
            "configuration": deepcopy(self.raw),
        }


def load_config(kind="Pipeline", config_path=None):
    return Configuration(kind, config_path)


def cli_config(default="Pipeline"):
    parser = argparse.ArgumentParser(description="路径和参数均读取 config 下的 YAML")
    parser.add_argument(
        "--config", type=Path, default=None, help=f"配置文件，默认 config/{default}.yaml"
    )
    args = parser.parse_args()
    if args.config is None:
        return load_config(default)
    # Shared modules select their YAML category from its root entry sections.
    data = yaml.load(args.config.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    kind = "Register" if isinstance(data, dict) and "RunRegistration" in data else "Pipeline"
    return load_config(kind, args.config)
