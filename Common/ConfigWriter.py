"""保存可独立运行的 YAML 快照；注释按相邻短行对齐。"""

import re
from pathlib import Path

import yaml


class _Dumper(yaml.SafeDumper):
    pass


def _sequence(dumper, value):
    inline = all(not isinstance(v, (list, dict)) for v in value) and len(str(value)) < 100
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=inline)


_Dumper.add_representer(list, _sequence)

NOTES = {
    "enabled": "是否从原始 METABAG 重新生成帧，禁止隐式复用历史结果",
    "source_dir": "本次原始录制目录，内部须含 data/data_*",
    "left_transform_file": "RAW LAS 到本次调平坐标的矩阵，由 ExtractGround 生成",
    "allow_free_space": "是否允许合格帧做射线见证；仍须下游保护框与独立视角检查",
    "frame_period": "LIO 去畸变帧导出间隔，秒",
    "duration": "读取录制时长，秒；0 表示完整录制",
    "build_jobs": "首次构建项目内 LIO 源码的并发数",
    "register_config": "配准与地图参数模板，与本文件共用 ROOT_DIR 和批次",
    "scans": "按顺序处理的原始录制；矩阵仅作初值，仍须通过配准验收",
    "initial_raw_to_reference": "本录制 RAW LAS 到参考录制 RAW LAS 的粗配准初值",
    "reference_scan": "最终融合使用的参考录制名称",
    "reference_raw_to_map": "参考录制 RAW LAS 到投影坐标的固定刚体变换",
    "export_preview": "是否从最终融合点云导出 PLY 预览",
    "map_quarter_turns": "另存地图的逆时针90度旋转次数，保留原投影地图",
    "map_translation_xy": "另存地图坐标的平移，米；与旋转一起应用于地图原点",
    "PrepareTemporalFrames": "原始 data 准备：解码、去畸变、帧定位、来源核验",
    "RunDataset": "完整批次入口，所有输入与初值均显式配置",
    "name": "录制名称，对应本批次的单云目录",
}


def _key_paths(node, path=()):
    """按 YAML 层级定位映射键；序列项使用通用路径，便于跨扫描复用说明。"""
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            child_path = (*path, key.value)
            yield key.start_mark.line, child_path
            yield from _key_paths(value, child_path)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            yield from _key_paths(item, (*path, "*"))


def save_config(path, data, comment_sources=()):
    """写出完整 YAML；按模块保留模板注释，短参数相邻对齐，长行就近注释。"""
    comments = {}
    for source in comment_sources:
        content = Path(source).read_text(encoding="utf-8")
        lines = content.splitlines()
        for line_number, key_path in _key_paths(yaml.compose(content)):
            match = re.search(r"\s+#\s*(.*)", lines[line_number])
            if match:
                comments.setdefault(key_path, match[1])
    content = yaml.dump(data, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=100000)
    key_paths = dict(_key_paths(yaml.compose(content)))
    pairs = []
    for number, line in enumerate(content.splitlines()):
        key_path = key_paths.get(number)
        note = (
            comments.get(key_path, NOTES.get(key_path[-1], "参数项"))
            if key_path
            else "列表按顺序配置"
        )
        pairs.append((line, note))
    rendered = []
    index = 0
    while index < len(pairs):
        line, _ = pairs[index]
        depth = len(line) - len(line.lstrip())
        end = index + 1
        if depth and not line.endswith(":"):
            while end < len(pairs):
                other = pairs[end][0]
                if len(other) - len(other.lstrip()) != depth or other.endswith(":"):
                    break
                end += 1
        short_lengths = [len(text) for text, _ in pairs[index:end] if len(text) <= 64]
        width = max(short_lengths, default=0)
        for text, comment in pairs[index:end]:
            padding = max(2, width - len(text) + 2) if len(text) <= 64 else 2
            rendered.append(text + " " * padding + "# " + comment)
        index = end
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rendered) + "\n", encoding="utf-8")
