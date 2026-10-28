"""串联原始 LAS、METABAG、单云处理、配准融合和地图导出的完整入口。

输入名单、初值和坐标约定读取 Pipeline.yaml；配准参数读取 Register.yaml。
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from Common.Configuration import load_config
from Common.ConfigWriter import save_config
from Common.PointCloudIO import sha256, validate_transform, write_report

ROOT = Path(__file__).resolve().parent


def export_map_frame(directory, quarter_turns, translation):
    """栅格仅做整数90度坐标变换，不补点、不画墙、不删像素。"""
    source = directory / "map"
    target = directory / "map_latest"
    if target.exists():
        raise FileExistsError(target)
    matrix = validate_transform(
        np.array(
            [[0.0, -1, 0, translation[0]], [1, 0, 0, translation[1]], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
    )
    if quarter_turns != 1:
        raise ValueError("当前地图坐标导出约定为逆时针90度；其他角度请显式实现坐标原点推导")
    settings = yaml.safe_load((source / "map.yaml").read_text())
    arr = np.array(Image.open(source / "map.pgm"))
    resolution = settings["resolution"]
    h, w = arr.shape
    x, y, _ = settings["origin"]
    settings["origin"] = [
        float(translation[0] - (y + h * resolution)),
        float(translation[1] + x),
        0.0,
    ]
    turned = np.rot90(arr, quarter_turns)
    target.mkdir()
    Image.fromarray(turned).save(target / "map.pgm")
    Image.fromarray(turned).save(target / "map.png")
    (target / "map.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    np.save(target / "T_map_to_latest.npy", matrix, allow_pickle=False)
    original = np.array(Image.open(source / "map.png"))
    if not np.array_equal(np.rot90(original, quarter_turns), turned):
        raise RuntimeError("PGM/PNG 像素不一致")
    return {
        "directory": str(target),
        "rotation_quarter_turns": quarter_turns,
        "translation_xy": translation,
        "pixel_changes_other_than_rotation": False,
        "origin": settings["origin"],
        "size": [turned.shape[1], turned.shape[0]],
    }


def run(config_path=None, batch_name=None):
    """按两份 YAML 运行完整批次；batch_name 可覆盖输出名，返回批次报告。"""
    config = load_config("Pipeline", config_path)
    base = copy.deepcopy(config.raw)
    base["ROOT_DIR"] = str(config.root_dir)
    if batch_name is not None:
        base["run"]["batch_name"] = batch_name
    dataset = config.section("RunDataset")
    params = dataset["parameters"]
    scans = dataset["scans"]
    if not scans:
        raise ValueError("至少需要一份录制")
    single_scan = len(scans) == 1
    reference = scans[0]["name"] if single_scan else params["reference_scan"]
    names = [s["name"] for s in scans]
    if len(set(names)) != len(names) or reference not in names:
        raise ValueError("录制名须唯一，且包含参考录制")
    for name in names:
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("录制名称须为单个目录名")
    from Common.OutputLayout import OutputLayout

    layout = OutputLayout(config.root_dir, base["run"])
    active = layout.active_root
    if active.exists():
        raise FileExistsError(f"拒绝覆盖已有批次：{active}；请换 --batch-name")
    templates = [config.path]
    register = None
    if not single_scan:
        register = load_config("Register", dataset["register_config"])
        templates.append(dataset["register_config"])
    rawpaths = {}
    for item in scans:
        source = Path(item["input_las"])
        if not source.is_file():
            raise FileNotFoundError(source)
        files = sorted((source.parent / "data").glob("data_*"), key=lambda p: int(p.name[5:]))
        if not files:
            raise FileNotFoundError(source.parent / "data")
        rawpaths.update({str(p): sha256(p) for p in [source, *files]})
        if not single_scan:
            validate_transform(item["initial_raw_to_reference"])
    active.mkdir(parents=True)
    work = layout.work_dir
    (work / "configs").mkdir(parents=True)
    (work / "logs").mkdir()
    report = {
        "status": "running",
        "input_hashes": rawpaths,
        "uses_historical_run": False,
        "stages": [],
        "source_hashes": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in ROOT.rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".cpp", ".h", ".hpp", ".msg")
            and not {"run", "build", "__pycache__"}.intersection(p.relative_to(ROOT).parts)
        },
    }
    start = time.monotonic()
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }

    def stage(name, script, cfg):
        begin = time.monotonic()
        print("START", name, flush=True)
        command = [sys.executable, str(ROOT / script), "--config", str(cfg)]
        with (work / "logs" / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        report["stages"].append(
            {
                "stage": name,
                "command": command,
                "exit_code": result.returncode,
                "elapsed_seconds": time.monotonic() - begin,
            }
        )
        write_report(active / "dataset_manifest.json", report)
        if result.returncode:
            raise RuntimeError(f"{name} 失败：{work / 'logs' / f'{name}.log'}")
        print("END", name, flush=True)

    try:
        for item in scans:
            data = copy.deepcopy(base)
            data["run"]["scan_name"] = item["name"]
            data["LastoPcd"]["input_las"] = str(item["input_las"])
            data["PrepareTemporalFrames"].update(
                enabled=True,
                source_dir=str(Path(item["input_las"]).parent),
                allow_free_space=item["allow_free_space"],
            )
            data["TemporalFilter"]["parameters"]["protected_bounds"] = item["protected_bounds"]
            data["RunPointCloudPipeline"].update(
                export_grid_input=single_scan, enable_temporal_filter=True
            )
            cfg = work / "configs" / f"Pipeline_{item['name']}.yaml"
            save_config(cfg, data, templates)
            stage(item["name"] + "_convert", "FormatChange/01LastoPcd.py", cfg)
            stage(item["name"] + "_pipeline", "RunPointCloudPipeline.py", cfg)
            stage(item["name"] + "_check", "Common/CheckPointCloudPipeline.py", cfg)
        if single_scan:
            # 单份使用自身处理坐标和 Pipeline 导图参数，无需配准任务或矩阵。
            from Common.OutputLayout import publish_cloud
            from Common.PointCloudIO import sidecar_path

            single_config = load_config("Pipeline", cfg)
            scan_dir = single_config.layout.scan_dir
            pipeline = json.loads((scan_dir / "pipeline_report.json").read_text())
            validation = json.loads((scan_dir / "validation.json").read_text())
            if pipeline["status"] != "complete" or not validation["accepted"]:
                raise RuntimeError("单份处理未完成验收")
            final_cloud = publish_cloud(
                pipeline["output_pcd"], layout.result_dir / "cloud/processed.pcd"
            )
            if (
                sha256(final_cloud) != pipeline["output_sha256"]
                or sha256(sidecar_path(final_cloud)) != pipeline["output_sidecar_sha256"]
            ):
                raise RuntimeError("发布的单份点云或旁文件与处理结果不一致")
            stage("map", "FormatChange/03PcdtoGridMap.py", cfg)
            if params["export_preview"]:
                preview = copy.deepcopy(data)
                preview["PcdtoPly"].update(input_pcd=final_cloud, output_dir="{RESULT_DIR}/preview")
                preview_cfg = work / "configs/Preview.yaml"
                save_config(preview_cfg, preview, templates)
                stage("preview", "FormatChange/02PcdtoPly.py", preview_cfg)
            map_settings = single_config.section("PcdtoGridMap")
            map_dir = map_settings["output_dir"]
            summary = {
                "status": "complete",
                "mode": "single_scan",
                "registrations": [],
                "registration_performed": False,
                "fusion_performed": False,
                "final_cloud": final_cloud,
                "output_sha256": sha256(final_cloud),
                "output_sidecar_sha256": sha256(sidecar_path(final_cloud)),
                "transforms": None,
                "pipeline_report": str(scan_dir / "pipeline_report.json"),
                "grid_input": pipeline["grid_input"],
                "map_dir": str(map_dir),
                "unknown_as_free": map_settings["parameters"]["unknown_as_free"],
                "map_sha256": {
                    f"{map_settings['parameters']['map_name']}.{ext}": sha256(
                        map_dir / f"{map_settings['parameters']['map_name']}.{ext}"
                    )
                    for ext in ("png", "pgm", "yaml")
                },
            }
            write_report(layout.result_dir / "result_report.json", summary)
            write_report(
                layout.result_dir / "validation.json",
                {
                    "accepted": True,
                    "steps": [validation],
                    "published_cloud_matches_stage": True,
                    "scope": "单份来源、调平、时序、投影与发布一致性；未执行配准融合",
                },
            )
            write_report(
                active / "manifest.json",
                {
                    "status": "complete",
                    "mode": "single_scan",
                    "jobs": [],
                    "completed_steps": [],
                    "result": str(layout.result_dir),
                    "final_cloud": final_cloud,
                    "result_report": str(layout.result_dir / "result_report.json"),
                },
            )
            report.update(mode="single_scan", registration_performed=False, fusion_performed=False)
        else:
            ref_level = np.load(
                layout.scan_root / reference / "T_input_to_processed.npy", allow_pickle=False
            )
            data = copy.deepcopy(register.raw)
            data["ROOT_DIR"] = str(config.root_dir)
            data["run"] = {k: v for k, v in base["run"].items() if k != "scan_name"}
            (work / "00_input/initials").mkdir(parents=True, exist_ok=True)
            for item in scans:
                if item["name"] == reference:
                    continue
                level = np.load(
                    layout.scan_root / item["name"] / "T_input_to_processed.npy", allow_pickle=False
                )
                initial = (
                    ref_level @ np.array(item["initial_raw_to_reference"]) @ np.linalg.inv(level)
                )
                np.save(
                    work / "00_input/initials" / f"T_{item['name']}_to_full.npy",
                    validate_transform(initial),
                    allow_pickle=False,
                )
            transform = validate_transform(params["reference_raw_to_map"]) @ np.linalg.inv(
                ref_level
            )
            data["PrepareGridInput"]["parameters"]["coordinate_transform"] = transform.tolist()
            cfg = work / "configs/Register.yaml"
            save_config(cfg, data, templates)
            stage("registration", "RunRegistration.py", cfg)
            if params["export_preview"]:
                stage("preview", "FormatChange/02PcdtoPly.py", cfg)
            report["map_frame_export"] = export_map_frame(
                layout.result_dir, params["map_quarter_turns"], params["map_translation_xy"]
            )
        report["inputs_unchanged"] = all(sha256(p) == h for p, h in rawpaths.items())
        if not report["inputs_unchanged"]:
            raise RuntimeError("原始输入执行中变化")
        report.update(status="complete", result_dir=str(layout.result_dir))
        print("DATASET_COMPLETE", layout.result_dir, flush=True)
        return report
    except Exception as e:
        report.update(status="failed", error=str(e))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        write_report(active / "dataset_manifest.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--batch-name", default=None, help="新批次目录名；已有目录拒绝覆盖")
    args = parser.parse_args()
    run(args.config, args.batch_name)


if __name__ == "__main__":
    main()
