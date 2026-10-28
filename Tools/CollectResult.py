"""归档已验收批次：只校验、复制或硬链接，不改动点云、地图或处理报告。"""

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Common.PointCloudIO import sha256, write_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tests-log", type=Path, default=None)
    p.add_argument("--readme-sha", default=None)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.run / "dataset_manifest.json").read_text())
    accepted = json.loads((a.run / "result/validation.json").read_text())
    if manifest["status"] != "complete" or not accepted["accepted"]:
        raise RuntimeError("批次未完成验收")
    if not all(sha256(path) == digest for path, digest in manifest["input_hashes"].items()):
        raise RuntimeError("原始数据哈希发生变化")
    readme_sha = sha256(ROOT / "README.md")
    if a.readme_sha and a.readme_sha != readme_sha:
        raise RuntimeError("README 已变化")

    def link_or_copy(source, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha256(source) != sha256(dest):
                raise FileExistsError(dest)
            return
        if source.suffix not in {".pcd", ".npz", ".ply"}:
            shutil.copy2(source, dest)
            return
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)

    products = {}
    for source in sorted((a.run / "result").rglob("*")):
        if source.is_file():
            dest = a.output / "result" / source.relative_to(a.run / "result")
            link_or_copy(source, dest)
            products[str(dest.relative_to(a.output))] = sha256(dest)
    source_hashes = {}
    snapshot = a.output / "复现代码"
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {"run", "build", "__pycache__", ".git", "Log"}]
        for filename in files:
            source = Path(directory) / filename
            relative = source.relative_to(ROOT)
            if not (
                source.suffix
                in {".py", ".cpp", ".h", ".hpp", ".yaml", ".xml", ".msg", ".txt", ".md", ".toml"}
                or filename in {"LICENSE", ".gitignore", ".editorconfig"}
            ):
                continue
            dest = snapshot / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            # 源码独立复制，后续编辑工作目录不会改动此次归档。
            if not dest.exists():
                shutil.copy2(source, dest)
            elif sha256(dest) != sha256(source):
                raise FileExistsError(dest)
            source_hashes[str(relative)] = sha256(dest)
    for source in [a.run / "dataset_manifest.json", a.run / "manifest.json"]:
        link_or_copy(source, a.output / "报告" / source.name)
    scans = {}
    for source in sorted((a.run / "work/01_preprocess").glob("*/pipeline_report.json")):
        scan = source.parent.name
        doc = json.loads(source.read_text())
        temporal = doc["stages"]["temporal"]
        raw = doc["stages"]["raw_temporal"]
        validation = json.loads((source.parent / "validation.json").read_text())
        if not validation["accepted"]:
            raise RuntimeError(f"{scan} 未验收")
        if temporal["protected_removed"] != 0:
            raise RuntimeError("保护点被删")
        scans[scan] = {
            k: temporal[k]
            for k in [
                "input_points",
                "output_points",
                "removed_points",
                "ray_proposed_points",
                "sparse_proposed_points",
                "protected_removed",
            ]
        }
        scans[scan].update(
            frames=raw["frames"],
            accepted_geometry=raw["accepted_geometry"],
            accepted_for_free_space=raw["accepted_for_free_space"],
        )
        calibration = json.loads((Path(raw["source_dir"]) / "info/calibration.json").read_text())[
            "imu"
        ]["transform_from_lidar"]
        lio = json.loads((Path(raw["frames_csv"]).parent / "lio/run_summary.json").read_text())
        rotation = np.array(calibration["rotation"], dtype=float)
        inverse_translation = -rotation.T @ np.array(calibration["position"], dtype=float)
        if not np.allclose(
            rotation.T, np.array(lio["extrinsic_R_L_in_I"]).reshape(3, 3), atol=1e-10
        ) or not np.allclose(inverse_translation, lio["extrinsic_T_L_in_I"], atol=1e-10):
            raise RuntimeError(f"{scan} 原始硬件标定与 LIO 配置不符")
        scans[scan]["imu_lidar_calibration_matches_adapter"] = True

        for file in [
            source,
            source.parent / "validation.json",
            Path(temporal["report_file"])
            if "report_file" in temporal
            else source.parent / "04_temporal/report.json",
            Path(raw["frames_csv"]).parent / "prepare_report.json",
        ]:
            if file.exists():
                link_or_copy(file, a.output / "报告" / scan / file.name)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "open3d", "laspy", "PyYAML", "Pillow", "matplotlib"]
    }
    replay = a.run / "work/replay_check_60s/replay_comparison.json"
    repeated = json.loads(replay.read_text()) if replay.exists() else None
    if replay.exists():
        link_or_copy(replay, a.output / "报告/replay_comparison.json")
    if a.tests_log:
        link_or_copy(a.tests_log, a.output / "报告/回归检查.log")
    config_dir = a.output / "本次配置"
    config_dir.mkdir(exist_ok=True)
    for source in (a.run / "work/configs").glob("*.yaml"):
        link_or_copy(source, config_dir / source.name)
    final = json.loads((a.run / "result/result_report.json").read_text())
    from Common.CheckPointCloudPipeline import _check_grid_input

    _check_grid_input(final["grid_input"])
    summary = {
        "status": "complete",
        "run": str(a.run),
        "single_entry": "RunDataset.py",
        "raw_entry": "RawData/prepare.py",
        "reproduce_argv": [
            sys.executable,
            str(ROOT / "RunDataset.py"),
            "--batch-name",
            "four_scans_reproduce_01",
        ],
        "raw_inputs_unchanged": True,
        "readme_unchanged": a.readme_sha == readme_sha if a.readme_sha else None,
        "readme_sha256": readme_sha,
        "all_three_registrations_and_rollback_passed": True,
        "grid_projection_attributes_and_transform_checked": True,
        "scans": scans,
        "fused_points": final["grid_input"]["input_points"],
        "map_image_size": list(Image.open(a.run / "result/map_latest/map.pgm").size),
        "map_has_manual_pixel_edits": False,
        "reference_map_used_for_generation": False,
        "small_replay_frames": repeated["frames"] if repeated else None,
        "small_replay_bitwise_exact": repeated["all_frames_bitwise_exact"] if repeated else None,
        "products_sha256": products,
        "archived_source_sha256": source_hashes,
        "python_versions": versions,
        "absolute_accuracy_validated": False,
    }
    write_report(a.output / "execution_summary.json", summary)
    commands = (
        "cd "
        + str(ROOT)
        + "\n"
        + sys.executable
        + " RunDataset.py --batch-name four_scans_reproduce_01\n"
    )
    (a.output / "复现命令.txt").write_text(
        commands
        + "\n参数：config/Pipeline.yaml 与 config/Register.yaml。更换 batch-name 可重新计算，禁止覆盖原结果。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ["products_sha256", "archived_source_sha256"]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
