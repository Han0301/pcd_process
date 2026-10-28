"""从单次 METABAG 生成去畸变帧、定位结果和 TemporalFilter 所需 CSV。

执行原始索引核验与同次 LAS 帧定位；路径及流程参数读取 PrepareTemporalFrames 节。
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from Common.PointCloudIO import sha256, write_report
from RawData.validate_frames import validate


def build(jobs=2):
    """按项目内源码指纹复用或重建离线 LIO，返回可执行文件路径。"""
    directory = ROOT / "build/raw_fastlio"
    binary = directory / "fastlio_offline"
    sources = {
        str(p.relative_to(HERE)): sha256(p)
        for p in HERE.rglob("*")
        if p.suffix in (".cpp", ".h", ".hpp", ".xml", ".msg") or p.name == "CMakeLists.txt"
    }
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    stamp = directory / "source_sha256.txt"
    if binary.exists() and stamp.exists() and stamp.read_text().strip() == fingerprint:
        return binary
    directory.mkdir(parents=True, exist_ok=True)
    # 构建只使用本项目源码；ROS、PCL 和 Livox 消息是明示的系统依赖。
    command = [
        "cmake",
        "-S",
        str(HERE / "fast_lio"),
        "-B",
        str(directory),
        "-DPYTHON_EXECUTABLE=/usr/bin/python3",
        "-DPython3_EXECUTABLE=/usr/bin/python3",
        "-DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so",
        "-DPYTHON_INCLUDE_DIR=/usr/include/python3.12",
        "-DCMAKE_PREFIX_PATH=/opt/ros/jazzy;/home/h/lidar_ws/install/livox_ros_driver2",
    ]
    for args in [command, ["cmake", "--build", str(directory), "-j", str(jobs)]]:
        subprocess.run(args, check=True, cwd=ROOT)
    stamp.write_text(fingerprint + "\n")
    write_report(
        directory / "build_provenance.json", {"sources": sources, "configure_command": command}
    )
    return binary


def prepare(
    source_dir,
    input_pcd,
    output_dir,
    left_transform_file,
    allow_free_space=False,
    frame_period=0.5,
    duration=0.0,
    build_jobs=2,
):
    """核验原始录制并生成位姿 CSV；input_pcd 为调平前的预处理云。"""
    source, target, out, left = map(
        lambda p: Path(p).expanduser().resolve(),
        (source_dir, input_pcd, output_dir, left_transform_file),
    )
    if out.exists():
        raise FileExistsError(out)
    if not target.is_file() or not left.is_file():
        raise FileNotFoundError("先完成点云预处理与地面调平")
    binary = build(build_jobs)
    raw = sorted((source / "data").glob("data_*"), key=lambda p: int(p.name[5:]))
    if not raw:
        raise FileNotFoundError(source / "data")
    hashes = {str(p): sha256(p) for p in raw}
    hashes[str(target)] = sha256(target)
    hashes[str(left)] = sha256(left)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "source_dir": str(source),
        "input_hashes": hashes,
        "stages": [],
        "uses_historical_outputs": False,
        "same_capture_target_is_independent_truth": False,
        "allow_free_space_with_downstream_optical_guards": allow_free_space,
        "executable_sha256": sha256(binary),
        "build_provenance_sha256": sha256(binary.parent / "build_provenance.json"),
    }
    start = time.monotonic()

    def run(name, args, env=None):
        begin = time.monotonic()
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        report["stages"].append(
            {
                "stage": name,
                "command": list(map(str, args)),
                "exit_code": result.returncode,
                "seconds": time.monotonic() - begin,
            }
        )
        if result.returncode:
            raise RuntimeError(f"{name} 失败：{out / (name + '.log')}")

    try:
        run(
            "audit",
            [
                sys.executable,
                str(HERE / "audit_metabag.py"),
                "--source",
                str(source),
                "--output",
                str(out / "raw_audit"),
            ],
        )
        run(
            "lio",
            [
                str(binary),
                "--data-dir",
                str(source / "data"),
                "--output",
                str(out / "lio"),
                "--duration",
                str(duration),
                "--frame-period",
                str(frame_period),
            ],
        )
        report["raw_index_validation"] = validate(source, out / "raw_audit", out / "lio")
        (out / "localization").mkdir()
        env = {**os.environ, "TEMPORAL_SCAN_DIR": str(out), "TEMPORAL_TARGET_PCD": str(target)}
        run(
            "localization",
            [
                sys.executable,
                str(HERE / "localize_frames.py"),
                "--mode",
                "full",
                "--work-dir",
                str(out),
                "--input-pcd",
                str(target),
            ],
            env,
        )
        rows = list(csv.DictReader((out / "localization/full/localized_frames.csv").open()))
        fields = [
            "file",
            "point_format",
            "stamp",
            "accepted_geometry",
            "accepted_for_free_space",
            "holdout_plane_p95_m",
            "holdout_distance_p95_m",
            "left_transform_file",
        ] + [f"m{i}{j}" for i in range(4) for j in range(4)]
        exported = []
        for row in rows:
            accepted = (
                row["accepted_geometry"].lower() == "true"
                and row["accepted_temporal_geometry"].lower() == "true"
            )
            item = {
                "file": str(out / "lio" / row["file"]),
                "point_format": "xyzi_float32_le",
                "stamp": row["stamp"],
                "accepted_geometry": accepted,
                "accepted_for_free_space": bool(accepted and allow_free_space),
                "holdout_plane_p95_m": row["fixed_holdout_plane_p95_after_m"],
                "holdout_distance_p95_m": row["holdout_distance_p95_m"],
                "left_transform_file": str(left),
            }
            item.update({f"m{i}{j}": row[f"m{i}{j}"] for i in range(4) for j in range(4)})
            exported.append(item)
        with (out / "frames.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(exported)
        if not any(r["accepted_geometry"] for r in exported):
            raise RuntimeError("无时序几何合格帧，拒绝把空证据当作筛选成功")
        if any(sha256(p) != h for p, h in hashes.items()):
            raise RuntimeError("原始数据或输入在执行中变化")
        report.update(
            status="complete",
            frames=len(rows),
            accepted_geometry=sum(r["accepted_geometry"] for r in exported),
            accepted_for_free_space=sum(r["accepted_for_free_space"] for r in exported),
            frames_csv=str(out / "frames.csv"),
            frames_csv_sha256=sha256(out / "frames.csv"),
            localized_frames_csv_sha256=sha256(out / "localization/full/localized_frames.csv"),
            input_data_unchanged=True,
        )
        return report
    except Exception as e:
        report.update(status="failed", error=str(e))
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        write_report(out / "prepare_report.json", report)


def main():
    from Common.Configuration import cli_config

    cfg = cli_config("Pipeline")
    s = cfg.section("PrepareTemporalFrames")
    result = prepare(
        s["source_dir"],
        s["input_pcd"],
        s["output_dir"],
        s["left_transform_file"],
        allow_free_space=s["allow_free_space"],
        **s["parameters"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
