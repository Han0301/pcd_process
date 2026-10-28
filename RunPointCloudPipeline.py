"""单份 PCD 的预处理、地面、裁剪与可选时序筛选入口。

输入单位为米；LAS 先由 FormatChange/01LastoPcd.py 转换。参数读取 Pipeline.yaml。
"""

import time
from importlib import import_module
from pathlib import Path

import numpy as np

from Common.Configuration import cli_config, load_config
from Common.OutputLayout import publish_cloud
from Common.PointCloudIO import pcd_path, sha256, sidecar_path, write_report

# 单份处理编排
preprocess = import_module("Pipeline.01PreprocessPointCloud").preprocess
extract_ground = import_module("Pipeline.02ExtractGround").extract_ground
crop_cloud = import_module("Pipeline.03CropPointCloud").crop_cloud
prepare_grid_input = import_module("Pipeline.04PrepareGridInput").prepare_grid_input
temporal_filter = import_module("Pipeline.05TemporalFilter").temporal_filter


def run_single_pipeline(
    input_pcd,
    output_dir,
    *,
    preprocess_params=None,
    ground_params=None,
    crop_params=None,
    export_grid_input=None,
    stage_output_dirs=None,
    grid_input_params=None,
    config_info=None,
    enable_temporal_filter=False,
    temporal_params=None,
    frames_csv=None,
    raw_frame_settings=None,
):
    """处理单份米制 PCD，发布带来源的 processed.pcd 并返回阶段报告。"""
    if export_grid_input is None:
        export_grid_input = load_config("Pipeline").section("RunPointCloudPipeline")[
            "export_grid_input"
        ]
    source = pcd_path(input_pcd)
    out = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if out.exists():
        raise FileExistsError(f"输出目录必须是新目录：{out}")
    stage_dirs = {
        "preprocess": out / "01_preprocess",
        "ground": out / "02_ground",
        "crop": out / "03_crop",
        "temporal": out / "04_temporal",
        "grid_input": out / "05_grid_input",
    }
    stage_dirs.update(stage_output_dirs or {})
    configs = {
        "preprocess": dict(preprocess_params or {}),
        "ground": dict(ground_params or {}),
        "crop": dict(crop_params or {}),
        "temporal": dict(temporal_params or {}),
    }
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    inputs = {str(source): sha256(source)}
    if sidecar_path(source).is_file():
        inputs[str(sidecar_path(source))] = sha256(sidecar_path(source))
    report = {
        "mode": "single_point_cloud",
        "status": "running",
        "input": str(source),
        "input_sha256": inputs[str(source)],
        "stages": {},
        "parameters": configs,
        "export_grid_input": bool(export_grid_input),
        "enable_temporal_filter": bool(enable_temporal_filter),
        "registration_performed": False,
        "fusion_performed": False,
    }
    report.update(config_info or {})
    if str(sidecar_path(source)) in inputs:
        report["input_sidecar_sha256"] = inputs[str(sidecar_path(source))]
    try:
        stages = report["stages"]
        stages["preprocess"] = preprocess(source, stage_dirs["preprocess"], **configs["preprocess"])
        stages["ground"] = extract_ground(
            stages["preprocess"]["output_pcd"], stage_dirs["ground"], **configs["ground"]
        )
        stages["crop"] = crop_cloud(
            stages["ground"]["output_pcd"], stage_dirs["crop"], **configs["crop"]
        )
        stage_output = stages["crop"]["output_pcd"]
        if enable_temporal_filter:
            if frames_csv is None:
                raise ValueError("启用时序筛选必须提供 frames_csv")
            if raw_frame_settings is not None:
                from RawData.prepare import prepare

                raw_settings = dict(raw_frame_settings)
                raw_parameters = raw_settings.pop("parameters")
                raw_settings.pop("enabled")
                stages["raw_temporal"] = prepare(**raw_settings, **raw_parameters)
                if (
                    Path(stages["raw_temporal"]["frames_csv"]).resolve()
                    != Path(frames_csv).resolve()
                ):
                    raise ValueError(
                        "PrepareTemporalFrames 输出与 TemporalFilter.frames_csv 不一致"
                    )
            stages["temporal"] = temporal_filter(
                stage_output, frames_csv, stage_dirs["temporal"], **configs["temporal"]
            )
            stage_output = stages["temporal"]["output_pcd"]
        output = publish_cloud(stage_output, out / "processed.pcd")
        transform = out / "T_input_to_processed.npy"
        np.save(
            transform,
            np.load(stages["ground"]["transform"], allow_pickle=False),
            allow_pickle=False,
        )
        report.update(
            output_pcd=output,
            stage_output_pcd=stage_output,
            output_sidecar=str(sidecar_path(output)),
            output_sha256=sha256(output),
            output_sidecar_sha256=sha256(sidecar_path(output)),
            transform_file=str(transform),
        )
        if export_grid_input:
            report["grid_input"] = prepare_grid_input(
                output, stage_dirs["grid_input"], **(grid_input_params or {})
            )
        if not all(sha256(p) == value for p, value in inputs.items()):
            raise RuntimeError("输入 PCD 或旁文件在运行中变化")
        report.update(status="complete", inputs_unchanged=True)
        print(f"单份 PCD 处理完成：{output}", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        write_report(out / "pipeline_report.json", report)


# 统一调用
def main(config=None):
    config = config or load_config("Pipeline")
    settings = config.section("RunPointCloudPipeline")
    modules = {
        "preprocess": "PreprocessPointCloud",
        "ground": "ExtractGround",
        "crop": "CropPointCloud",
        "temporal": "TemporalFilter",
        "grid_input": "PrepareGridInput",
    }
    stages = {key: config.section(name) for key, name in modules.items()}
    params = dict(stages["preprocess"]["parameters"])
    enabled = settings["enable_temporal_filter"]
    raw_settings = (
        config.section("PrepareTemporalFrames") if "PrepareTemporalFrames" in config.raw else None
    )
    raw_enabled = bool(enabled and raw_settings and raw_settings["enabled"])
    if enabled and not raw_enabled and not stages["temporal"]["frames_csv"].is_file():
        raise FileNotFoundError(stages["temporal"]["frames_csv"])
    required = [
        settings["output_dir"],
        stages["preprocess"]["output_dir"],
        stages["ground"]["output_dir"],
        stages["crop"]["output_dir"],
    ]
    if enabled:
        required.append(stages["temporal"]["output_dir"])
    if settings["export_grid_input"]:
        required.append(stages["grid_input"]["output_dir"])
    if raw_enabled:
        required.append(raw_settings["output_dir"])
    if len(required) != len(set(required)):
        raise ValueError("各模块输出目录必须不同")
    for directory in required:
        if directory.exists():
            raise FileExistsError(directory)
    source = settings["input_pcd"]
    if (
        source.stat().st_size > settings["auto_stream_threshold_bytes"]
        and not sidecar_path(source).exists()
    ):
        # ASCII/compressed PCD cannot use the native binary memmap route.
        from Common.PointCloudIO import binary_pcd

        try:
            binary_pcd(source)
        except ValueError:
            pass
        else:
            if params["voxel_size"] > 0 and not params["save_removed"]:
                params["streaming"] = True
    return run_single_pipeline(
        source,
        settings["output_dir"],
        preprocess_params=params,
        ground_params=stages["ground"]["parameters"],
        crop_params=stages["crop"]["parameters"],
        export_grid_input=settings["export_grid_input"],
        stage_output_dirs={key: value["output_dir"] for key, value in stages.items()},
        grid_input_params=stages["grid_input"]["parameters"],
        config_info=config.provenance(),
        enable_temporal_filter=enabled,
        temporal_params=stages["temporal"]["parameters"],
        frames_csv=stages["temporal"]["frames_csv"],
        raw_frame_settings=raw_settings if raw_enabled else None,
    )


if __name__ == "__main__":
    main(cli_config("Pipeline"))
