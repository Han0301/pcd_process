"""按 Register.yaml 逐轮配准、融合并检查处理后 PCD，最后导出地图。"""

import gc
import importlib.util
import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np

from Common.CheckPointCloudPipeline import check_pipeline
from Common.Configuration import cli_config, load_config
from Common.OutputLayout import publish_cloud
from Common.PointCloudIO import pcd_path, sha256, sidecar_path, validate_transform, write_report

# 两份 PCD 配准编排
register_clouds = import_module("Register.01RegisterPointClouds").register_clouds
fuse_clouds = import_module("Register.02FusePointClouds").fuse_clouds


def run_registration(
    source_pcd,
    target_pcd,
    output_dir,
    *,
    initial_transform=None,
    registration_params=None,
    fuse_after_registration=None,
    fusion_params=None,
    module_output_dirs=None,
):
    """配准两份处理云，通过验收后按需融合；返回矩阵、输出和阶段报告。"""
    if fuse_after_registration is None:
        fuse_after_registration = load_config("Register").section("RunRegistration")[
            "fuse_after_registration"
        ]
    source, target = (pcd_path(p) for p in (source_pcd, target_pcd))
    if not source.is_file() or not target.is_file():
        raise FileNotFoundError("SOURCE_PCD 和 TARGET_PCD 必须都存在")
    if source == target:
        raise ValueError("source 与 target 不能是同一个文件")
    out = Path(output_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"输出目录必须是新目录：{out}")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    inputs = {str(p): sha256(p) for p in (source, target)}
    sidecars = {
        str(sidecar_path(p)): sha256(sidecar_path(p))
        for p in (source, target)
        if sidecar_path(p).is_file()
    }
    dirs = {"registration": out / "01_registration", "fusion": out / "02_fusion"}
    dirs.update(module_output_dirs or {})
    report = {
        "mode": "two_processed_clouds",
        "status": "running",
        "source_pcd": str(source),
        "target_pcd": str(target),
        "input_sha256": inputs,
        "input_sidecar_sha256": sidecars,
        "preprocessing_performed": False,
        "parameters": {
            "registration": registration_params or {},
            "fusion": fusion_params or {},
            "fuse_after_registration": bool(fuse_after_registration),
        },
    }
    try:
        if isinstance(initial_transform, (str, Path)):
            path = Path(initial_transform).expanduser().resolve()
            initial_transform = (
                np.load(path, allow_pickle=False)
                if path.suffix.lower() == ".npy"
                else np.loadtxt(path)
            )
            report["initial_transform_input"] = str(path)
            report["initial_transform_sha256"] = sha256(path)
        if initial_transform is not None:
            initial_transform = validate_transform(initial_transform)
        result = register_clouds(
            source, target, dirs["registration"], initial_transform, **(registration_params or {})
        )
        report["registration"] = result
        if not result["accepted"]:
            report["status"] = "rejected"
            raise RuntimeError(f"配准未通过，停止融合：{result['rejection_reasons']}")
        report.update(
            aligned_pcd=result["output_pcd"],
            transform_file=result["transform_file"],
            output_pcd=result["output_pcd"],
        )
        if fuse_after_registration:
            report["fusion"] = fuse_clouds(
                [target, result["output_pcd"]],
                dirs["fusion"],
                registration_reports=[None, result["report_file"]],
                **(fusion_params or {}),
            )
            report["output_pcd"] = report["fusion"]["output_pcd"]
        if not all(sha256(p) == digest for p, digest in {**inputs, **sidecars}.items()):
            raise RuntimeError("输入点云或来源旁文件在运行中变化")
        report.update(
            status="complete", inputs_unchanged=True, output_sha256=sha256(report["output_pcd"])
        )
        print(f"配准完成：{report['aligned_pcd']}", flush=True)
        return report
    except Exception as error:
        if report["status"] != "rejected":
            report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        write_report(out / "registration_pipeline_report.json", report)


# 批次执行与地图导出
def main(config=None):
    config = config or load_config("Register")
    settings = config.section("RunRegistration")
    jobs = settings["jobs"]
    layout = config.layout
    active_root, _work_dir, result_dir = layout.active_root, layout.work_dir, layout.result_dir
    registration_params = config.parameters("RegisterPointClouds")
    fusion_params = config.parameters("FusePointClouds")
    checker = config.section("CheckPointCloudPipeline")

    def initial_for(job):
        return (
            job["initial_transform"]
            if job["initial_transform"] is not None
            else settings["initial_transform"]
        )

    if not jobs:
        raise ValueError("jobs 不能为空")
    if len(jobs) > 1 and not settings["fuse_after_registration"]:
        raise ValueError("连续融合任务需要 fuse_after_registration=true；仅配准请只保留一项 JOB")
    manifest_path = active_root / "manifest.json"
    if manifest_path.exists() or result_dir.exists():
        raise FileExistsError(
            "本批次已有运行记录或结果，请在 config/Register.yaml 中换批次或试验名称"
        )
    # 运行前检查全部源云，避免执行到后半段才发现输入缺失。
    for index, job in enumerate(jobs):
        if index == 0 and job["target"] == "previous":
            raise ValueError("首轮 target 必须是实际参考点云路径")
        required = [job["source"]] + ([job["target"]] if index == 0 else [])
        initial = initial_for(job)
        if isinstance(initial, (str, Path)):
            required.append(initial)
        for item in required:
            if not Path(item).is_file():
                raise FileNotFoundError(item)
            if Path(item).suffix.lower() == ".pcd" and not sidecar_path(item).is_file():
                raise FileNotFoundError(sidecar_path(item))
        if Path(job["output_dir"]).exists():
            raise FileExistsError(job["output_dir"])
    module_dirs = []
    for job in jobs:
        scoped = config.with_context(JOB_DIR=job["output_dir"])
        module_dirs.append(scoped.section("RegisterPointClouds")["output_dir"])
        if settings["fuse_after_registration"]:
            module_dirs.append(scoped.section("FusePointClouds")["output_dir"])
    if len(module_dirs) != len(set(module_dirs)):
        raise ValueError("逐轮模块的输出目录重复，请使用 {JOB_DIR} 区分任务")
    for directory in module_dirs:
        if directory.exists():
            raise FileExistsError(directory)
    if settings["export_map"]:
        for name in ("PrepareGridInput", "PcdtoGridMap"):
            directory = config.section(name)["output_dir"]
            if directory.exists():
                raise FileExistsError(directory)
        script = Path(__file__).resolve().parent / "FormatChange/03PcdtoGridMap.py"
        if not script.is_file():
            raise FileNotFoundError(script)
    outputs = [Path(job["output_dir"]).resolve() for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("各 JOB 的 output 必须不同")
    start = time.monotonic()
    manifest = {
        "status": "running",
        "experiment": layout.experiment_name or None,
        "jobs": jobs,
        "completed_steps": [],
        "result": None,
        "parameters": {
            "registration": registration_params,
            "fusion": fusion_params,
            "fuse": settings["fuse_after_registration"],
            "export_map": settings["export_map"],
        },
        "initial_transform": settings["initial_transform"],
    }
    manifest.update(config.provenance())
    write_report(manifest_path, manifest)
    results, validations = [], []
    try:
        for index, job in enumerate(jobs):
            source, target = job["source"], job["target"]
            if target == "previous":
                target = Path(results[-1]["output_pcd"])
            step_config = config.with_context(JOB_DIR=job["output_dir"])
            registration_module = step_config.section("RegisterPointClouds")
            fusion_module = step_config.section("FusePointClouds")
            if index and Path(target).resolve() != Path(results[-1]["output_pcd"]).resolve():
                raise ValueError("后一轮 target 必须是前一轮实际返回的融合结果")
            print(f"开始配准：{source} → {target}", flush=True)
            result = run_registration(
                source,
                target,
                job["output_dir"],
                initial_transform=initial_for(job),
                registration_params={**registration_params, **job.get("parameters", {})},
                fuse_after_registration=settings["fuse_after_registration"],
                fusion_params={**fusion_params, **job.get("fusion_parameters", {})},
                module_output_dirs={
                    "registration": registration_module["output_dir"],
                    "fusion": fusion_module["output_dir"],
                },
            )
            validations.append(
                check_pipeline(
                    job["output_dir"], checker["validation_name"], checker["transform_atol"]
                )
            )
            results.append(result)
            manifest["completed_steps"].append(str(job["output_dir"]))
            write_report(manifest_path, manifest)
            gc.collect()
        cloud_name = "fused.pcd" if settings["fuse_after_registration"] else "aligned.pcd"
        final_cloud = publish_cloud(results[-1]["output_pcd"], result_dir / "cloud" / cloud_name)
        transform_dir = result_dir / "transforms"
        transform_dir.mkdir(parents=True, exist_ok=False)
        for index, result in enumerate(results, 1):
            # 每轮处理后 source -> target 的矩阵；不是原始 PCD 的累计变换。
            transform = np.load(result["transform_file"], allow_pickle=False)
            np.save(
                transform_dir / f"T_step_{index:02d}_source_to_target.npy",
                transform,
                allow_pickle=False,
            )
            np.savetxt(transform_dir / f"T_step_{index:02d}_source_to_target.txt", transform)
        summary = {
            "status": "running",
            "registrations": manifest["completed_steps"],
            "final_cloud": final_cloud,
            "output_sha256": sha256(final_cloud),
            "output_sidecar_sha256": sha256(sidecar_path(final_cloud)),
            "transforms": str(transform_dir),
            "map_dir": None,
        }
        if settings["export_map"]:
            prepare_grid_input = import_module("Pipeline.04PrepareGridInput").prepare_grid_input
            grid_settings = config.section("PrepareGridInput")
            projected = prepare_grid_input(
                final_cloud, grid_settings["output_dir"], **grid_settings["parameters"]
            )
            script = Path(__file__).resolve().parent / "FormatChange/03PcdtoGridMap.py"
            spec = importlib.util.spec_from_file_location("pcd_to_grid", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            map_settings = config.section("PcdtoGridMap")
            map_dir = map_settings["output_dir"]
            module.main(input_pcd=projected["output_pcd"], output_dir=map_dir, config=config)
            summary.update(
                grid_input=projected,
                map_dir=str(map_dir),
                unknown_as_free=map_settings["parameters"]["unknown_as_free"],
                grid_script_sha256=sha256(script),
                map_sha256={
                    f"{map_settings['parameters']['map_name']}.{extension}": sha256(
                        map_dir / f"{map_settings['parameters']['map_name']}.{extension}"
                    )
                    for extension in ("png", "pgm", "yaml")
                },
            )
        if sha256(final_cloud) != results[-1]["output_sha256"] or sha256(
            sidecar_path(final_cloud)
        ) != sha256(sidecar_path(results[-1]["output_pcd"])):
            raise RuntimeError("发布的最终点云或旁文件与末轮结果不一致")
        summary["status"] = "complete"
        write_report(result_dir / "result_report.json", summary)
        write_report(
            result_dir / "validation.json",
            {
                "accepted": True,
                "steps": validations,
                "published_cloud_matches_stage": sha256(final_cloud)
                == results[-1]["output_sha256"],
                "absolute_accuracy_validated": False,
                "scope": "每轮来源、配准矩阵、融合回滚；地图仅记录生成文件哈希",
            },
        )
        manifest.update(
            status="complete",
            result=str(result_dir),
            final_cloud=final_cloud,
            result_report=str(result_dir / "result_report.json"),
        )
        print(f"批次完成，结果目录：{result_dir}", flush=True)
        return results
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        for job in jobs:
            report_path = Path(job["output_dir"]) / "registration_pipeline_report.json"
            if report_path.is_file():
                if json.loads(report_path.read_text(encoding="utf-8")).get("status") == "rejected":
                    manifest["status"] = "rejected"
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - start
        write_report(manifest_path, manifest)


if __name__ == "__main__":
    main(cli_config("Register"))
