"""合成数据回归检查；不读取真实录制，也不向项目 run 写入产物。

运行：python -m unittest discover -s tests -v
"""

import ast
import copy
import csv
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from unittest import mock

import laspy
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Common.CheckPointCloudPipeline import _check_grid_input, _check_subset, _check_temporal
from Common.Configuration import load_config
from Common.PointCloudIO import Cloud, read_cloud, sha256, write_cloud


class WorkspaceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="pcd_workspace_check_")
        cls.base = Path(cls.temporary.name)
        cls.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        x, y = np.meshgrid(np.arange(0, 6, 0.06), np.arange(0, 5, 0.06))
        parts = [np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]]
        for axis, position, span in [(0, 0, 5), (1, 0, 6), (0, 6, 4)]:
            a, z = np.meshgrid(np.arange(0, span, 0.06), np.arange(0.12, 2.4, 0.06))
            points = np.c_[a.ravel(), np.full(a.size, position), z.ravel()]
            if axis == 0:
                points[:, [0, 1]] = points[:, [1, 0]]
            parts.append(points)
        cls.room = np.concatenate(parts)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def document(self, kind, batch):
        data = copy.deepcopy(load_config(kind).raw)
        data["ROOT_DIR"] = str(self.base)
        data["run"]["batch_name"] = batch
        if kind == "Pipeline":
            if "PrepareTemporalFrames" in data:
                data["PrepareTemporalFrames"]["enabled"] = False
            data["TemporalFilter"]["parameters"].update(
                protected_bounds_transform_file=None, protected_bounds=[]
            )
        return data

    def save(self, name, data):
        path = self.base / (name + ".yaml")
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def cli(self, script, path=None, succeeds=True):
        command = [sys.executable, str(ROOT / script)]
        command += ["--config", str(path)] if path else ["--help"]
        result = subprocess.run(
            command, cwd=self.base, env=self.env, capture_output=True, text=True
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def frames(self, name, world_points, endpoints_for=None):
        folder = self.base / name
        folder.mkdir()
        fields = [
            "file",
            "stamp",
            "accepted_geometry",
            "accepted_for_free_space",
            "holdout_plane_p95_m",
            "holdout_distance_p95_m",
        ]
        fields += [f"m{i}{j}" for i in range(4) for j in range(4)]
        path = folder / "frames.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(3):
                origin = np.array([index * 0.7, 0.0, 0.0])
                local = world_points - origin
                if endpoints_for is not None:
                    delta = np.asarray(endpoints_for) - origin
                    endpoint = delta * (1 + 2.0 / np.linalg.norm(delta))
                    local = np.vstack([local, endpoint])
                filename = folder / f"frame_{index}.pcd"
                write_cloud(filename, Cloud(local))
                matrix = np.eye(4)
                matrix[:3, 3] = origin
                row = dict(
                    file=filename.name,
                    stamp=index * 3.0,
                    accepted_geometry="true",
                    accepted_for_free_space="true",
                    holdout_plane_p95_m=0.01,
                    holdout_distance_p95_m=0.01,
                )
                row.update({f"m{i}{j}": matrix[i, j] for i in range(4) for j in range(4)})
                writer.writerow(row)
        return path

    def test_01_syntax_imports_cli_and_readme(self):
        import Common.Configuration as configuration

        for path in ROOT.rglob("*.py"):
            if {"__pycache__", "tests", "build", "run"}.intersection(path.relative_to(ROOT).parts):
                continue
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            with mock.patch.object(
                configuration, "load_config", side_effect=AssertionError("eager config read")
            ):
                loaded = importlib.import_module(module)
                importlib.reload(loaded)
            # 恢复导入探针期间绑定的 load_config，避免污染后续入口测试。
            importlib.reload(loaded)
            if 'if __name__ == "__main__"' in source:
                self.cli(str(path.relative_to(ROOT)))
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        import re

        for relative in re.findall(
            r"(?:Common|Pipeline|Register|FormatChange)/[A-Za-z0-9_]+\.py", readme
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
        for path in (ROOT / "config").glob("*.yaml"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if re.match(
                    r"\s*(?:-\s+)?[A-Za-z_][A-Za-z_0-9]*\s*:", line
                ) and not line.rstrip().endswith(":"):
                    self.assertIn(" # ", line)
        for path in (ROOT / ".vscode").glob("*.json"):
            json.loads(path.read_text())

    def test_02_configuration_and_path_reuse(self):
        data = self.document("Pipeline", "configuration")
        data["ExtractGround"]["output_dir"] = "{RunPointCloudPipeline.output_dir}/changed_ground"
        path = self.save("configuration", data)
        config = load_config("Pipeline", path)
        self.assertEqual(
            config.section("CropPointCloud")["input_pcd"],
            config.section("ExtractGround")["output_dir"] / "leveled.pcd",
        )
        cases = [
            lambda d: d["PreprocessPointCloud"]["parameters"].update(statistical="false"),
            lambda d: d["ExtractGround"]["parameters"].update(typo=1),
            lambda d: d["TemporalFilter"]["parameters"].update(independent_views=0),
            lambda d: d["TemporalFilter"]["parameters"].update(frame_range=[2, 1]),
            lambda d: d["PcdtoGridMap"]["parameters"].update(resolution=0),
            lambda d: d["ExtractGround"].update(output_dir="{ExtractGround.output_dir}"),
        ]
        for index, mutate in enumerate(cases):
            broken = copy.deepcopy(data)
            mutate(broken)
            with self.assertRaises(ValueError):
                load_config("Pipeline", self.save("bad_" + str(index), broken))
        duplicate = self.base / "duplicate.yaml"
        duplicate.write_text('ROOT_DIR: ".."\nROOT_DIR: ".."\n')
        with self.assertRaisesRegex(ValueError, "重复键"):
            load_config("Pipeline", duplicate)

    def test_03_pointcloud_precision_and_binding(self):
        points = np.array([[0.1234567890123, 2, 3], [2.2, -4, 5]])
        cloud = Cloud(
            points,
            np.array([[1, 2, 3], [250, 251, 252]], dtype=np.uint8),
            extras={"tag": np.array([2**60 + 1, 2**60 + 2], dtype=np.int64)},
        )
        path = self.base / "precision.pcd"
        write_cloud(path, cloud)
        output = read_cloud(path)
        np.testing.assert_array_equal(output.points, points)
        np.testing.assert_array_equal(output.extras["tag"], cloud.extras["tag"])
        np.testing.assert_array_equal(output.colors, cloud.colors)
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            read_cloud(path)

    def test_04_format_and_standalone_pipeline(self):
        data = self.document("Pipeline", "standalone")
        data["run"]["scan_name"] = "scan_a"
        source = self.base / "room.las"
        las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
        las.header.scales = [0.001] * 3
        las.x, las.y, las.z = self.room.T
        las.red = np.full(len(self.room), 200, dtype=np.uint16)
        las.green = np.full(len(self.room), 100, dtype=np.uint16)
        las.blue = np.full(len(self.room), 50, dtype=np.uint16)
        las.write(source)
        data["LastoPcd"]["input_las"] = str(source)
        data["PreprocessPointCloud"]["parameters"]["statistical"] = False
        data["ExtractGround"]["parameters"].update(sample_stride=4, ransac_iterations=40)
        data["PcdtoGridMap"]["parameters"].update(map_name="standalone_map", unknown_as_free=False)
        path = self.save("standalone", data)
        modules = [
            "FormatChange/01LastoPcd.py",
            "Pipeline/01PreprocessPointCloud.py",
            "Pipeline/02ExtractGround.py",
            "Pipeline/03CropPointCloud.py",
            "Pipeline/04PrepareGridInput.py",
            "FormatChange/03PcdtoGridMap.py",
            "FormatChange/02PcdtoPly.py",
        ]
        for module in modules:
            self.cli(module, path)
        config = load_config("Pipeline", path)
        converted = config.section("LastoPcd")["output_pcd"]
        digest = sha256(converted)
        self.cli(modules[0], path, succeeds=False)
        self.assertEqual(sha256(converted), digest)
        self.assertTrue((config.layout.result_dir / "map/standalone_map.yaml").is_file())

    def test_05_streaming_and_projection_modes(self):
        preprocess = importlib.import_module("Pipeline.01PreprocessPointCloud").preprocess
        points = np.array([[0.01, 0.01, 0], [0.02, 0.02, 0], [1.01, 0, 0], [1.02, 0, 0], [2, 0, 0]])
        source = self.base / "binary.pcd"
        write_cloud(source, Cloud(points))
        Path(str(source) + ".npz").unlink()
        with redirect_stdout(io.StringIO()):
            report = preprocess(
                source,
                self.base / "streamed",
                voxel_size=0.1,
                statistical=False,
                save_removed=False,
                streaming=True,
            )
        _check_subset(report, "preprocess")
        self.assertEqual(int(read_cloud(report["output_pcd"]).raw_count.sum()), len(points))
        grid = importlib.import_module("Pipeline.04PrepareGridInput").prepare_grid_input
        cloud = Cloud(
            np.array([[0.0, 0, 2], [1, 0, 3]]),
            extras={
                "ground_mask": np.array([True, False]),
                "obstacle_mask": np.array([False, True]),
                "ground_supported": np.ones(2, dtype=bool),
                "ground_height": np.full(2, 2.0),
            },
        )
        source = self.base / "classified.pcd"
        write_cloud(source, cloud)
        for relative in [True, False]:
            result = grid(
                source, self.base / ("projection_" + str(relative)), normalize_local_ground=relative
            )
            _check_grid_input(result)
        cloud.extras["ground_mask"] = cloud.extras["ground_mask"][:, None]
        malformed = self.base / "bad_classification.pcd"
        write_cloud(malformed, cloud)
        with self.assertRaisesRegex(ValueError, "长度 N"):
            grid(malformed, self.base / "must_not_exist")
        self.assertFalse((self.base / "must_not_exist").exists())

    def test_06_temporal_review_apply_and_guards(self):
        temporal = partial(
            importlib.import_module("Pipeline.05TemporalFilter").temporal_filter,
            protected_bounds_transform_file=None,
            protected_bounds=[],
            sparse_filter=False,
        )
        points = np.array([[2.0, 0, 0], [4, 1, 1], [3, 0, 1]])
        source = self.base / "temporal_source.pcd"
        write_cloud(source, Cloud(points))
        frames = self.frames("temporal_frames", points[:2], endpoints_for=points[2])
        review_data = self.document("Pipeline", "temporal_standalone")
        review_data["TemporalFilter"].update(
            input_pcd=str(source), frames_csv=str(frames), output_dir=str(self.base / "review_cli")
        )
        review_data["TemporalFilter"]["parameters"].update(protect_horizontal=False, mode="review")
        self.cli("Pipeline/05TemporalFilter.py", self.save("temporal_standalone", review_data))
        review = json.loads((self.base / "review_cli/report.json").read_text())
        self.assertEqual(review["removed_points"], 0)
        self.assertEqual(review["proposed_points"], 1)
        _check_temporal(review)
        with redirect_stdout(io.StringIO()):
            applied = temporal(
                source, frames, self.base / "applied", mode="apply", protect_horizontal=False
            )
            blocked = temporal(
                source,
                frames,
                self.base / "blocked",
                mode="apply",
                protect_horizontal=False,
                protected_bounds=[[2, 2.2, -0.1, 0.1, 0.2, 1.0]],
            )
        self.assertEqual(applied["removed_points"], 1)
        self.assertEqual(blocked["removed_points"], 0)
        self.assertEqual(applied["protected_removed"], 0)
        _check_temporal(applied)
        _check_temporal(blocked)
        np.testing.assert_array_equal(read_cloud(applied["output_pcd"]).points, points[:2])

    def test_06b_temporal_native_bin_and_format_validation(self):
        temporal = partial(
            importlib.import_module("Pipeline.05TemporalFilter").temporal_filter,
            protected_bounds_transform_file=None,
            protected_bounds=[],
            sparse_filter=False,
        )
        points = np.array([[2.0, 0, 0], [4, 1, 1], [3, 0, 1]])
        source = self.base / "temporal_bin_source.pcd"
        write_cloud(source, Cloud(points))
        path = self.frames("temporal_bin_frames", points[:2], endpoints_for=points[2])
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = list(reader.fieldnames), list(reader)
        for row in rows:
            original = path.parent / row["file"]
            cloud = read_cloud(original)
            binary = original.with_suffix(".bin")
            np.c_[cloud.points, np.ones(len(cloud))].astype("<f4").tofile(binary)
            row.update(file=binary.name, point_format="xyzi_float32_le")
        fields.append("point_format")

        def save_rows():
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        save_rows()
        with redirect_stdout(io.StringIO()):
            report = temporal(
                source, path, self.base / "bin_applied", mode="apply", protect_horizontal=False
            )
        self.assertEqual(report["removed_points"], 1)
        _check_temporal(report)
        np.testing.assert_array_equal(read_cloud(report["output_pcd"]).points, points[:2])
        rows[0]["point_format"] = ""
        save_rows()
        with self.assertRaisesRegex(ValueError, "point_format"):
            temporal(
                source, path, self.base / "bin_unspecified", mode="apply", protect_horizontal=False
            )
        rows[0]["point_format"] = "xyzi_float32_le"
        save_rows()
        (path.parent / rows[0]["file"]).write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "N×4"):
            temporal(
                source, path, self.base / "bin_invalid", mode="apply", protect_horizontal=False
            )

    def test_06c_temporal_bounded_query_matches_full_nearest(self):
        query = importlib.import_module("Pipeline.05TemporalFilter")._ray_query
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(20261027)
        directions = rng.normal(size=(2000, 3))
        directions[:, 2] *= 0.2
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        length = rng.uniform(0.3, 30, len(directions))
        rays = directions * length[:, None]
        unit = np.r_[directions[:500], rng.normal(size=(2000, 3))]
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        ranges = rng.uniform(0.5, 20, len(unit))
        ranges[:12] = np.r_[
            np.nextafter([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], np.inf), [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        ]
        delta = unit * ranges[:, None]
        params = dict(ray_distance=0.04, endpoint_margin=0.3)
        distance, nearest = cKDTree(rays / np.linalg.norm(rays, axis=1)[:, None]).query(
            delta / ranges[:, None]
        )
        expected = (distance * ranges < params["ray_distance"]) & (
            np.linalg.norm(rays, axis=1)[nearest] > ranges + params["endpoint_margin"]
        )
        actual, indices = query(delta, ranges, rays, params)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(indices[actual], nearest[actual])

    def test_06d_sparse_temporal_without_free_permission(self):
        temporal = partial(
            importlib.import_module("Pipeline.05TemporalFilter").temporal_filter,
            protected_bounds_transform_file=None,
            protected_bounds=[],
            sparse_filter=False,
        )
        floor = np.c_[np.linspace(0, 1, 20), np.zeros(20), np.zeros(20)]
        points = np.r_[floor, [[5.0, 5, 1], [8, 0, 1], [10, 3, 1], [12, 2, 1]]]
        raw = np.ones(len(points), dtype=np.uint32)
        raw[-2] = 10
        source = self.base / "sparse_temporal_source.pcd"
        write_cloud(source, Cloud(points, raw_count=raw))
        frames = self.frames("sparse_temporal_frames", np.array([[8.0, 0, 1]]))
        with frames.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = list(reader.fieldnames), list(reader)
        for row in rows:
            row["accepted_for_free_space"] = "false"
        with frames.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with redirect_stdout(io.StringIO()):
            report = temporal(
                source,
                frames,
                self.base / "sparse_temporal",
                mode="apply",
                protect_horizontal=False,
                sparse_filter=True,
                protected_bounds=[[11.8, 12.2, 1.8, 2.2, 0.8, 1.2]],
            )
        self.assertEqual(report["strict_free_frames"], 0)
        self.assertEqual(report["sparse_proposed_points"], 1)
        self.assertEqual(report["ray_proposed_points"], 0)
        self.assertEqual(report["removed_points"], 1)
        _check_temporal(report)
        np.testing.assert_array_equal(
            read_cloud(report["output_pcd"]).points, np.delete(points, 20, axis=0)
        )

    def test_06e_global_projection_keeps_unsupported_wall(self):
        grid = importlib.import_module("Pipeline.04PrepareGridInput").prepare_grid_input
        cloud = Cloud(
            np.array([[0.0, 0, 0], [1, 0, 1], [2, 0, 1.5], [10, 0, 1]]),
            extras={
                "ground_mask": np.array([True, False, False, False]),
                "obstacle_mask": np.array([False, True, False, False]),
                "ground_supported": np.array([True, True, False, False]),
                "ground_height": np.array([0.0, 0, np.nan, np.nan]),
            },
        )
        source = self.base / "global_projection_source.pcd"
        write_cloud(source, cloud)
        matrix = np.array([[0.0, -1, 0, 3], [1, 0, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]])
        report = grid(
            source,
            self.base / "global_projection",
            selection_mode="global_height",
            normalize_local_ground=False,
            coordinate_transform=matrix.tolist(),
            bounds=[[2.0, -0.1, -1.0], [4.0, 3.0, 3.0]],
        )
        self.assertEqual(report["output_points"], 3)
        self.assertEqual(report["unsupported_points_retained"], 1)
        _check_grid_input(report)
        np.testing.assert_array_equal(read_cloud(source).points, cloud.points)
        with self.assertRaisesRegex(ValueError, "不得同时"):
            grid(
                source,
                self.base / "bad_global_projection",
                selection_mode="global_height",
                normalize_local_ground=True,
            )

    def test_06f_frame_left_transform_and_provenance(self):
        temporal = partial(
            importlib.import_module("Pipeline.05TemporalFilter").temporal_filter,
            protected_bounds_transform_file=None,
            protected_bounds=[],
            sparse_filter=False,
        )
        points = np.array([[2.0, 0, 0], [4, 1, 1]])
        matrix = np.array([[0.0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 0.2], [0, 0, 0, 1]])
        left = self.base / "left_transform.npy"
        np.save(left, matrix, allow_pickle=False)
        frames = self.frames("left_frames", points)
        with frames.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = list(reader.fieldnames), list(reader)
        fields.append("left_transform_file")
        for row in rows:
            row.update(left_transform_file=str(left), accepted_for_free_space="false")
        with frames.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        source = self.base / "left_source.pcd"
        write_cloud(source, Cloud(points @ matrix[:3, :3].T + matrix[:3, 3]))
        with redirect_stdout(io.StringIO()):
            report = temporal(
                source, frames, self.base / "left_temporal", mode="apply", protect_horizontal=False
            )
        self.assertEqual(report["strong_multiview_points"], 2)
        self.assertEqual(report["input_hashes"][str(left)], sha256(left))
        _check_temporal(report)
        np.save(left, np.eye(4), allow_pickle=False)
        with self.assertRaises(AssertionError):
            _check_temporal(report)

    def test_07_runs_temporal_registration_and_rollback(self):
        outputs = []
        for index in range(2):
            source = self.base / f"run_raw_{index}.pcd"
            points = np.vstack([self.room, [3.0, 2.0, 1.0]]) if index == 0 else self.room.copy()
            write_cloud(source, Cloud(points, scan_id=np.full(len(points), index, dtype=np.int32)))
            if index == 1:
                import open3d as o3d

                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                self.assertTrue(o3d.io.write_point_cloud(str(source), cloud, write_ascii=True))
                Path(str(source) + ".npz").unlink()
            data = self.document("Pipeline", "run_pipeline")
            data["run"]["scan_name"] = f"scan_{index}"
            data["PreprocessPointCloud"]["input_pcd"] = str(source)
            data["PreprocessPointCloud"]["parameters"]["statistical"] = False
            data["ExtractGround"]["parameters"].update(sample_stride=4, ransac_iterations=40)
            data["RunPointCloudPipeline"].update(
                export_grid_input=True, enable_temporal_filter=index == 0
            )
            data["PreprocessPointCloud"]["output_dir"] = (
                "{RunPointCloudPipeline.output_dir}/custom_pre"
            )
            if index == 0:
                data["TemporalFilter"]["parameters"]["mode"] = "apply"
                data["TemporalFilter"]["frames_csv"] = str(
                    self.frames("run_frames", self.room, endpoints_for=[3.0, 2.0, 1.0])
                )
            else:
                data["RunPointCloudPipeline"]["auto_stream_threshold_bytes"] = 1
            path = self.save(f"run_pipeline_{index}", data)
            self.cli("RunPointCloudPipeline.py", path)
            self.cli("Common/CheckPointCloudPipeline.py", path)
            config = load_config("Pipeline", path)
            report = json.loads((config.layout.scan_dir / "pipeline_report.json").read_text())
            self.assertEqual("temporal" in report["stages"], index == 0)
            if index == 0:
                self.assertEqual(report["stages"]["temporal"]["removed_points"], 1)
            else:
                self.assertFalse(report["stages"]["preprocess"]["streaming"])
            outputs.append(str(config.layout.scan_dir / "processed.pcd"))
        data = self.document("Register", "run_registration")
        data["RegisterPointClouds"]["output_dir"] = "{JOB_DIR}/custom_registration"
        data["FusePointClouds"]["output_dir"] = "{JOB_DIR}/custom_fusion"
        data["PcdtoGridMap"]["parameters"].update(map_name="configured_map", unknown_as_free=False)
        data["RunRegistration"]["initial_transform"] = np.eye(4).tolist()
        jobs = data["RunRegistration"]["jobs"][:2]
        for index, job in enumerate(jobs):
            job.update(
                source=outputs[1 - index],
                target=outputs[0] if index == 0 else "previous",
                output_dir="{REGISTRATION_ROOT}/step_" + str(index),
                initial_transform=None,
            )
        data["RunRegistration"]["jobs"] = jobs
        path = self.save("run_registration", data)
        self.cli("RunRegistration.py", path)
        config = load_config("Register", path)
        summary = json.loads((config.layout.active_root / "manifest.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertTrue((config.layout.result_dir / "map/configured_map.pgm").is_file())
        # Same standalone settings, in a separate trial, plus non-map / rejected branches.
        standalone = copy.deepcopy(data)
        standalone["run"]["experiment_name"] = "manual"
        standalone["RegisterPointClouds"].update(
            source_pcd=outputs[1], target_pcd=outputs[0], initial_transform=np.eye(4).tolist()
        )
        path = self.save("manual_register", standalone)
        for module in [
            "Register/01RegisterPointClouds.py",
            "Register/02FusePointClouds.py",
            "Pipeline/04PrepareGridInput.py",
            "FormatChange/03PcdtoGridMap.py",
            "FormatChange/02PcdtoPly.py",
        ]:
            self.cli(module, path)
        for name, rejected in [("only_alignment", False), ("rejected", True)]:
            branch = copy.deepcopy(data)
            branch["run"]["experiment_name"] = name
            branch["RunRegistration"]["jobs"] = branch["RunRegistration"]["jobs"][:1]
            branch["RunRegistration"].update(export_map=False, fuse_after_registration=rejected)
            if rejected:
                branch["RegisterPointClouds"]["parameters"]["check_stability"] = False
            path = self.save(name, branch)
            self.cli("RunRegistration.py", path, succeeds=not rejected)
            config = load_config("Register", path)
            status = json.loads((config.layout.active_root / "manifest.json").read_text())["status"]
            self.assertEqual(status, "rejected" if rejected else "complete")
            if rejected:
                self.assertFalse(config.layout.result_dir.exists())
            else:
                self.assertTrue((config.layout.result_dir / "cloud/aligned.pcd.npz").is_file())

    def test_09_map_frame_export_preserves_pixel_world_coordinates(self):
        from PIL import Image

        from RunDataset import export_map_frame

        directory = self.base / "rotated_map"
        (directory / "map").mkdir(parents=True)
        pixels = np.array([[0, 254, 254], [254, 0, 254]], dtype=np.uint8)
        for ext in ["png", "pgm"]:
            Image.fromarray(pixels).save(directory / "map" / ("map." + ext))
        settings = dict(image="map.pgm", resolution=0.05, origin=[-11.0, -15.35, 0.0], negate=0)
        (directory / "map/map.yaml").write_text(yaml.safe_dump(settings))
        report = export_map_frame(directory, 1, [24.325, -3.975])
        output = np.array(Image.open(directory / "map_latest/map.pgm"))
        np.testing.assert_array_equal(output, np.rot90(pixels))
        self.assertEqual(report["size"], [2, 3])
        x, y, _ = report["origin"]
        oldrows, oldcols = np.where(np.flipud(pixels) == 0)
        oldxy = np.c_[oldcols, oldrows] * 0.05 + [-11.0, -15.35] + 0.025
        expected = np.c_[-oldxy[:, 1] + 24.325, oldxy[:, 0] - 3.975]
        newrows, newcols = np.where(np.flipud(output) == 0)
        actual = np.c_[newcols, newrows] * 0.05 + [x, y] + 0.025
        np.testing.assert_allclose(
            actual[np.argsort(actual[:, 0])], expected[np.argsort(expected[:, 0])], atol=1e-12
        )
        with self.assertRaises(FileExistsError):
            export_map_frame(directory, 1, [24.325, -3.975])

    def test_10_source_archive_keeps_project_metadata(self):
        from PIL import Image

        folder = self.base / "archive_run"
        (folder / "work/configs").mkdir(parents=True)
        (folder / "work/01_preprocess").mkdir()
        (folder / "result/map_latest").mkdir(parents=True)
        source = self.base / "archive_source.pcd"
        write_cloud(
            source,
            Cloud(
                np.array([[0.0, 0, 0], [1.0, 0, 1]]),
                extras={
                    "ground_mask": np.array([True, False]),
                    "obstacle_mask": np.array([False, True]),
                    "ground_supported": np.array([True, True]),
                    "ground_height": np.zeros(2),
                },
            ),
        )
        prepare = importlib.import_module("Pipeline.04PrepareGridInput").prepare_grid_input
        grid = prepare(
            source,
            folder / "work/03_grid_input",
            normalize_local_ground=True,
            selection_mode="local_ground",
            coordinate_transform=None,
            bounds=None,
        )
        documents = {
            "dataset_manifest.json": {
                "status": "complete",
                "input_hashes": {str(source): sha256(source)},
            },
            "manifest.json": {"status": "complete"},
            "result/validation.json": {"accepted": True},
            "result/result_report.json": {"grid_input": grid},
        }
        for name, document in documents.items():
            (folder / name).write_text(json.dumps(document), encoding="utf-8")
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(folder / "result/map_latest/map.pgm")
        destination = self.base / "archived_result"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "Tools/CollectResult.py"),
                "--run",
                str(folder),
                "--output",
                str(destination),
            ],
            cwd=self.base,
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ("README.md", "pyproject.toml", ".editorconfig", ".gitignore"):
            archived = destination / "复现代码" / name
            self.assertTrue(archived.is_file(), f"源码归档缺少 {name}")
            self.assertEqual(archived.read_bytes(), (ROOT / name).read_bytes())

    def test_11_single_dataset_exports_map_without_registration_config(self):
        import RunDataset as dataset
        import RunPointCloudPipeline as pipeline
        from RawData import prepare as raw_prepare

        recording = self.base / "single_recording"
        (recording / "data").mkdir(parents=True)
        (recording / "data/data_0").write_bytes(b"synthetic sensor fixture")
        source = recording / "colorized-realtime.las"
        las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
        las.x, las.y, las.z = self.room.T
        las.write(source)
        data = self.document("Pipeline", "single_dataset")
        data["RunDataset"]["register_config"] = str(self.base / "not_needed.yaml")
        data["RunDataset"]["scans"] = [
            {
                "name": "solo",
                "input_las": str(source),
                "allow_free_space": False,
                "protected_bounds": [],
            }
        ]
        # 留下模板原有 reference_scan 和矩阵；单份不应依赖它们。
        data["RunDataset"]["parameters"]["export_preview"] = True
        data["TemporalFilter"]["parameters"].update(mode="apply", sparse_filter=False)
        data["ExtractGround"]["parameters"].update(sample_stride=4, ransac_iterations=40)
        cfg = self.save("single_dataset", data)
        actual_run = subprocess.run

        def sensor_fixture(**settings):
            # 仅替代需要真实传感器二进制的离线 LIO；Python 处理与导图实际执行。
            output = Path(settings["output_dir"])
            output.parent.mkdir(parents=True, exist_ok=True)
            cloud = read_cloud(Path(settings["input_pcd"]).parent.parent / "02_ground/leveled.pcd")
            frames = self.frames(str(output), cloud.points)
            return {
                "status": "complete",
                "source_dir": str(recording),
                "input_hashes": {str(recording / "data/data_0"): sha256(recording / "data/data_0")},
                "uses_historical_outputs": False,
                "input_data_unchanged": True,
                "frames_csv": str(frames),
                "frames_csv_sha256": sha256(frames),
                "frames": 3,
                "accepted_geometry": 3,
                "accepted_for_free_space": 0,
            }

        def execute(command, **kwargs):
            if Path(command[1]).name != "RunPointCloudPipeline.py":
                return actual_run(command, **kwargs)
            with mock.patch.object(raw_prepare, "prepare", side_effect=sensor_fixture):
                with redirect_stdout(kwargs["stdout"]):
                    pipeline.main(load_config("Pipeline", command[-1]))
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(dataset.subprocess, "run", side_effect=execute):
            with redirect_stdout(io.StringIO()):
                try:
                    result = dataset.run(cfg)
                except (ValueError, KeyError, FileNotFoundError) as error:
                    self.fail(f"单份输入应直接导图，不依赖配准参数：{error}")
        folder = self.base / "run/single_dataset"
        self.assertEqual(result["status"], "complete")
        self.assertFalse((folder / "work/02_registration").exists())
        summary = json.loads((folder / "result/result_report.json").read_text())
        self.assertEqual(summary["registrations"], [])
        self.assertFalse(summary["registration_performed"])
        self.assertFalse(summary["fusion_performed"])
        self.assertEqual(Path(summary["final_cloud"]).name, "processed.pcd")
        source_cloud = folder / "work/01_preprocess/solo/processed.pcd"
        self.assertEqual(sha256(summary["final_cloud"]), sha256(source_cloud))
        self.assertEqual(
            sha256(str(summary["final_cloud"]) + ".npz"), sha256(str(source_cloud) + ".npz")
        )
        self.assertTrue((folder / "result/preview/processed.ply").is_file())
        for ext in ("png", "pgm", "yaml"):
            self.assertTrue((folder / "result/map" / ("map." + ext)).is_file())
        self.assertTrue(json.loads((folder / "result/validation.json").read_text())["accepted"])
        _check_grid_input(summary["grid_input"])
        with self.assertRaises(FileExistsError):
            dataset.run(cfg)

    def test_08_reject_lossy_attribute_promotion(self):
        harmonize = importlib.import_module("Register.02FusePointClouds")._harmonize
        clouds = [
            Cloud(np.array([[0.0, 0, 0]]), extras={"stamp": np.array([2**60 + 1], dtype=np.int64)}),
            Cloud(
                np.array([[1.0, 0, 0]]), extras={"stamp": np.array([2**60 + 2], dtype=np.uint64)}
            ),
        ]
        with self.assertRaisesRegex(ValueError, "丢失精度"):
            harmonize(clouds, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
