"""独立重算原始录制的一小段，与指定 LIO 结果逐帧比较；不修改参考结果。"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Common.PointCloudIO import sha256
from RawData.prepare import build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reference-lio", type=Path, required=True)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.duration <= 0:
        raise ValueError("复核时长必须为正")
    if a.output.exists():
        raise FileExistsError(a.output)
    binary = build(2)
    command = [
        str(binary),
        "--data-dir",
        str(a.source / "data"),
        "--output",
        str(a.output),
        "--duration",
        str(a.duration),
        "--frame-period",
        "0.5",
    ]
    subprocess.run(command, check=True, env={**os.environ, "OMP_NUM_THREADS": "4"})
    frames = list(csv.DictReader((a.output / "frames.csv").open()))
    checks = []
    for row in frames:
        current = a.output / row["file"]
        reference = a.reference_lio / row["file"]
        data = np.fromfile(current, dtype="<f4").reshape(-1, 4)
        old = np.fromfile(reference, dtype="<f4").reshape(-1, 4)
        if data.shape != old.shape:
            raise AssertionError(f"帧点数不同：{current.name}")
        exact = sha256(current) == sha256(reference)
        checks.append(
            {
                "file": row["file"],
                "points": len(data),
                "bitwise_exact": exact,
                "max_absolute_difference": float(np.max(np.abs(data - old))) if data.size else 0.0,
            }
        )
    result = {
        "command": command,
        "frames": len(checks),
        "all_frames_bitwise_exact": all(r["bitwise_exact"] for r in checks),
        "executable_sha256": sha256(binary),
        "checks": checks,
    }
    (a.output / "replay_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False, indent=2)
    )
    if not result["all_frames_bitwise_exact"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
