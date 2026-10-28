"""比较已生成的两张地图；只输出报告和对照图，不参与地图生成。"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt, label


def statistics(mask, reference, resolution, tolerance):
    ids, n = label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(ids.ravel())[1:]
    missing = int(
        (reference & (distance_transform_edt(~mask) > tolerance / resolution + 1e-12)).sum()
    )
    extra = int(
        (mask & (distance_transform_edt(~reference) > tolerance / resolution + 1e-12)).sum()
    )
    return {
        "occupied_cells": int(mask.sum()),
        "components": int(n),
        "small_components_le4": int((sizes <= 4).sum()),
        "small_component_cells_le4": int(sizes[sizes <= 4].sum()),
        "reference_cells_without_match_15cm": missing,
        "extra_cells_farther_than_reference_15cm": extra,
        "reference_coverage_15cm": 1 - missing / max(1, int(reference.sum())),
        "candidate_supported_by_reference_15cm": 1 - extra / max(1, int(mask.sum())),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reference-origin", type=float, nargs=2, required=True)
    p.add_argument("--reference-resolution", type=float, default=0.05)
    p.add_argument("--tolerance", type=float, default=0.15)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--baseline-mask",
        type=Path,
        default=None,
        help="可选：以历史共同坐标保存的 comparison_masks.npz，仅用于画调整前一栏",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    ref = np.array(Image.open(a.reference)) == 0
    cfg = yaml.safe_load(a.map.with_suffix(".yaml").read_text())
    image = np.array(Image.open(a.map))
    rows, cols = np.where(np.flipud(image) == 0)
    xy = (
        np.c_[cols, rows] * cfg["resolution"]
        + np.array(cfg["origin"][:2])
        + cfg["resolution"] * 0.5
    )
    ij = np.floor((xy - np.array(a.reference_origin)) / a.reference_resolution).astype(np.int64)
    valid = (ij >= 0).all(1) & (ij[:, 0] < ref.shape[1]) & (ij[:, 1] < ref.shape[0])
    mask = np.zeros(ref.shape, bool)
    mask[ref.shape[0] - 1 - ij[valid, 1], ij[valid, 0]] = True
    metrics = {
        "reference": statistics(ref, ref, a.reference_resolution, a.tolerance),
        "current": statistics(mask, ref, a.reference_resolution, a.tolerance),
        "current_map": str(a.map),
        "reference_map": str(a.reference),
        "reference_origin": a.reference_origin,
        "resolution": a.reference_resolution,
        "tolerance_m": a.tolerance,
        "reference_is_evaluation_target_not_independent_truth": True,
        "map_has_not_been_changed_by_evaluation": True,
    }
    panels = [(ref, "你的最新地图")]
    if a.baseline_mask:
        with np.load(a.baseline_mask) as z:
            previous = np.rot90(np.flipud(z["previous"]))
        if previous.shape != ref.shape:
            raise ValueError("调整前对照坐标尺寸不符")
        metrics["previous"] = statistics(previous, ref, a.reference_resolution, a.tolerance)
        panels.append((previous, "调整前"))
    panels.append((mask, "原始数据全部重算后"))
    (a.output / "comparison.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(a.output / "comparison_masks.npz", reference=ref, current=mask)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 7), constrained_layout=True)
    for ax, (data, title) in zip(axes, panels):
        ax.imshow(~data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
    fig.savefig(a.output / "调整前后对照.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
