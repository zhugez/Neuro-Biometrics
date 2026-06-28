"""Render charts for the 16:4 open-set evaluation.

Reads ONLY the V6 open-set result JSON(s):
- experiments/v6_openset/output_v6_openset_baselines.json (prior-work, no denoiser)
- experiments/v6_openset/output_v6_openset_bimodal.json   (flagship V4 bimodal)

Outputs (docs/ssrc2026/figures/):
- openset_auroc_compare.png : AUROC by model x noise type (primary open-set metric)
- openset_eer_compare.png   : open-set EER by model x noise type

Flagship and prior-work models are merged into one grouped-bar chart for a
direct "ours vs prior-work under open-set" comparison. This script never reads
or overwrites the standard V4/V5 outputs, and never touches
scripts/ssrc_make_charts.py. If a result JSON is missing (the GPU runs have not
been done yet), it prints a hint and skips that source.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "ssrc2026" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NOISE_KEYS = ["gaussian", "powerline", "emg"]
NOISE_LABELS = ["Gaussian", "Power-line", "EMG"]

SOURCES = [
    ("prior-work", ROOT / "experiments/v6_openset/output_v6_openset_baselines.json"),
    ("ours", ROOT / "experiments/v6_openset/output_v6_openset_bimodal.json"),
]

PALETTE = ["#1f4e79", "#c8553d", "#588157", "#9aa3af", "#4b6584", "#b08968"]


def _load(path: Path):
    if not path.exists():
        print(f"[skip] missing {path} (run the --openset experiment first)")
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("split_mode") != "openset_16_4":
        print(f"[warn] {path} is not an openset_16_4 file (split_mode="
              f"{data.get('split_mode')!r}); skipping")
        return None
    return data


def _series(data, stat_key):
    """Return [(model_name, [val_per_noise...]), ...] for one stat key."""
    models = []
    for r in data["results"]:
        if r["model_name"] not in models:
            models.append(r["model_name"])
    out = []
    for mdl in models:
        by_noise = {r["noise_type"]: r["stats"].get(stat_key, 0.0)
                    for r in data["results"] if r["model_name"] == mdl}
        out.append((mdl, [by_noise.get(k, 0.0) for k in NOISE_KEYS]))
    return out


def grouped_bar(out_path, series, title, ylabel, ylim=(0.0, 1.0),
                baseline=None, baseline_label=None):
    x = np.arange(len(NOISE_LABELS))
    n = max(1, len(series))
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=300)
    for i, (label, vals) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=PALETTE[i % len(PALETTE)], edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    if baseline is not None:
        ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.2,
                   label=baseline_label or f"random = {baseline:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(NOISE_LABELS, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _merged_series(datasets, stat_key, tag_prior=True):
    """Combine models across all loaded sources into one series list."""
    series = []
    for tag, data in datasets:
        for mdl, vals in _series(data, stat_key):
            label = f"{mdl} ({tag})" if tag_prior else mdl
            series.append((label, vals))
    return series


def main():
    datasets = [(name, _load(path)) for name, path in SOURCES]
    datasets = [(name, d) for name, d in datasets if d is not None]
    if not datasets:
        print("No V6 open-set result JSON found. Run e.g.:\n"
              "  python -m experiments.v6_openset.main --seeds 3")
        return

    grouped_bar(
        FIG / "openset_auroc_compare.png",
        _merged_series(datasets, "auroc_mean"),
        "16:4 Open-Set AUROC: ours vs prior-work by noise type",
        ylabel="AUROC",
        ylim=(0.0, 1.0),
        baseline=0.5, baseline_label="chance (AUROC = 0.5)",
    )

    grouped_bar(
        FIG / "openset_eer_compare.png",
        _merged_series(datasets, "open_set_eer_mean"),
        "16:4 Open-Set EER: ours vs prior-work by noise type",
        ylabel="Open-set EER (lower is better)",
        ylim=(0.0, 0.5),
    )


if __name__ == "__main__":
    main()
