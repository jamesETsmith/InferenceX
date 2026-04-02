"""Plot MI350X vs MI355X benchmark comparisons.

Usage:
    python plot_comparison.py <mi350x_csv> <mi355x_csv> [-o output_dir]

Generates two plots:
  1. Mean E2E Latency vs Throughput/GPU
  2. Median Interactivity vs Throughput/GPU
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from adjustText import adjust_text
except ImportError:
    print("matplotlib, seaborn, and adjustText are required: pip install matplotlib seaborn adjustText", file=sys.stderr)
    sys.exit(1)

sns.set_theme(context="talk", style="whitegrid")

SERIES_STYLES = [
    {"color": "#e74c3c", "marker": "o"},   # red circle
    {"color": "#2980b9", "marker": "s"},   # blue square
    {"color": "#27ae60", "marker": "D"},   # green diamond
    {"color": "#8e44ad", "marker": "^"},   # purple triangle
]


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV, skipping comment lines (# prefix)."""
    rows = []
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append(row)
    return rows


def group_by_tp(rows: list[dict]) -> dict[int, list[dict]]:
    """Split rows into groups keyed by TP value, deduplicated and sorted by concurrency."""
    groups: dict[int, list[dict]] = {}
    for r in rows:
        tp = int(r["TP"])
        groups.setdefault(tp, []).append(r)
    for tp in groups:
        seen: set[int] = set()
        deduped = []
        for r in sorted(groups[tp], key=lambda r: r.get("Date", ""), reverse=True):
            conc = int(r["Concurrency"])
            if conc not in seen:
                seen.add(conc)
                deduped.append(r)
        deduped.sort(key=lambda r: int(r["Concurrency"]))
        groups[tp] = deduped
    return groups


def make_label(hw: str, tp: int, isl: str, osl: str) -> str:
    return f"{hw.upper()} TP={tp} ({isl}/{osl})"


def _build_series(
    *csv_groups: tuple[str, dict[int, list[dict]]],
) -> list[tuple[list[dict], dict, str]]:
    """Build a flat list of (data, style_dict, label) tuples from grouped CSV data."""
    series = []
    idx = 0
    for _hw_label, tp_groups in csv_groups:
        for tp in sorted(tp_groups):
            data = tp_groups[tp]
            style = SERIES_STYLES[idx % len(SERIES_STYLES)]
            label = make_label(data[0]["Hardware"], tp, data[0]["ISL"], data[0]["OSL"])
            series.append((data, style, label))
            idx += 1
    return series


ADJUST_TEXT_KW = dict(
    min_arrow_len=15,
    force_points=(2.0, 2.0),
    force_text=(1.5, 1.5),
    expand=(2.0, 2.0),
    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
)


def _pad_axes(ax: plt.Axes, pad_frac: float = 0.08) -> None:
    """Expand axis limits by a fraction so labels near the edges have room."""
    for getter, setter in [(ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim)]:
        lo, hi = getter()
        margin = (hi - lo) * pad_frac
        setter(lo - margin, hi + margin)


def plot_e2e_vs_throughput(series: list[tuple[list[dict], dict, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    texts = []
    for data, style, label in series:
        tput = [float(r["Throughput/GPU (tok/s)"]) for r in data]
        e2e = [float(r["Mean E2E Latency (ms)"]) for r in data]
        concs = [int(r["Concurrency"]) for r in data]

        ax.plot(e2e, tput, color=style["color"], marker=style["marker"],
                linewidth=2, markersize=9, label=label, zorder=3)
        for e, t, c in zip(e2e, tput, concs):
            texts.append(ax.text(e, t, f"c={c}", fontsize=11, color=style["color"],
                                 ha="center", va="bottom"))

    _pad_axes(ax)
    adjust_text(texts, ax=ax, **ADJUST_TEXT_KW)

    ax.set_xlabel("Mean E2E Latency (s)")
    ax.set_ylabel("Throughput / GPU (tok/s)")
    ax.set_title("Kimi-K2.5 FP4 — E2E Latency vs Throughput", fontweight="bold")
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def plot_interactivity_vs_throughput(series: list[tuple[list[dict], dict, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    texts = []
    for data, style, label in series:
        tput = [float(r["Throughput/GPU (tok/s)"]) for r in data]
        intvty = [float(r["Median Interactivity (tok/s/user)"]) for r in data]
        concs = [int(r["Concurrency"]) for r in data]

        ax.plot(intvty, tput, color=style["color"], marker=style["marker"],
                linewidth=2, markersize=9, label=label, zorder=3)
        for i, t, c in zip(intvty, tput, concs):
            texts.append(ax.text(i, t, f"c={c}", fontsize=11, color=style["color"],
                                 ha="center", va="bottom"))

    _pad_axes(ax)
    adjust_text(texts, ax=ax, **ADJUST_TEXT_KW)

    ax.set_xlabel("Median Interactivity (tok/s/user)")
    ax.set_ylabel("Throughput / GPU (tok/s)")
    ax.set_title("Kimi-K2.5 FP4 — Interactivity vs Throughput", fontweight="bold")
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MI350X vs MI355X benchmark comparison.")
    parser.add_argument("mi350x_csv", type=Path, help="Path to MI350X results CSV")
    parser.add_argument("mi355x_csv", type=Path, help="Path to MI355X results CSV")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Directory for output PNGs (default: same dir as MI350X CSV)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.mi350x_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mi350x_all = load_csv(args.mi350x_csv)
    mi355x_all = load_csv(args.mi355x_csv)

    if not mi350x_all:
        print("No MI350X data found.", file=sys.stderr)
        sys.exit(1)
    if not mi355x_all:
        print("No MI355X data found.", file=sys.stderr)
        sys.exit(1)

    mi350x_by_tp = group_by_tp(mi350x_all)
    mi355x_by_tp = group_by_tp(mi355x_all)

    for hw, groups in [("MI350X", mi350x_by_tp), ("MI355X", mi355x_by_tp)]:
        for tp, rows in sorted(groups.items()):
            print(f"  {hw} TP={tp}: {len(rows)} rows")

    series = _build_series(("mi350x", mi350x_by_tp), ("mi355x", mi355x_by_tp))

    plot_e2e_vs_throughput(series, out_dir / "tput_vs_e2e_latency.png")
    plot_interactivity_vs_throughput(series, out_dir / "tput_vs_interactivity.png")


if __name__ == "__main__":
    main()
