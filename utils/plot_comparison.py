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
except ImportError:
    print("matplotlib and seaborn are required: pip install matplotlib seaborn", file=sys.stderr)
    sys.exit(1)

sns.set_theme(context="talk", style="whitegrid")

# ColorBrewer "Paired" palette strategy:
#   Hue encodes TP (orange family = TP4, blue family = TP8).
#   Lightness/saturation encodes hardware (strong = MI350X, soft = MI355X).
SERIES_COLORS: dict[tuple[int, str], str] = {
    (4, "mi350x"): "#ff7f00",   # strong orange
    (4, "mi355x"): "#fdbf6f",   # soft gold
    (8, "mi350x"): "#1f78b4",   # strong blue
    (8, "mi355x"): "#a6cee3",   # soft blue
}
COLOR_DEFAULT = "#33a02c"

HW_MARKERS: dict[str, str] = {
    "mi350x": "o",   # circle
    "mi355x": "s",   # square
}
HW_MARKER_DEFAULT = "D"


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
    for hw_key, tp_groups in csv_groups:
        marker = HW_MARKERS.get(hw_key, HW_MARKER_DEFAULT)
        for tp in sorted(tp_groups):
            data = tp_groups[tp]
            color = SERIES_COLORS.get((tp, hw_key), COLOR_DEFAULT)
            style = {"color": color, "marker": marker, "hw": hw_key}
            label = make_label(data[0]["Hardware"], tp, data[0]["ISL"], data[0]["OSL"])
            series.append((data, style, label))
    return series


# MI355X labels above the trace, MI350X labels below
ANNOTATION_OFFSET = {"mi355x": (0, 16), "mi350x": (0, -16)}
ANNOTATION_VA = {"mi355x": "bottom", "mi350x": "top"}


def _pad_axes(ax: plt.Axes, pad_frac: float = 0.10) -> None:
    """Expand axis limits by a fraction so labels near the edges have room."""
    for getter, setter in [(ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim)]:
        lo, hi = getter()
        margin = (hi - lo) * pad_frac
        setter(lo - margin, hi + margin)


def plot_e2e_vs_throughput(series: list[tuple[list[dict], dict, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    for data, style, label in series:
        tput = [float(r["Throughput/GPU (tok/s)"]) for r in data]
        e2e = [float(r["Mean E2E Latency (ms)"]) for r in data]
        concs = [int(r["Concurrency"]) for r in data]
        hw = style["hw"]
        offset = ANNOTATION_OFFSET.get(hw, (0, 16))
        va = ANNOTATION_VA.get(hw, "bottom")

        ax.plot(e2e, tput, color=style["color"], marker=style["marker"],
                linewidth=3, markersize=10, label=label, zorder=3)
        for e, t, c in zip(e2e, tput, concs):
            ax.annotate(f"c={c}", (e, t), textcoords="offset points", xytext=offset,
                        fontsize=13, fontweight="bold", color=style["color"],
                        ha="center", va=va)

    _pad_axes(ax)

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

    for data, style, label in series:
        tput = [float(r["Throughput/GPU (tok/s)"]) for r in data]
        intvty = [float(r["Median Interactivity (tok/s/user)"]) for r in data]
        concs = [int(r["Concurrency"]) for r in data]
        hw = style["hw"]
        offset = ANNOTATION_OFFSET.get(hw, (0, 16))
        va = ANNOTATION_VA.get(hw, "bottom")

        ax.plot(intvty, tput, color=style["color"], marker=style["marker"],
                linewidth=3, markersize=10, label=label, zorder=3)
        for i, t, c in zip(intvty, tput, concs):
            ax.annotate(f"c={c}", (i, t), textcoords="offset points", xytext=offset,
                        fontsize=13, fontweight="bold", color=style["color"],
                        ha="center", va=va)

    _pad_axes(ax)

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
