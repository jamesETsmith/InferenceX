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


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV, skipping comment lines (# prefix)."""
    rows = []
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append(row)
    return rows


def make_label(hw: str, isl: str, osl: str) -> str:
    return f"{hw.upper()} ({isl}/{osl})"


def plot_e2e_vs_throughput(mi350x: list[dict], mi355x: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for data, color, marker in [(mi350x, "#e74c3c", "o"), (mi355x, "#2980b9", "s")]:
        tput = [float(r["Throughput/GPU (tok/s)"]) for r in data]
        e2e = [float(r["Mean E2E Latency (ms)"]) for r in data]
        concs = [int(r["Concurrency"]) for r in data]
        label = make_label(data[0]["Hardware"], data[0]["ISL"], data[0]["OSL"])

        ax.plot(e2e, tput, color=color, marker=marker, linewidth=2, markersize=9, label=label, zorder=3)
        for e, t, c in zip(e2e, tput, concs):
            ax.annotate(f"c={c}", (e, t), textcoords="offset points", xytext=(0, 14),
                        fontsize=13, color=color, ha="center")

    ax.set_xlabel("Mean E2E Latency (s)")
    ax.set_ylabel("Throughput / GPU (tok/s)")
    ax.set_title("Kimi-K2.5 FP4 — E2E Latency vs Throughput (TP=8)", fontweight="bold")
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def plot_interactivity_vs_throughput(mi350x: list[dict], mi355x: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for data, color, marker in [(mi350x, "#e74c3c", "o"), (mi355x, "#2980b9", "s")]:
        tput = [float(r["Throughput/GPU (tok/s)"]) for r in data]
        intvty = [float(r["Median Interactivity (tok/s/user)"]) for r in data]
        concs = [int(r["Concurrency"]) for r in data]
        label = make_label(data[0]["Hardware"], data[0]["ISL"], data[0]["OSL"])

        ax.plot(intvty, tput, color=color, marker=marker, linewidth=2, markersize=9, label=label, zorder=3)
        for i, t, c in zip(intvty, tput, concs):
            ax.annotate(f"c={c}", (i, t), textcoords="offset points", xytext=(0, 14),
                        fontsize=13, color=color, ha="center")

    ax.set_xlabel("Median Interactivity (tok/s/user)")
    ax.set_ylabel("Throughput / GPU (tok/s)")
    ax.set_title("Kimi-K2.5 FP4 — Interactivity vs Throughput (TP=8)", fontweight="bold")
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

    # Filter MI355X to TP=8 only
    mi355x_tp8 = [r for r in mi355x_all if int(r["TP"]) == 8]

    if not mi350x_all:
        print("No MI350X data found.", file=sys.stderr)
        sys.exit(1)
    if not mi355x_tp8:
        print("No MI355X TP=8 data found.", file=sys.stderr)
        sys.exit(1)

    # Sort by concurrency for connected line plots
    mi350x_all.sort(key=lambda r: int(r["Concurrency"]))
    mi355x_tp8.sort(key=lambda r: int(r["Concurrency"]))

    n350 = len(mi350x_all)
    n355 = len(mi355x_tp8)
    print(f"  MI350X: {n350} rows | MI355X (TP=8): {n355} rows")

    plot_e2e_vs_throughput(mi350x_all, mi355x_tp8, out_dir / "tput_vs_e2e_latency.png")
    plot_interactivity_vs_throughput(mi350x_all, mi355x_tp8, out_dir / "tput_vs_interactivity.png")


if __name__ == "__main__":
    main()
