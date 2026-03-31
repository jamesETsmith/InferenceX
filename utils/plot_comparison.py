"""Plot MI350X vs MI355X benchmark comparisons.

Usage:
    python plot_comparison.py <mi350x_csv> <mi355x_csv> [-o output_dir]

Generates two plots:
  1. Throughput/GPU vs Mean E2E Latency
  2. Throughput/GPU vs Median Interactivity
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


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


def plot_throughput_vs_e2e(mi350x: list[dict], mi355x: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for data, color, marker in [(mi350x, "#e74c3c", "o"), (mi355x, "#2980b9", "s")]:
        tput = [float(r["Throughput/GPU (tok/s)"]) for r in data]
        e2e = [float(r["Mean E2E Latency (ms)"]) for r in data]
        concs = [int(r["Concurrency"]) for r in data]
        label = make_label(data[0]["Hardware"], data[0]["ISL"], data[0]["OSL"])

        ax.plot(tput, e2e, color=color, marker=marker, linewidth=2, markersize=8, label=label, zorder=3)
        for t, e, c in zip(tput, e2e, concs):
            ax.annotate(f"c={c}", (t, e), textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=color)

    ax.set_xlabel("Throughput / GPU (tok/s)", fontsize=12)
    ax.set_ylabel("Mean E2E Latency (s)", fontsize=12)
    ax.set_title("Kimi-K2.5 FP4 — Throughput vs E2E Latency (TP=8)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def plot_throughput_vs_interactivity(mi350x: list[dict], mi355x: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for data, color, marker in [(mi350x, "#e74c3c", "o"), (mi355x, "#2980b9", "s")]:
        tput = [float(r["Throughput/GPU (tok/s)"]) for r in data]
        intvty = [float(r["Median Interactivity (tok/s/user)"]) for r in data]
        concs = [int(r["Concurrency"]) for r in data]
        label = make_label(data[0]["Hardware"], data[0]["ISL"], data[0]["OSL"])

        ax.plot(tput, intvty, color=color, marker=marker, linewidth=2, markersize=8, label=label, zorder=3)
        for t, i, c in zip(tput, intvty, concs):
            ax.annotate(f"c={c}", (t, i), textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=color)

    ax.set_xlabel("Throughput / GPU (tok/s)", fontsize=12)
    ax.set_ylabel("Median Interactivity (tok/s/user)", fontsize=12)
    ax.set_title("Kimi-K2.5 FP4 — Throughput vs Interactivity (TP=8)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
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

    plot_throughput_vs_e2e(mi350x_all, mi355x_tp8, out_dir / "tput_vs_e2e_latency.png")
    plot_throughput_vs_interactivity(mi350x_all, mi355x_tp8, out_dir / "tput_vs_interactivity.png")


if __name__ == "__main__":
    main()
