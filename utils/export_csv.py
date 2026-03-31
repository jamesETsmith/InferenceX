"""Export processed (agg_*) benchmark JSON results to the InferenceX website CSV format.

Usage:
    python export_csv.py <results_dir> [-o output.csv]

The output CSV is column-compatible with the CSVs downloadable from https://inferencex.com,
so rows from both sources can be concatenated for side-by-side comparison.
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Model prefix → display name used on the website.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "kimik2.5": "Kimi-K2.5",
    "dsr1": "DeepSeek-R1",
    "gptoss": "GPT-OSS-120B",
}

CSV_COLUMNS = [
    "Model",
    "ISL",
    "OSL",
    "Hardware",
    "Hardware Key",
    "Framework",
    "Precision",
    "TP",
    "Concurrency",
    "Date",
    "Throughput/GPU (tok/s)",
    "Output Throughput/GPU (tok/s)",
    "Input Throughput/GPU (tok/s)",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Std TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Std TPOT (ms)",
    "Mean Interactivity (tok/s/user)",
    "Median Interactivity (tok/s/user)",
    "P99 Interactivity (tok/s/user)",
    "Std Interactivity (tok/s/user)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
    "Std ITL (ms)",
    "Mean E2E Latency (ms)",
    "Median E2E Latency (ms)",
    "P99 E2E Latency (ms)",
    "Std E2E Latency (ms)",
    "Disaggregated",
    "Num Prefill GPUs",
    "Num Decode GPUs",
    "Spec Decoding",
    "EP",
    "DP Attention",
    "Is Multinode",
]


def _model_display_name(r: dict[str, Any]) -> str:
    prefix = r.get("infmax_model_prefix", "")
    if prefix in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[prefix]
    # Fallback: strip org prefix from served model name (e.g. "amd/Kimi-K2.5-MXFP4" → "Kimi-K2.5-MXFP4")
    model = r.get("model", prefix)
    return model.split("/")[-1] if "/" in model else model


def result_to_row(r: dict[str, Any]) -> dict[str, Any]:
    """Map a single processed-result dict to a website-CSV row dict."""
    is_multi = r.get("is_multinode", False)

    return {
        "Model": _model_display_name(r),
        "ISL": r["isl"],
        "OSL": r["osl"],
        "Hardware": r["hw"],
        "Hardware Key": f"{r['hw']}_{r['framework']}",
        "Framework": r["framework"],
        "Precision": r["precision"],
        "TP": r.get("tp", r.get("prefill_tp", "")),
        "Concurrency": r["conc"],
        "Date": r.get("date", date.today().isoformat()),
        "Throughput/GPU (tok/s)": r["tput_per_gpu"],
        "Output Throughput/GPU (tok/s)": r["output_tput_per_gpu"],
        "Input Throughput/GPU (tok/s)": r["input_tput_per_gpu"],
        "Mean TTFT (ms)": r.get("mean_ttft", ""),
        "Median TTFT (ms)": r.get("median_ttft", ""),
        "P99 TTFT (ms)": r.get("p99_ttft", ""),
        "Std TTFT (ms)": r.get("std_ttft", ""),
        "Mean TPOT (ms)": r.get("mean_tpot", ""),
        "Median TPOT (ms)": r.get("median_tpot", ""),
        "P99 TPOT (ms)": r.get("p99_tpot", ""),
        "Std TPOT (ms)": r.get("std_tpot", ""),
        "Mean Interactivity (tok/s/user)": r.get("mean_intvty", ""),
        "Median Interactivity (tok/s/user)": r.get("median_intvty", ""),
        "P99 Interactivity (tok/s/user)": r.get("p99_intvty", ""),
        "Std Interactivity (tok/s/user)": r.get("std_intvty", ""),
        "Mean ITL (ms)": r.get("mean_itl", ""),
        "Median ITL (ms)": r.get("median_itl", ""),
        "P99 ITL (ms)": r.get("p99_itl", ""),
        "Std ITL (ms)": r.get("std_itl", ""),
        "Mean E2E Latency (ms)": r.get("mean_e2el", ""),
        "Median E2E Latency (ms)": r.get("median_e2el", ""),
        "P99 E2E Latency (ms)": r.get("p99_e2el", ""),
        "Std E2E Latency (ms)": r.get("std_e2el", ""),
        "Disaggregated": str(r.get("disagg", False)).lower(),
        "Num Prefill GPUs": r.get("num_prefill_gpu", ""),
        "Num Decode GPUs": r.get("num_decode_gpu", ""),
        "Spec Decoding": r.get("spec_decoding", "none"),
        "EP": r.get("ep", r.get("prefill_ep", "")),
        "DP Attention": str(r.get("dp_attention", r.get("prefill_dp_attention", ""))).lower(),
        "Is Multinode": str(is_multi).lower(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export processed benchmark results to InferenceX website CSV format.")
    parser.add_argument("results_dir", type=Path, help="Directory containing processed (agg_*) JSON files")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for path in sorted(args.results_dir.rglob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if "is_multinode" not in data:
            print(f"Skipping (not a processed result): {path}", file=sys.stderr)
            continue
        results.append(data)

    if not results:
        print("No processed results found.", file=sys.stderr)
        sys.exit(1)

    results.sort(key=lambda r: (r.get("infmax_model_prefix", ""), r["hw"], r["framework"], r["isl"], r["osl"], r["conc"]))

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for r in results:
        writer.writerow(result_to_row(r))

    if args.output:
        out.close()
        print(f"Wrote {len(results)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
