"""Post-processing pipeline for local MI350X benchmark results.

Parses config values (TP, ISL, OSL, precision, …) from each result filename,
enriches the raw JSON via process_result.py, then runs the downstream
summarise → export CSV → comparison-plot pipeline.

Filename convention:
    {model}_{precision}_{hardware}_tp{N}_isl{N}_osl{N}_conc{N}.json

Usage:
    python utils/analyze_mi350x_results.py [options]
    python utils/analyze_mi350x_results.py --raw-dir results_mi350x --out-dir agg_results_mi350x
    python utils/analyze_mi350x_results.py --image vllm/vllm-openai-rocm:v0.19.0
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

FILENAME_RE = re.compile(
    r"^(?P<model>.+?)_(?P<precision>fp\d+)_(?P<hardware>[a-z0-9]+)"
    r"_tp(?P<tp>\d+)_isl(?P<isl>\d+)_osl(?P<osl>\d+)_conc(?P<conc>\d+)\.json$"
)

REPO_DIR = Path(__file__).resolve().parent.parent


def parse_filename(name: str) -> dict[str, str] | None:
    """Extract config fields encoded in a result filename."""
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return m.groupdict()


def process_one(
    raw_json: Path,
    out_dir: Path,
    *,
    defaults: dict[str, str],
) -> Path | None:
    """Run process_result.py for a single raw JSON, return the output path or None."""
    rel = raw_json.relative_to(REPO_DIR)
    basename_no_ext = str(rel.with_suffix(""))  # e.g. results_mi350x/kimik2.5_..._conc4
    agg_output = REPO_DIR / f"agg_{basename_no_ext}.json"

    parsed = parse_filename(raw_json.name)
    if parsed is None:
        print(f"  [warn] Cannot parse filename, skipping: {raw_json.name}", file=sys.stderr)
        return None

    # Read the raw JSON to pull framework from the 'backend' field
    with open(raw_json) as f:
        raw = json.load(f)
    framework = raw.get("backend", defaults.get("FRAMEWORK", "vllm"))

    env = {
        **os.environ,
        "RUNNER_TYPE": parsed["hardware"],
        "FRAMEWORK": framework,
        "PRECISION": parsed["precision"],
        "MODEL_PREFIX": parsed["model"],
        "ISL": parsed["isl"],
        "OSL": parsed["osl"],
        "TP": parsed["tp"],
        "RESULT_FILENAME": basename_no_ext,
        # Apply caller-provided defaults for fields not in the filename
        **{k: v for k, v in defaults.items() if k not in (
            "RUNNER_TYPE", "FRAMEWORK", "PRECISION", "MODEL_PREFIX",
            "ISL", "OSL", "TP", "RESULT_FILENAME",
        )},
    }

    tag = f"TP={parsed['tp']} ISL={parsed['isl']} OSL={parsed['osl']} conc={parsed['conc']}"
    print(f"  Processing: {rel}  ({tag})")

    result = subprocess.run(
        [sys.executable, str(REPO_DIR / "utils" / "process_result.py")],
        cwd=str(REPO_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [error] process_result.py failed for {raw_json.name}:\n{result.stderr}", file=sys.stderr)
        return None

    print(f"    -> {agg_output.relative_to(REPO_DIR)}")
    return agg_output


def run_step(label: str, cmd: list[str]) -> bool:
    """Run a subprocess, printing its stdout. Returns True on success."""
    print(f"\n=== {label} ===\n", flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_DIR))
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse local MI350X benchmark results (process → summarise → CSV → plots).",
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_DIR / "results_mi350x",
                        help="Directory containing raw benchmark JSON files")
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "agg_results_mi350x",
                        help="Output directory for processed results")
    parser.add_argument("--comparison-csv", type=Path, default=None,
                        help="MI355X CSV for comparison plots (default: <out-dir>/mi355x_results.csv)")
    parser.add_argument("--image", default="vllm/vllm-openai-rocm:v0.18.0",
                        help="Docker image used for the benchmark run")
    parser.add_argument("--spec-decoding", default="false")
    parser.add_argument("--disagg", default="false")
    parser.add_argument("--is-multinode", default="false")
    parser.add_argument("--ep-size", default="1")
    parser.add_argument("--dp-attention", default="false")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_csv = args.comparison_csv or (out_dir / "mi355x_results.csv")

    defaults = {
        "IMAGE": args.image,
        "SPEC_DECODING": args.spec_decoding,
        "DISAGG": args.disagg,
        "IS_MULTINODE": args.is_multinode,
        "EP_SIZE": args.ep_size,
        "DP_ATTENTION": args.dp_attention,
    }

    # --- Step 1: Process each raw result --------------------------------------
    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        print(f"No JSON files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Step 1: Processing raw results ({len(raw_files)} files) ===\n", flush=True)

    processed = 0
    for raw_json in raw_files:
        result = process_one(raw_json, out_dir, defaults=defaults)
        if result is not None:
            processed += 1

    print(f"\n  {processed} result(s) ready in {out_dir.relative_to(REPO_DIR)}")

    # --- Step 2: Summarise ----------------------------------------------------
    py = sys.executable
    run_step("Step 2: Generating summary table",
             [py, str(REPO_DIR / "utils" / "summarize.py"), str(out_dir)])

    # --- Step 3: Export CSV ---------------------------------------------------
    csv_output = out_dir / "results.csv"
    run_step("Step 3: Exporting CSV",
             [py, str(REPO_DIR / "utils" / "export_csv.py"), str(out_dir), "-o", str(csv_output)])
    print(f"CSV written to: {csv_output.relative_to(REPO_DIR)}")

    # --- Step 4: Comparison plots ---------------------------------------------
    if comparison_csv.exists():
        run_step("Step 4: Generating comparison plots",
                 [py, str(REPO_DIR / "utils" / "plot_comparison.py"),
                  str(csv_output), str(comparison_csv), "-o", str(out_dir)])
    else:
        print(f"\n=== Step 4: Skipped (no comparison CSV at {comparison_csv.relative_to(REPO_DIR)}) ===")
        print(f"  Place the MI355X website CSV there to enable comparison plots.")


if __name__ == "__main__":
    main()
