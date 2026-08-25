#!/usr/bin/env python3
"""Run the complete paper reproduction in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(script: Path, *arguments: object) -> None:
    command = [sys.executable, str(script), *(str(value) for value in arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--skip-attribution", action="store_true")
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    repo = scripts.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    run(scripts / "run_proposed.py", "--data", args.data.resolve(), "--output-dir", output / "proposed", "--jobs", args.jobs)
    run(scripts / "run_comparators.py", "--data", args.data.resolve(), "--output-dir", output / "comparators", "--jobs", args.jobs)
    run(scripts / "run_paired_stats.py", "--proposed", output / "proposed", "--comparators", output / "comparators", "--output-dir", output / "paired_stats", "--comparator-set", "main")
    run(scripts / "run_calibration_hosmer.py", "--results", output, "--output-dir", output / "calibration")
    run(scripts / "run_ablation_stages.py", "--results", output, "--ablation-members", output / "comparators", "--output-dir", output / "ablation")
    run(scripts / "run_endpoint_sensitivity.py", "--proposed", output / "proposed", "--output-dir", output / "endpoint")
    run(scripts / "run_revision_analyses.py", "--results", output, "--output-dir", output / "revision")
    run(scripts / "run_structural_facts.py", "--root", repo, "--output-dir", output / "structural")
    if not args.skip_attribution:
        run(scripts / "run_attribution.py", "--data", args.data.resolve(), "--output-dir", output / "attribution", "--label-map", repo / "configs/feature_labels.csv")
    run(scripts / "make_figures.py", "--results", output, "--output", output / "figures")
    run(scripts / "build_tables.py", "--results", output, "--output", output / "tables")
    run(scripts / "verify_release.py", "--results", output)


if __name__ == "__main__":
    main()
