#!/usr/bin/env python3
"""Compute quartiles of % difference between baseline and simulator for each metric."""

import argparse
import numpy as np
from plot_stats import read_simtight_trace
from generate_table import KERNEL_ORDER, short_name


METRICS = ["Cycles", "Instrs", "Retries", "DRAMAccs", "IPC", "WallTime_ms"]


def compute_pct_diffs(baseline, simulator, metric):
    diffs = []
    for kernel in KERNEL_ORDER:
        if kernel not in baseline or kernel not in simulator:
            continue
        b = baseline[kernel][metric]
        s = simulator[kernel][metric]
        if b != 0:
            diffs.append((s - b) / b * 100)
    return np.array(diffs)


def generate_quartile_table(baseline_file, simulator_file, metrics=None):
    baseline = read_simtight_trace(baseline_file)
    simulator = read_simtight_trace(simulator_file)

    if metrics is None:
        metrics = METRICS

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quartiles of \% difference between simulator and baseline}")
    lines.append(r"\label{tab:quartiles}")
    lines.append(r"\begin{tabular}{|l|r|r|r|r|r|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Metric} & \textbf{Min} & \textbf{Q1} & "
        r"\textbf{Median} & \textbf{Q3} & \textbf{Max} \\"
    )
    lines.append(r"\hline")

    for metric in metrics:
        diffs = compute_pct_diffs(baseline, simulator, metric)
        if len(diffs) == 0:
            continue
        q0 = np.min(diffs)
        q1 = np.percentile(diffs, 25)
        q2 = np.median(diffs)
        q3 = np.percentile(diffs, 75)
        q4 = np.max(diffs)
        lines.append(
            rf"{metric} & {q0:+.2f}\% & {q1:+.2f}\% & "
            rf"{q2:+.2f}\% & {q3:+.2f}\% & {q4:+.2f}\% \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute quartiles of % difference per metric")
    parser.add_argument("--baseline", default="trace_simtight.log", help="Baseline trace file")
    parser.add_argument("--simulator", default="trace.log", help="Simulator trace file")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, help="Metrics to include")
    args = parser.parse_args()

    print(generate_quartile_table(args.baseline, args.simulator, args.metrics))


if __name__ == "__main__":
    main()
