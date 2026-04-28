#!/usr/bin/env python3
"""Generate a LaTeX comparison table for any metric from trace logs."""

import argparse
import sys
from plot_stats import read_simtight_trace

# InHouse alphabetical, then Samples alphabetical
KERNEL_ORDER = [
    "InHouse/BlockedStencil",
    "InHouse/MotionEst",
    "InHouse/StripedStencil",
    "InHouse/VecGCD",
    "Samples/BitonicSortLarge",
    "Samples/BitonicSortSmall",
    "Samples/Histogram",
    "Samples/MatMul",
    "Samples/MatVecMul",
    "Samples/Reduce",
    "Samples/Scan",
    "Samples/SparseMatVecMul",
    "Samples/Transpose",
    "Samples/VecAdd",
]

VALID_METRICS = ["Cycles", "Instrs", "Susps", "Retries", "DRAMAccs", "IPC", "WallTime_ms"]


def short_name(kernel):
    return kernel.split("/", 1)[1]


def generate_table(baseline_file, simulator_file, metric, label=None, caption=None):
    baseline = read_simtight_trace(baseline_file)
    simulator = read_simtight_trace(simulator_file)

    if label is None:
        label = f"tab:{metric.lower()}"
    if caption is None:
        caption = f"{metric} comparison between our simulator and baseline"

    fmt = ".2f" if metric == "IPC" else "d"

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{|l|r|r|r|}")
    lines.append(r"\hline")
    lines.append(
        rf"\textbf{{Kernel}} & \textbf{{Baseline {metric}}} & "
        rf"\textbf{{Simulator {metric}}} & \textbf{{\% Difference}} \\"
    )
    lines.append(r"\hline")

    for kernel in KERNEL_ORDER:
        if kernel not in baseline or kernel not in simulator:
            continue
        b = baseline[kernel][metric]
        s = simulator[kernel][metric]
        if b != 0:
            pct = (s - b) / b * 100
        else:
            pct = 0.0

        if fmt == "d":
            b_str = str(int(b))
            s_str = str(int(s))
        else:
            b_str = f"{b:.2f}"
            s_str = f"{s:.2f}"

        lines.append(
            rf"{short_name(kernel)} & {b_str} & {s_str} & {pct:+.2f}\% \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_scheduler_table(fair_file, random_file, metric, label=None, caption=None):
    fair = read_simtight_trace(fair_file)
    random = read_simtight_trace(random_file)

    if label is None:
        label = f"tab:sched_{metric.lower()}"
    if caption is None:
        caption = f"{metric} comparison between fair and random warp schedulers"

    fmt = ".2f" if metric == "IPC" else "d"

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{|l|r|r|r|}")
    lines.append(r"\hline")
    lines.append(
        rf"\textbf{{Kernel}} & \textbf{{Fair {metric}}} & "
        rf"\textbf{{Random {metric}}} & \textbf{{\% Difference}} \\"
    )
    lines.append(r"\hline")

    for kernel in KERNEL_ORDER:
        if kernel not in fair or kernel not in random:
            continue
        f = fair[kernel][metric]
        r = random[kernel][metric]
        if f != 0:
            pct = (r - f) / f * 100
        else:
            pct = 0.0

        if fmt == "d":
            f_str = str(int(f))
            r_str = str(int(r))
        else:
            f_str = f"{f:.2f}"
            r_str = f"{r:.2f}"

        lines.append(
            rf"{short_name(kernel)} & {f_str} & {r_str} & {pct:+.2f}\% \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX comparison table from trace logs")
    subparsers = parser.add_subparsers(dest="command")

    baseline_parser = subparsers.add_parser("baseline", help="Compare simulator vs SIMTight baseline")
    baseline_parser.add_argument("metric", choices=VALID_METRICS, help="Metric to compare")
    baseline_parser.add_argument("--baseline", default="trace_simtight.log", help="Baseline trace file")
    baseline_parser.add_argument("--simulator", default="trace.log", help="Simulator trace file")
    baseline_parser.add_argument("--label", help="LaTeX label (default: tab:<metric>)")
    baseline_parser.add_argument("--caption", help="Table caption")

    sched_parser = subparsers.add_parser("scheduler", help="Compare fair vs random warp scheduler")
    sched_parser.add_argument("metric", choices=VALID_METRICS, help="Metric to compare")
    sched_parser.add_argument("--fair", default="trace.log", help="Fair scheduler trace file")
    sched_parser.add_argument("--random", default="trace_random_scheduler.log", help="Random scheduler trace file")
    sched_parser.add_argument("--label", help="LaTeX label (default: tab:sched_<metric>)")
    sched_parser.add_argument("--caption", help="Table caption")

    args = parser.parse_args()

    if args.command == "scheduler":
        print(generate_scheduler_table(args.fair, args.random, args.metric, args.label, args.caption))
    elif args.command == "baseline":
        print(generate_table(args.baseline, args.simulator, args.metric, args.label, args.caption))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
