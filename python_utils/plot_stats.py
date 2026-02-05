import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _read_one_stat_block(f, first_line=None):
    """Read one stat block (Cycles through DRAMAccs), return (block_dict, last_line).
    If first_line is provided, use it as the first line of the block; otherwise read from f.
    """
    block = {"Cycles": 0, "Instrs": 0, "Susps": 0, "Retries": 0, "DRAMAccs": 0}
    line = first_line
    while True:
        if line is None:
            line = f.readline()
        parts = line.split(":")
        if parts[0] == "Self test":
            return block, line
        if parts[0] not in ["Detected an error at index", "Expected value", "Computed value"] and parts[0] in block:
            block[parts[0]] = int(parts[1], 16)
        if parts[0] == "DRAMAccs":
            return block, line
        line = None


def read_simtight_trace(file, expand_subkernels=False):
    """Read SIMTight trace file into a dict of kernel name -> stats.

    If expand_subkernels is False (default), kernels that run multiple times
    (e.g. Samples/BitonicSortLarge with 3 sub-kernels) are combined into one
    entry. If True, each invocation is stored as a separate entry with labels
    like "Samples/BitonicSortLarge (1)", "Samples/BitonicSortLarge (2)", etc.,
    so they can be plotted as separate bars.
    """
    f = open(file, "r")
    data = {}
    line = f.readline()
    while line:
        parts = line.split(" ")

        if len(parts) == 0 or parts[0] != "Running":
            line = f.readline()
            continue

        kernel_base = parts[2].strip()
        inv = 0
        first_line = None  # when expand_subkernels: next block may start with this line

        while True:
            inv += 1
            kernel = f"{kernel_base} ({inv})" if expand_subkernels else kernel_base
            if kernel not in data:
                data[kernel] = {
                    "Cycles": 0,
                    "Instrs": 0,
                    "Susps": 0,
                    "Retries": 0,
                    "DRAMAccs": 0,
                }

            block, line = _read_one_stat_block(f, first_line=first_line)
            first_line = None
            for k, v in block.items():
                data[kernel][k] += v

            data[kernel]["IPC"] = (
                float(data[kernel]["Instrs"]) / data[kernel]["Cycles"]
                if data[kernel]["Cycles"] else 0.0
            )

            if not expand_subkernels:
                # Consume remaining stat blocks for this kernel and aggregate
                while True:
                    line = f.readline()
                    if not line or line.strip().startswith("Running kernel") or line.strip().startswith("Self test"):
                        break
                    if line.strip().startswith("Cycles:"):
                        block, line = _read_one_stat_block(f, first_line=line)
                        for k, v in block.items():
                            data[kernel][k] += v
                        data[kernel]["IPC"] = (
                            float(data[kernel]["Instrs"]) / data[kernel]["Cycles"]
                            if data[kernel]["Cycles"] else 0.0
                        )
                break

            line = f.readline()
            if not line or not line.strip().startswith("Cycles:"):
                break
            first_line = line  # next iteration reads a block starting with this line
    return data


def _bar_fig(kernels, mine_vals, simtight_vals, ylabel, title):
    """Build a grouped bar chart comparing Mine vs SIMTight."""
    fig = go.Figure(
        data=[
            go.Bar(name="Mine", x=kernels, y=mine_vals),
            go.Bar(name="SIMTight", x=kernels, y=simtight_vals),
        ]
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Kernel",
        yaxis_title=ylabel,
        title=title,
        legend_title="GPU",
    )
    return fig


def _common_kernels(data, simtight):
    """Return sorted list of kernel names that appear in both data and simtight."""
    return sorted(set(data.keys()) & set(simtight.keys()))


def plot_gpu_instrs(data, simtight):
    kernels = _common_kernels(data, simtight)
    gpu_instrs = [data[name]["Instrs"] for name in kernels]
    simtight_gpu_instrs = [simtight[name]["Instrs"] for name in kernels]
    fig = _bar_fig(
        kernels, gpu_instrs, simtight_gpu_instrs, "Instrs", "GPU Instrs by GPU, Kernel"
    )
    fig.show()


def plot_gpu_cycles(data, simtight):
    kernels = _common_kernels(data, simtight)
    gpu_cycles = [data[name]["Cycles"] for name in kernels]
    simtight_gpu_cycles = [simtight[name]["Cycles"] for name in kernels]
    fig = _bar_fig(
        kernels, gpu_cycles, simtight_gpu_cycles, "Cycles", "GPU Cycles by GPU, Kernel"
    )
    fig.show()


def plot_dram_accs(data, simtight):
    kernels = _common_kernels(data, simtight)
    gpu_dram_accs = [data[name]["DRAMAccs"] for name in kernels]
    simtight_gpu_dram_accs = [simtight[name]["DRAMAccs"] for name in kernels]
    fig = _bar_fig(
        kernels,
        gpu_dram_accs,
        simtight_gpu_dram_accs,
        "DRAMAccs",
        "GPU DRAMAccs by GPU, Kernel",
    )
    fig.show()


def plot_ipc(data, simtight):
    kernels = _common_kernels(data, simtight)
    gpu_ipc = [data[name]["IPC"] for name in kernels]
    simtight_gpu_ipc = [simtight[name]["IPC"] for name in kernels]
    fig = _bar_fig(
        kernels, gpu_ipc, simtight_gpu_ipc, "IPC", "GPU IPC by GPU, Kernel"
    )
    fig.show()


def plot_gpu_retries(data, simtight):
    kernels = _common_kernels(data, simtight)
    gpu_retries = [data[name].get("Retries", 0) for name in kernels]
    simtight_gpu_retries = [simtight[name].get("Retries", 0) for name in kernels]
    fig = _bar_fig(
        kernels,
        gpu_retries,
        simtight_gpu_retries,
        "Retries",
        "GPU Retries by GPU, Kernel",
    )
    fig.show()


def plot_gpu_all(data, simtight):
    """Show each metric in its own figure."""
    plot_dram_accs(data, simtight)
    plot_gpu_cycles(data, simtight)
    plot_gpu_instrs(data, simtight)
    plot_ipc(data, simtight)


SINGLE_KERNEL_METRICS = [
    ("DRAMAccs", "DRAMAccs", "GPU DRAMAccs"),
    ("Cycles", "Cycles", "GPU Cycles"),
    ("Instrs", "Instrs", "GPU Instrs"),
    ("IPC", "IPC", "GPU IPC"),
]


def plot_single_kernel(data, simtight, kernel_name, metric=None):
    """Plot one or all metrics for a single kernel.

    Args:
        data, simtight: trace data dicts from read_simtight_trace.
        kernel_name: kernel to plot (e.g. "Samples/VecAdd").
        metric: one of "DRAMAccs", "Cycles", "Instrs", "IPC", or None for all.
    """
    if kernel_name not in data or kernel_name not in simtight:
        available = set(data.keys()) | set(simtight.keys())
        raise KeyError(
            f"Kernel {kernel_name!r} not found in trace data. "
            f"Available: {sorted(available)}"
        )
    valid = [m[0] for m in SINGLE_KERNEL_METRICS]
    if metric is not None and metric not in valid:
        raise ValueError(f"metric must be one of {valid}, got {metric!r}")
    mine = data[kernel_name]
    baseline = simtight[kernel_name]
    if metric is not None:
        # Single metric: one simple bar chart
        entry = next(m for m in SINGLE_KERNEL_METRICS if m[0] == metric)
        key, ylabel, title = entry
        fig = go.Figure(
            data=[
                go.Bar(name="Mine", x=[kernel_name], y=[mine[key]]),
                go.Bar(name="SIMTight", x=[kernel_name], y=[baseline[key]]),
            ]
        )
        fig.update_layout(
            barmode="group",
            xaxis_title="Kernel",
            yaxis_title=ylabel,
            title=f"{title} — {kernel_name}",
        )
        fig.show()
        return
    # All metrics: 2x2 subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[m[2] for m in SINGLE_KERNEL_METRICS],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    for idx, (key, ylabel, _) in enumerate(SINGLE_KERNEL_METRICS):
        row, col = idx // 2 + 1, idx % 2 + 1
        fig.add_trace(
            go.Bar(name="Mine", x=[kernel_name], y=[mine[key]], legendgroup="Mine"),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Bar(
                name="SIMTight",
                x=[kernel_name],
                y=[baseline[key]],
                legendgroup="SIMTight",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Kernel", row=row, col=col)
        fig.update_yaxes(title_text=ylabel, row=row, col=col)
    fig.update_layout(
        barmode="group",
        title_text=f"GPU stats for {kernel_name}",
        height=500,
    )
    fig.show()


if __name__ == "__main__":
    data = read_simtight_trace("trace.log")
    simtight = read_simtight_trace("trace_simtight.log")

    # plot_gpu_instrs(data, simtight)
    # plot_gpu_cycles(data, simtight)
    # plot_dram_accs(data, simtight)
    plot_gpu_all(data, simtight)
    #plot_single_kernel(data, simtight, "Samples/VecAdd", metric="Instrs")

    # Show BitonicSortLarge as three separate sub-kernels instead of one combined bar:
    # data_exp = read_simtight_trace("trace.log", expand_subkernels=True)
    # simtight_exp = read_simtight_trace("trace_simtight.log", expand_subkernels=True)
    # plot_gpu_cycles(data_exp, simtight_exp)  # BitonicSortLarge (1), (2), (3) as separate bars
