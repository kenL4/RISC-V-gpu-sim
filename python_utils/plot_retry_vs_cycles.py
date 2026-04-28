#!/usr/bin/env python3
"""Plot GPU cycles vs wall time to show CPU overhead effect."""

import matplotlib.pyplot as plt
import numpy as np

data = [
    # (Kernel, SIMTight_Cycles, Our_Cycles, SIMTight_WallTime_ms, Our_WallTime_ms)
    # Cycles from compare_results.py, wall times from trace logs
]

# Read wall times from trace files
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from plot_stats import read_simtight_trace

mine = read_simtight_trace("trace.log")
simtight = read_simtight_trace("trace_simtight.log")

kernels_common = sorted(set(mine.keys()) & set(simtight.keys()))

kernels = []
mine_cycles = []
st_cycles = []
mine_wall = []
st_wall = []

for k in kernels_common:
    if mine[k].get("WallTime_ms", 0) == 0 or simtight[k].get("WallTime_ms", 0) == 0:
        continue
    short = k.replace("Samples/", "").replace("InHouse/", "")
    kernels.append(short)
    mine_cycles.append(mine[k]["Cycles"])
    st_cycles.append(simtight[k]["Cycles"])
    mine_wall.append(mine[k]["WallTime_ms"])
    st_wall.append(simtight[k]["WallTime_ms"])

fig, ax = plt.subplots(figsize=(10, 6.5))

ax.scatter(mine_cycles, mine_wall, s=60, color="#636EFA", zorder=5, label="Ours")
ax.scatter(st_cycles, st_wall, s=60, color="#EF553B", zorder=5, marker="^", label="SIMTight")

# Draw lines connecting each kernel's two points
for i, kernel in enumerate(kernels):
    ax.plot([mine_cycles[i], st_cycles[i]], [mine_wall[i], st_wall[i]],
            color="grey", linewidth=0.8, alpha=0.4, zorder=3)

# Label each point (label once, midpoint between the two)
for i, kernel in enumerate(kernels):
    mid_x = (mine_cycles[i] + st_cycles[i]) / 2
    mid_y = max(mine_wall[i], st_wall[i])
    ax.annotate(kernel, (mid_x, mid_y),
                ha="center", va="bottom", fontsize=8.5, xytext=(0, 4),
                textcoords="offset points")

# Trendline on our data
coeffs = np.polyfit(mine_cycles, mine_wall, 1)
trend_x = np.linspace(0, max(max(mine_cycles), max(st_cycles)) * 1.05, 100)
trend_y = np.polyval(coeffs, trend_x)
r = np.corrcoef(mine_cycles, mine_wall)[0, 1]

ax.plot(trend_x, trend_y, color="#636EFA", linestyle="--", linewidth=1.5,
        alpha=0.4, label=f"Ours trend (r = {r:.2f})")

# Trendline on SIMTight data
coeffs_st = np.polyfit(st_cycles, st_wall, 1)
trend_y_st = np.polyval(coeffs_st, trend_x)
r_st = np.corrcoef(st_cycles, st_wall)[0, 1]

ax.plot(trend_x, trend_y_st, color="#EF553B", linestyle="--", linewidth=1.5,
        alpha=0.4, label=f"SIMTight trend (r = {r_st:.2f})")

ax.set_xlabel("GPU Cycles", fontsize=12)
ax.set_ylabel("Wall Time (ms)", fontsize=12)
ax.set_title("GPU Cycles vs Wall Time", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("figures/cycles_vs_walltime.svg", format="svg")
print(f"Saved to figures/cycles_vs_walltime.svg")
print(f"Ours trendline:    wall = {coeffs[0]:.6f} * cycles + ({coeffs[1]:.2f}),  r = {r:.3f}")
print(f"SIMTight trendline: wall = {coeffs_st[0]:.6f} * cycles + ({coeffs_st[1]:.2f}),  r = {r_st:.3f}")
print()
for i, k in enumerate(kernels):
    print(f"  {k:20s}  ours: {mine_cycles[i]:>7d} cyc / {mine_wall[i]:>5d} ms    ST: {st_cycles[i]:>7d} cyc / {st_wall[i]:>5d} ms")
