#!/usr/bin/env python3
"""Compare simulator results with SIMTight baseline."""

data = [
    # (Kernel, SIMTight_Cycles, Our_Cycles, SIMTight_Instrs, Our_Instrs, SIMTight_Retries, Our_Retries, SIMTight_DRAM, Our_DRAM)
    ("VecAdd",           0x18d8,  0x18c4,  0x26918,  0x26918,  0x3ab,  0x4d4,  0x874,  0x874),
    ("Histogram",        0x1c5e,  0x1c7d,  0x2c8c8,  0x2c8a8,  0x3fa,  0x4ab,  0x82e,  0x82e),
    ("Reduce",           0x36ff,  0x3700,  0x5cdee,  0x5cdee,  0x3db,  0x4d8,  0x6fd,  0x6fd),
    ("Scan",             0x89d1,  0x89ac,  0xfbfe4,  0xfbfa4,  0x630,  0x6f6,  0xb00,  0xb00),
    ("Transpose",        0x4834,  0x48ea,  0x80fc0,  0x80fc0,  0x58e,  0x6c3,  0x1940, 0x1940),
    ("MatVecMul",        0x4394,  0x4308,  0x68740,  0x68760,  0x55e,  0x605,  0x1700, 0x1700),
    ("MatMul",           0x1302c, 0x13079, 0x23dfa0, 0x23dfa0, 0xc31,  0xc15,  0x2100, 0x2100),
    ("BitonicSortSmall", 0x1d5a5, 0x1d06b, 0x2eeb40, 0x2eeb20, 0x1ee1, 0x1aa7, 0x2740, 0x2740),  # -1.11%
    ("BitonicSortLarge", 0x58247, 0x58552, 0x7c77d8, 0x7c77b8, 0xe325, 0xe99d, 0x2400, 0x2400),
    ("SparseMatVecMul",  0x40f9,  0x409d,  0x259e0,  0x259e0,  0x509,  0x67c,  0xfdf,  0xfdf),
    ("BlockedStencil",   0x4420,  0x43b0,  0x6c360,  0x6c340,  0x824,  0x70f,  0x1640, 0x1640),
    ("StripedStencil",   0x3e05,  0x3e57,  0x696a0,  0x696a0,  0x616,  0x68d,  0x18f8, 0x18f8),
    ("VecGCD",           0x2d47,  0x2d6c,  0x206fb,  0x206fb,  0x3ac,  0x4d5,  0x658,  0x658),
    ("MotionEst",        0xa3d7,  0xa283,  0xda0dc,  0xda0bc,  0x12d6, 0x1203, 0x2fa8, 0x2fa8),
]


def fmt_ratio(ours, baseline):
    """Format ratio as a percentage difference like '+12%' or '-5%'."""
    pct = (ours - baseline) / baseline * 100
    if pct >= 0:
        return f"+{pct:.1f}%"
    else:
        return f"{pct:.1f}%"


# Column headers
headers = [
    "Kernel",
    "ST Cycles", "Our Cycles", "Cyc %",
    "ST Instrs", "Our Instrs", "Ins %",
    "ST Retries", "Our Retries", "Ret %",
    "ST DRAM", "Our DRAM", "DRAM %",
]

# Build rows
rows = []
for (name, sc, oc, si, oi, sr, orr, sd, od) in data:
    rows.append([
        name,
        str(sc), str(oc), fmt_ratio(oc, sc),
        str(si), str(oi), fmt_ratio(oi, si),
        str(sr), str(orr), fmt_ratio(orr, sr),
        str(sd), str(od), fmt_ratio(od, sd),
    ])

# Compute column widths
widths = [len(h) for h in headers]
for row in rows:
    for i, cell in enumerate(row):
        widths[i] = max(widths[i], len(cell))

# Separator line
sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"


def fmt_row(cells):
    parts = []
    for i, cell in enumerate(cells):
        if i == 0:
            parts.append(cell.ljust(widths[i]))
        else:
            parts.append(cell.rjust(widths[i]))
    return "| " + " | ".join(parts) + " |"


# Print table
print(sep)
print(fmt_row(headers))
print(sep)
for row in rows:
    print(fmt_row(row))
print(sep)

# Summary statistics
print()
cycle_ratios = [(oc - sc) / sc * 100 for (_, sc, oc, _, _, _, _, _, _) in data]
instr_ratios = [(oi - si) / si * 100 for (_, _, _, si, oi, _, _, _, _) in data]
retry_ratios = [(orr - sr) / sr * 100 for (_, _, _, _, _, sr, orr, _, _) in data]
dram_ratios  = [(od - sd) / sd * 100 for (_, _, _, _, _, _, _, sd, od) in data]

print(f"Cycle difference  -- mean: {sum(cycle_ratios)/len(cycle_ratios):+.1f}%,  "
      f"min: {min(cycle_ratios):+.1f}%,  max: {max(cycle_ratios):+.1f}%")
print(f"Instr difference  -- mean: {sum(instr_ratios)/len(instr_ratios):+.1f}%,  "
      f"min: {min(instr_ratios):+.1f}%,  max: {max(instr_ratios):+.1f}%")
print(f"Retry difference  -- mean: {sum(retry_ratios)/len(retry_ratios):+.1f}%,  "
      f"min: {min(retry_ratios):+.1f}%,  max: {max(retry_ratios):+.1f}%")
print(f"DRAM  difference  -- mean: {sum(dram_ratios)/len(dram_ratios):+.1f}%,  "
      f"min: {min(dram_ratios):+.1f}%,  max: {max(dram_ratios):+.1f}%")
