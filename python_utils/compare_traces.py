#!/usr/bin/env python3
"""
Compare cycle-by-cycle instruction traces between SIMTight and our simulator.

Usage:
    python3 compare_traces.py <oursim_trace.csv> <simtight_trace.csv> [--benchmark NAME]

Trace format (both simulators, after stripping SIMTight's TRACE: prefix):
    cycle,0xPC,warp_id,lane_id,event_type

Event types:
    0 = MEM_REQ_ISSUE
    1 = DRAM_REQ_ISSUE
    2 = INSTR_EXEC
    3 = WARP_RETRY
    4 = WARP_SUSPEND
    5 = WARP_RESUME
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass


EVENT_NAMES = {
    0: "MEM_REQ_ISSUE",
    1: "DRAM_REQ_ISSUE",
    2: "INSTR_EXEC",
    3: "WARP_RETRY",
    4: "WARP_SUSPEND",
    5: "WARP_RESUME",
}


@dataclass
class TraceEvent:
    cycle: int
    pc: int
    warp_id: int
    lane_id: int
    event_type: int


def parse_trace(filename: str) -> list[TraceEvent]:
    """Parse a trace CSV file into a list of TraceEvents."""
    events = []
    with open(filename) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Skip address lines (from MEM_REQ_ISSUE events in our sim)
            if line.startswith("0x") and "," in line and line.count(",") <= 32:
                # This is an address list line, skip
                parts = line.split(",")
                if all(p.strip() == "" or p.strip().startswith("0x") for p in parts):
                    continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                cycle = int(parts[0])
                pc = int(parts[1], 16)
                warp_id = int(parts[2])
                lane_id = int(parts[3])
                event_type = int(parts[4])
                events.append(TraceEvent(cycle, pc, warp_id, lane_id, event_type))
            except (ValueError, IndexError):
                print(f"Warning: skipping malformed line {lineno}: {line[:80]}", file=sys.stderr)
    return events


def to_warp_level(events: list[TraceEvent]) -> list[TraceEvent]:
    """Reduce per-lane events to per-warp events.

    If a trace has per-lane events (lane_id >= 0), keep only lane 0 events
    as warp representatives. If already per-warp (lane_id == -1), return as-is.
    """
    has_per_lane = any(e.lane_id >= 0 for e in events[:100])
    if not has_per_lane:
        return events
    # Keep only lane 0 (or lane -1) events
    return [e for e in events if e.lane_id <= 0]


def filter_events(events: list[TraceEvent], event_type: int) -> list[TraceEvent]:
    return [e for e in events if e.event_type == event_type]


def event_counts(events: list[TraceEvent]) -> dict[int, int]:
    counts = defaultdict(int)
    for e in events:
        counts[e.event_type] += 1
    return dict(counts)


def per_warp_events(events: list[TraceEvent], event_type: int) -> dict[int, list[TraceEvent]]:
    """Group events of a given type by warp_id, preserving order."""
    warps = defaultdict(list)
    for e in events:
        if e.event_type == event_type:
            warps[e.warp_id].append(e)
    return dict(warps)


def compare_scheduling(our_events: list[TraceEvent], st_events: list[TraceEvent],
                       max_show: int = 20) -> list[tuple[int, int, int, int, int]]:
    """
    Compare the global scheduling order (which warp executes at which cycle).
    Only considers INSTR_EXEC events.

    Returns list of (index, our_cycle, our_warp, st_cycle, st_warp) for divergences.
    """
    our_exec = filter_events(our_events, 2)  # INSTR_EXEC
    st_exec = filter_events(st_events, 2)

    divergences = []
    min_len = min(len(our_exec), len(st_exec))
    for i in range(min_len):
        o = our_exec[i]
        s = st_exec[i]
        if o.warp_id != s.warp_id or o.pc != s.pc:
            divergences.append((i, o.cycle, o.warp_id, s.cycle, s.warp_id))
            if len(divergences) >= max_show:
                break
    return divergences


def per_warp_drift(our_events: list[TraceEvent], st_events: list[TraceEvent]) -> dict[int, dict]:
    """
    For each warp, compare the cycle of the Nth instruction execution.
    Returns per-warp stats: {warp_id: {our_total, st_total, drift, pct, max_drift_instr, max_drift_val}}
    """
    our_warps = per_warp_events(our_events, 2)
    st_warps = per_warp_events(st_events, 2)

    all_warps = sorted(set(our_warps.keys()) | set(st_warps.keys()))
    result = {}

    for w in all_warps:
        our_instrs = our_warps.get(w, [])
        st_instrs = st_warps.get(w, [])

        if not our_instrs or not st_instrs:
            result[w] = {
                "our_count": len(our_instrs),
                "st_count": len(st_instrs),
                "our_total": our_instrs[-1].cycle - our_instrs[0].cycle if our_instrs else 0,
                "st_total": st_instrs[-1].cycle - st_instrs[0].cycle if st_instrs else 0,
                "drift": 0,
                "pct": 0.0,
                "max_drift_instr": -1,
                "max_drift_val": 0,
                "max_drift_pc": 0,
            }
            continue

        our_first = our_instrs[0].cycle
        st_first = st_instrs[0].cycle

        our_total = our_instrs[-1].cycle - our_first if len(our_instrs) > 1 else 0
        st_total = st_instrs[-1].cycle - st_first if len(st_instrs) > 1 else 0

        drift = our_total - st_total
        pct = (drift / st_total * 100) if st_total else 0.0

        # Find instruction with largest per-instruction drift
        max_drift_instr = -1
        max_drift_val = 0
        max_drift_pc = 0
        min_len = min(len(our_instrs), len(st_instrs))
        for i in range(min_len):
            o_rel = our_instrs[i].cycle - our_first
            s_rel = st_instrs[i].cycle - st_first
            d = abs(o_rel - s_rel)
            if d > abs(max_drift_val):
                max_drift_val = o_rel - s_rel
                max_drift_instr = i
                max_drift_pc = our_instrs[i].pc

        result[w] = {
            "our_count": len(our_instrs),
            "st_count": len(st_instrs),
            "our_total": our_total,
            "st_total": st_total,
            "drift": drift,
            "pct": pct,
            "max_drift_instr": max_drift_instr,
            "max_drift_val": max_drift_val,
            "max_drift_pc": max_drift_pc,
        }

    return result


def per_warp_event_counts(events: list[TraceEvent]) -> dict[int, dict[int, int]]:
    """Per-warp counts of each event type."""
    result = defaultdict(lambda: defaultdict(int))
    for e in events:
        result[e.warp_id][e.event_type] += 1
    return dict(result)


def find_first_divergence(our_events: list[TraceEvent], st_events: list[TraceEvent]):
    """Find the first point where the two traces diverge in INSTR_EXEC sequence."""
    our_exec = filter_events(our_events, 2)
    st_exec = filter_events(st_events, 2)

    min_len = min(len(our_exec), len(st_exec))
    for i in range(min_len):
        o = our_exec[i]
        s = st_exec[i]
        if o.pc != s.pc or o.warp_id != s.warp_id:
            return i, o, s
    if len(our_exec) != len(st_exec):
        return min_len, None, None
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Compare instruction traces between simulators")
    parser.add_argument("oursim_trace", help="Trace file from our simulator")
    parser.add_argument("simtight_trace", help="Trace file from SIMTight")
    parser.add_argument("--benchmark", "-b", default="(unknown)", help="Benchmark name for header")
    parser.add_argument("--max-schedule-diff", type=int, default=20,
                        help="Max scheduling divergences to show")
    args = parser.parse_args()

    print(f"=== Trace Comparison: {args.benchmark} ===\n")

    print("Loading traces...")
    our_events_raw = parse_trace(args.oursim_trace)
    st_events_raw = parse_trace(args.simtight_trace)
    print(f"  OurSim:   {len(our_events_raw)} events (raw)")
    print(f"  SIMTight: {len(st_events_raw)} events (raw)")

    # Reduce to warp-level for comparison
    our_events = to_warp_level(our_events_raw)
    st_events = to_warp_level(st_events_raw)
    print(f"  OurSim:   {len(our_events)} events (warp-level)")
    print(f"  SIMTight: {len(st_events)} events (warp-level)")

    # --- Event counts ---
    print("\n--- Event Counts ---")
    our_counts = event_counts(our_events)
    st_counts = event_counts(st_events)
    all_types = sorted(set(our_counts.keys()) | set(st_counts.keys()))
    print(f"  {'Event':<16} {'OurSim':>10} {'SIMTight':>10} {'Diff':>10}")
    for t in all_types:
        name = EVENT_NAMES.get(t, f"TYPE_{t}")
        oc = our_counts.get(t, 0)
        sc = st_counts.get(t, 0)
        diff = oc - sc
        print(f"  {name:<16} {oc:>10} {sc:>10} {diff:>+10}")

    # --- Cycle range ---
    print("\n--- Cycle Range ---")
    if our_events and st_events:
        our_min_c = min(e.cycle for e in our_events)
        our_max_c = max(e.cycle for e in our_events)
        st_min_c = min(e.cycle for e in st_events)
        st_max_c = max(e.cycle for e in st_events)
        print(f"  OurSim:   cycle {our_min_c} to {our_max_c} (span {our_max_c - our_min_c})")
        print(f"  SIMTight: cycle {st_min_c} to {st_max_c} (span {st_max_c - st_min_c})")

    # --- First divergence ---
    print("\n--- First Scheduling Divergence ---")
    idx, our_evt, st_evt = find_first_divergence(our_events, st_events)
    if idx is None:
        print("  No divergence found — instruction sequences match exactly!")
    elif our_evt is None:
        print(f"  Traces match for {idx} instructions, then one trace ends")
    else:
        print(f"  At instruction #{idx}:")
        print(f"    OurSim:   cycle={our_evt.cycle}, PC=0x{our_evt.pc:08x}, warp={our_evt.warp_id}")
        print(f"    SIMTight: cycle={st_evt.cycle}, PC=0x{st_evt.pc:08x}, warp={st_evt.warp_id}")
        if our_evt.pc != st_evt.pc:
            print(f"    -> Different PCs!")
        if our_evt.warp_id != st_evt.warp_id:
            print(f"    -> Different warps scheduled!")

    # --- Scheduling divergences ---
    print(f"\n--- Scheduling Divergences (first {args.max_schedule_diff}) ---")
    divs = compare_scheduling(our_events, st_events, args.max_schedule_diff)
    if not divs:
        print("  No scheduling divergences in instruction sequence.")
    else:
        print(f"  {'Instr#':>8} {'Our cyc':>10} {'Our warp':>10} {'ST cyc':>10} {'ST warp':>10}")
        for idx, oc, ow, sc, sw in divs:
            print(f"  {idx:>8} {oc:>10} {ow:>10} {sc:>10} {sw:>10}")

    # --- Per-warp drift ---
    print("\n--- Per-Warp Cycle Drift (INSTR_EXEC) ---")
    drift = per_warp_drift(our_events, st_events)
    if drift:
        print(f"  {'Warp':>6} {'Our#':>8} {'ST#':>8} {'OurSpan':>10} {'STSpan':>10} {'Drift':>8} {'Pct':>8}")
        for w in sorted(drift.keys()):
            d = drift[w]
            pct_str = f"{d['pct']:+.1f}%"
            print(f"  {w:>6} {d['our_count']:>8} {d['st_count']:>8} "
                  f"{d['our_total']:>10} {d['st_total']:>10} {d['drift']:>+8} {pct_str:>8}")

        # Largest per-instruction drift
        print("\n--- Largest Per-Instruction Drift ---")
        top_drift = sorted(drift.items(), key=lambda x: abs(x[1]["max_drift_val"]), reverse=True)[:10]
        for w, d in top_drift:
            if d["max_drift_instr"] >= 0:
                print(f"  Warp {w}, instr #{d['max_drift_instr']} "
                      f"(PC=0x{d['max_drift_pc']:08x}): drift={d['max_drift_val']:+d} cycles")

    # --- Per-warp event type distribution ---
    print("\n--- Per-Warp Event Distribution ---")
    our_warp_counts = per_warp_event_counts(our_events)
    st_warp_counts = per_warp_event_counts(st_events)
    all_warps = sorted(set(our_warp_counts.keys()) | set(st_warp_counts.keys()))
    # Show types present in the data
    types_present = sorted(set(t for wc in [our_warp_counts, st_warp_counts]
                               for counts in wc.values() for t in counts.keys()))
    type_names = [EVENT_NAMES.get(t, f"T{t}") for t in types_present]

    header = f"  {'Warp':>6} " + " ".join(f"{'Our_'+n:>14} {'ST_'+n:>14}" for n in type_names)
    print(header)
    for w in all_warps:
        oc = our_warp_counts.get(w, {})
        sc = st_warp_counts.get(w, {})
        parts = [f"  {w:>6}"]
        for t in types_present:
            parts.append(f" {oc.get(t, 0):>14} {sc.get(t, 0):>14}")
        print("".join(parts))

    print()


if __name__ == "__main__":
    main()
