#!/usr/bin/env python3
"""Compares the A and B runs of tools/macos/bench-ab.sh.

Reads <dir>/A*.json and <dir>/B*.json (VICTORY_BENCH_OUT reports), prints
each run's median frame time, each side's median, the difference B - A and
a bootstrap 95 % interval of it. An interval that holds 0 means the runs
cannot tell A from B.
"""
import glob
import json
import os
import random
import statistics
import sys


def side(d, name):
    out = []
    for path in sorted(glob.glob(os.path.join(d, f"{name}[0-9]*.json"))):
        with open(path) as f:
            r = json.load(f)
        if r.get("device_removed"):
            print(f"  {os.path.basename(path)}: device removed, left out")
            continue
        out.append((os.path.basename(path), r["frame_ms"]["median"], r["frame_ms"]["p95"], r["world_frames"]))
    return out


def interval(a, b, rounds=10000, seed=1):
    rng = random.Random(seed)
    diffs = sorted(
        statistics.median(rng.choices(b, k=len(b))) - statistics.median(rng.choices(a, k=len(a)))
        for _ in range(rounds)
    )
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds)]


def main():
    d = sys.argv[1]
    sets = os.path.join(d, "sets.txt")
    if os.path.exists(sets):
        print(open(sets).read().rstrip())
    a, b = side(d, "A"), side(d, "B")
    for name, med, p95, n in a + b:
        print(f"  {name}: median {med:7.1f} ms, p95 {p95:7.1f} ms, {n} frames")
    if len(a) < 2 or len(b) < 2:
        sys.exit("need at least two runs a side")
    ma, mb = [m for _, m, _, _ in a], [m for _, m, _, _ in b]
    xa, xb = statistics.median(ma), statistics.median(mb)
    lo, hi = interval(ma, mb)
    print(f"A median {xa:.1f} ms ({1000 / xa:.2f} fps), B median {xb:.1f} ms ({1000 / xb:.2f} fps)")
    print(f"B - A {xb - xa:+.1f} ms ({100 * (xb - xa) / xa:+.1f} %), 95 % interval {lo:+.1f} to {hi:+.1f} ms")
    if lo <= 0 <= hi:
        print("the interval holds 0: no difference these runs can show")


if __name__ == "__main__":
    main()
