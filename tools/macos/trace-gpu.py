#!/usr/bin/env python3
"""GPU time of a Metal System Trace of the game, by channel, command buffer and encoder.

Record while the game runs (pid of the title's wine process):

    xcrun xctrace record --template 'Metal System Trace' --attach <pid> --time-limit 20s --output gpu.trace

Then:

    python3 -I tools/macos/trace-gpu.py gpu.trace          # exports metal-gpu-intervals.xml next to it
    python3 -I tools/macos/trace-gpu.py metal-gpu-intervals.xml

D3DMetal labels no encoder with the host's names, so encoders are known by
their kind and their index in their command buffer ("Compute Command 4").
Match the busiest against the log's `census` lines (VICTORY_DISPATCH_CENSUS=1).
"""
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

LABEL = re.compile(r"Command Buffer (\d+):(\w+) Command (\d+)")


def export(trace):
    out = os.path.join(os.path.dirname(os.path.abspath(trace)), "metal-gpu-intervals.xml")
    if not os.path.exists(out):
        xpath = '//trace-toc[1]/run[1]/data[1]/table[@schema="metal-gpu-intervals"]'
        with open(out, "wb") as f:
            subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath], stdout=f, check=True)
    return out


def rows(path):
    cols, ids = [], {}

    def get(el):
        ref = el.get("ref")
        if ref is not None:
            return ids.get(ref, (None, None))
        v = (el.text, el.get("fmt"))
        if el.get("id") is not None:
            ids[el.get("id")] = v
        for c in el.iter():
            if c is not el and c.get("id") is not None:
                ids[c.get("id")] = (c.text, c.get("fmt"))
        return v

    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "mnemonic":
            cols.append(el.text)
        elif el.tag == "row":
            row = {c: get(k) for c, k in zip(cols, list(el))}
            el.clear()
            yield row


def fmt(row, key, default=""):
    return (row.get(key) or (None, default))[1] or default


def main():
    path = sys.argv[1]
    if path.endswith(".trace"):
        path = export(path)
    proc = sys.argv[2] if len(sys.argv) > 2 else "wine"
    intervals = []
    by_channel = defaultdict(lambda: [0, 0])
    by_encoder = defaultdict(lambda: [0, 0, 0])  # (kind, index) -> ns, count, max ns
    by_buffer = defaultdict(lambda: [0, 0, None, 0])  # buffer -> ns, encoders, start, compute ns
    singles = []
    for row in rows(path):
        try:
            start, dur = int(row["start"][0]), int(row["duration"][0])
        except (KeyError, TypeError, ValueError):
            continue
        if proc not in fmt(row, "process") or fmt(row, "event-depth") != "0":
            continue
        channel, label = fmt(row, "channel-name", "?"), fmt(row, "event-label")
        intervals.append((start, start + dur))
        by_channel[channel][0] += dur
        by_channel[channel][1] += 1
        m = LABEL.search(label)
        if not m:
            continue
        kind, index = m.group(2), int(m.group(3))
        cb = fmt(row, "metal-command-buffer-id", m.group(1))
        e = by_encoder[(kind, index)]
        e[0] += dur
        e[1] += 1
        e[2] = max(e[2], dur)
        b = by_buffer[cb]
        b[0] += dur
        b[1] += 1
        b[2] = start if b[2] is None else min(b[2], start)
        if kind == "Compute":
            b[3] += dur
        singles.append((dur, start, channel, label.split("  (")[0]))
    if not intervals:
        sys.exit(f"no GPU intervals of a process named like {proc!r}")
    intervals.sort()
    busy, (cs, ce) = 0, intervals[0]
    t0, t1 = cs, max(e for _, e in intervals)
    for s, e in intervals[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = t1 - t0

    def ms(ns):
        return ns / 1e6

    print(f"span {span / 1e9:.2f} s, GPU busy {busy / 1e9:.2f} s ({100 * busy / span:.0f} %), idle {100 - 100 * busy / span:.0f} %")
    print("\n== by channel (summed per channel; channels overlap)")
    for ch, (d, n) in sorted(by_channel.items(), key=lambda x: -x[1][0]):
        print(f"{d / 1e9:8.2f} s {n:8d} intervals  {ch}")
    big = sorted(by_buffer.items(), key=lambda x: -x[1][0])
    print(f"\n== command buffers: {len(big)}; the 10 busiest (ms, encoders, ms compute, start s)")
    for cb, (d, n, s, comp) in big[:10]:
        print(f"{ms(d):9.1f} ms {n:6d} enc {ms(comp):9.1f} ms compute  at {(s - t0) / 1e9:7.3f} s  {cb}")
    print("\n== by encoder kind and index in its command buffer (s, count, avg ms, max ms)")
    for (kind, index), (d, n, mx) in sorted(by_encoder.items(), key=lambda x: -x[1][0])[:30]:
        print(f"{d / 1e9:8.2f} s {n:7d} {ms(d) / n:8.2f} {ms(mx):8.2f}  {kind} Command {index}")
    print("\n== compute encoder lengths (count, s)")
    comp = [d for d, _, ch, _ in singles if ch == "Compute"]
    lo = 0
    for hi in [0.05e6, 0.5e6, 2e6, 5e6, 10e6, 20e6, float("inf")]:
        sel = [d for d in comp if lo <= d < hi]
        print(f"  {ms(lo):6.2f}-{ms(hi):6.2f} ms: {len(sel):7d}  {sum(sel) / 1e9:7.2f} s")
        lo = hi
    print("\n== the 25 longest encoders (ms, at s, channel, label)")
    for d, s, ch, label in sorted(singles, reverse=True)[:25]:
        print(f"{ms(d):8.2f} ms at {(s - t0) / 1e9:7.3f} s  {ch:9s} {label}")


if __name__ == "__main__":
    main()
