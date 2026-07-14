#!/usr/bin/env python3
"""Split a Les Houches event file into N chunks of disjoint events, so a 50k-event
shower can run as N parallel jobs (each chunk -> one job -> one yoda; inverse-var
average the N yodas back to the full-stats result).

Each chunk = the shared <header>+<init> preamble (same cross section) + its disjoint
slice of <event> blocks + </LesHouchesEvents>. Disjoint events is the KEY: every job
showers DIFFERENT hard events, so the combined sample has the full hard-event stats
(not just N re-showers of the same 2k events).

Usage: split_lhe.py <in.lhe> <N> <out_prefix>   -> <out_prefix>_0.lhe .. _{N-1}.lhe
"""
import sys

infile, N, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = open(infile).read().splitlines(keepends=True)

# preamble = everything through </init>; footer = </LesHouchesEvents>
init_end = next(i for i, l in enumerate(lines) if l.strip() == "</init>")
preamble = lines[: init_end + 1]

# collect <event>...</event> blocks
events, i, n = [], init_end + 1, len(lines)
while i < n:
    if lines[i].lstrip().startswith("<event>"):
        j = i
        while j < n and not lines[j].lstrip().startswith("</event>"):
            j += 1
        events.append("".join(lines[i : j + 1]))
        i = j + 1
    else:
        i += 1

tot = len(events)
per = -(-tot // N)  # ceil
written = 0
for k in range(N):
    chunk = events[k * per : (k + 1) * per]
    if not chunk:
        break
    with open(f"{prefix}_{k}.lhe", "w") as fh:
        fh.writelines(preamble)
        fh.write("".join(chunk))
        fh.write("</LesHouchesEvents>\n")
    written += len(chunk)
print(f"split {tot} events -> {min(N, -(-tot//per))} chunks of ~{per} (wrote {written} total)")
