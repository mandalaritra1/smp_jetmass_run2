#!/usr/bin/env python3
"""Generate the five HT-binned Herwig7 DY fragments from the inclusive one by
swapping ONLY the gridpack path (brief §1, constraint #2 byte-for-byte).

Inputs:
  inputs/inclusive_fragment.py   the existing inclusive Herwig DY fragment
  inputs/gridpacks.txt           lines:  <BIN> <cvmfs .tar.xz path>

For each bin it copies the inclusive fragment, replaces the single existing
gridpack .tar.xz path with the bin's path, writes
  Configuration/GenProduction/python/Herwig7_DY_<BIN>_CH3_cff.py
and prints a unified diff so you can PROVE only that one line changed.
"""
import os, re, sys, difflib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "inputs", "inclusive_fragment.py")
MAP  = os.path.join(ROOT, "inputs", "gridpacks.txt")
OUTD = os.path.join(ROOT, "Configuration", "GenProduction", "python")

GRIDPACK_RE = re.compile(r'(/cvmfs/\S+?\.tar\.xz)')

def main():
    with open(SRC) as f:
        base = f.read()

    found = GRIDPACK_RE.findall(base)
    uniq = sorted(set(found))
    if len(uniq) != 1:
        sys.exit("ERROR: expected exactly ONE gridpack .tar.xz path in the "
                 "inclusive fragment, found %d: %s\nFix the regex or the input."
                 % (len(uniq), uniq))
    old_path = uniq[0]
    print("inclusive gridpack path: %s\n" % old_path)

    os.makedirs(OUTD, exist_ok=True)
    with open(MAP) as f:
        rows = [l.split() for l in f if l.strip() and not l.startswith("#")]

    for bin_name, new_path in rows:
        if not new_path.endswith(".tar.xz"):
            sys.exit("ERROR: %s path does not end in .tar.xz: %s" % (bin_name, new_path))
        new = base.replace(old_path, new_path)
        out = os.path.join(OUTD, "Herwig7_DY_%s_CH3_cff.py" % bin_name)
        with open(out, "w") as f:
            f.write(new)
        diff = list(difflib.unified_diff(
            base.splitlines(True), new.splitlines(True),
            fromfile="inclusive", tofile=bin_name))
        changed_lines = [d for d in diff if d.startswith(('+', '-')) and not d.startswith(('+++', '---'))]
        print("=== %s -> %s  (%d changed lines)" % (bin_name, out, len(changed_lines)))
        sys.stdout.writelines(changed_lines)
        if len(changed_lines) != 2:
            print("  !! WARNING: more than the single gridpack line changed — INSPECT.")
        print()

if __name__ == "__main__":
    main()
