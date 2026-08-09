#!/usr/bin/env python3
"""Fail if the pair_split routine regresses to legacy physics logic."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    code_only = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    code_only = re.sub(r"//.*", "", code_only)

    required = {
        "ycut_2p4": "static constexpr double YCUT = 2.4;",
        "lead_pt_200": "static constexpr double LEAD_PT_MIN = 200.0;",
        "third_pt_185": "static constexpr double THIRD_PT_MIN = 185.0;",
        "pair_boundary": "PI - PAIR_RECOIL_SCALE / ptAverage",
        "dijet_strict_gt": "dphi12 > pairBoundary",
        "trijet_strict_lt": "dphi12 < pairBoundary",
        "min_dphi": "minPairwiseDphi > 1.0",
        "softdrop": "fastjet::contrib::SoftDrop softDrop(0.0, 0.1)",
        "rho_definition": "2.0 * std::log10(groomedMass / (pt * JET_R))",
        "weighted_yoda_fill": "rhoG, eventWeight",
    }
    forbidden_patterns = {
        "legacy_fixed_dphi": r"dphi12\s*>\s*2(?:\.0)?\s*[;)]",
        "legacy_asymmetry": r"asym|asymm",
        "legacy_trijet_veto": r"vetoTrijet|trijet-priority veto",
        "mg10_cut": r"m_g\s*>\s*10|groomedMass\s*>\s*10",
    }
    missing = [name for name, text in required.items() if text not in source]
    forbidden = [
        name for name, pattern in forbidden_patterns.items()
        if re.search(pattern, code_only, flags=re.IGNORECASE)
    ]
    result = {"missing": missing, "forbidden": forbidden,
              "status": "pass" if not missing and not forbidden else "fail"}
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
