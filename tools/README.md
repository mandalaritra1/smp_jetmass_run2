# Production validation tools

`tools/` contains commands run on real analysis inputs or outputs. `tests/`
contains automated checks of their behavior using controlled inputs.

| Tool | Purpose | Automated tests |
|---|---|---|
| [check_das_event_counts.py](check_das_event_counts.py) | Compare processed event counts in production pickles with the expected dataset counts. | [test_check_das_event_counts.py](../tests/test_check_das_event_counts.py) |
| [compute_sumw_qcd_mg.py](compute_sumw_qcd_mg.py) | Recompute legacy LHE-weight denominators from NanoAOD files. Current hadronic processors accumulate genWeight separately. | No dedicated test module. |

Run the event-count check from the repository root:

```bash
python tools/check_das_event_counts.py /path/to/production.pkl
```

The default backend uses cached counts first, then lxplus SSH, DBS and McM.
The SSH backend uses the existing authenticated `lxplus` master without
prompting for a password or OTP. Its explicit option is `--backend lxplus`.

Run the checker's automated tests:

```bash
PYTHONPATH=. python -m pytest -q tests/test_check_das_event_counts.py
```

These tests use local filesets and mocked SSH responses; they neither contact
CERN nor validate a real production. A passing test suite does not replace
running the event-count check on the production outputs.
