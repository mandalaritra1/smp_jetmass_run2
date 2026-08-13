# notebooks/

Interactive scratchpads. Run them from the repository root.

## `pairsplit_validation_2018.ipynb`

Data/MC validation of the adopted `pair_split` hadronic selection, 2018 only.
62 cells: the original scaffold (variables, binning, empty `hist.Hist`
placeholders, one example figure) followed by four numbered sections and 22
figures.

| section | figures | content | input |
|---|---|---|---|
| 1 | Fig 1.1 - 1.10 | per-jet kinematics: $p_T$, $\eta$, $\phi$, $m_{\mathrm{SD}}$, ungroomed mass | pair_split skim tables |
| 2 | Fig 2.1 - 2.6 | $\log(\rho^2)$ per measured-jet $p_T$ bin, with the systematic envelope | `minimal_rho` all_syst |
| 3 | Fig 3.1 - 3.2 | $m_{\mathrm{SD}}$, $p_T$-inclusive, with the systematic envelope | `minimal_rho` all_syst |
| 4 | Fig 4.1 - 4.4 | boundary distance $D$ of the leading dijet pair | pair_split skim tables |

### Kernel

`Python (GluonJetMass .venv)` &rarr; `~/Projects/GluonJetMass/.venv`
(hist 2.10.1, mplhep 1.2.0, numpy 2.4.6). Registered as the kernelspec
`gluonjetmass`; re-register with

```bash
~/Projects/GluonJetMass/.venv/bin/python -m ipykernel install --user \
    --name gluonjetmass --display-name "Python (GluonJetMass .venv)"
```

The repository's own `.venv` has identical versions of everything the notebook
imports and works just as well.

To re-execute the whole notebook from the command line:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=gluonjetmass \
    --ExecutePreprocessor.timeout=1800 \
    notebooks/pairsplit_validation_2018.ipynb
```

### Caching and `FORCE_RECOMPUTE`

Each of the four *fill* cells pickles its finished `hist.Hist` objects into
`notebooks/.cache_pairsplit_2018/` and reloads them on the next run, so a
cold run takes ~16 s and a warm one ~9 s.

| file | written by |
|---|---|
| `kinematics_hists.pkl` | section-1 fill cell |
| `rho_variations.pkl` | section-2 loading cell |
| `msd_variations.pkl` | section-3 loading cell |
| `boundary_hists.pkl` | section-4 fill cell |

`FORCE_RECOMPUTE` is defined once, in the *Cache and the recompute switch*
cell. Set it to `True` (or delete the file named at the top of the fill cell)
after changing a binning, a column, or the selection. The `s*.npz` files in the
same directory are unrelated leftovers from an earlier scripted pass and are
not read by the notebook.

### Style rules for this notebook

* Flat, linear cells. **No helper functions for drawing.** The only imports
  from `scripts/validate_pairsplit.py` are the I/O helpers `stream_table`,
  `hist_variations`, `syst_total` and `area_norm`.
* Every tweakable quantity is an explicit variable in a `# ---- knobs`
  block at the top of its own cell, with a comment saying what it does.
* Histograms are `hist.Hist` with named axes and a `source` StrCategory axis
  (`mc` / `data`), weighted storage. Figures are drawn with
  `hist_data.plot_ratio(hist_mc, ...)`, one figure per cell.
* Never re-apply a cross section: the skim `weight` column and the stored
  histograms already carry `xs*lumi*1000/sumw` for MC, and the trigger prescale
  weight for data. See the module docstring of `scripts/validate_pairsplit.py`.

### Batch / deck version

The same comparisons, plus the yield ratio, the 2D $\Delta\phi$ map, the
orthogonality closure, the $|y|$ edge move and the fake/miss fractions, are
produced as PNG + PDF files by

```bash
python scripts/validate_pairsplit.py all --outdir DIR
```

That script is the one to extend when a figure needs to land in the AN or a
deck; the notebook is for interactive work.

## `run_analysis.ipynb`

Driver for local processor runs. See `README.md` at the repository root.
