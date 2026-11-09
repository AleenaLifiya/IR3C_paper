#!/usr/bin/env python
"""Shared data layer for the IR3C MedIA manuscript.

Every number in the paper is computed from the stored per-volume CSVs through this module.
Nothing is typed by hand. Source files are opened read-only.

Conventions (applied to every table and figure)
-----------------------------------------------
* Region: foreground voxels of the MISSING slices only (``region == "missing"``).
* Aggregation: average the cells of a row within a subject FIRST, then report mean and
  standard deviation across subjects. Repeated masks of one subject are never treated as
  independent observations.
* Paired statistics: per-subject differences; mean, t-based 95% CI, Wilcoxon signed-rank,
  per-subject win rate, Holm correction within a table.
* Checkpoint: IR3C is the BUDGET-MATCHED model everywhere (BraTS ``IR3_matched`` at
  step 12,510; IXI and OAIZIB fine-tuned FROM that checkpoint, ``*_matchedinit``).
  Full-budget IR3C appears only in the budget-sensitivity appendix table, and in the
  downstream segmentation experiment, whose archived reconstructions predate the matched
  checkpoint (stated in the caption).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/home/aleena/xdiffusion"
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
TABLES = os.path.join(PAPER, "tables")
FIGS = os.path.join(PAPER, "figs")

DATASETS = ["BraTS", "OAIZIB", "IXI"]
DS_LABEL = {"BraTS": "BraTS-GLI", "OAIZIB": "OAIZIB", "IXI": "IXI-tiny"}
DS_N = {"BraTS": 88, "OAIZIB": 103, "IXI": 100}
FAMILIES = ["U0", "U1", "U2", "U3", "U4", "U5", "U6"]
NONUNIFORM = FAMILIES[1:]
TRAIN_R = [2, 4, 8]

SRC = {
    "BraTS": os.path.join(ROOT, "IR3C_more_comparisons/metrics/results_combined.csv"),
    "OAIZIB": os.path.join(ROOT, "IR3C_experiments_V2/stage2_finetune/OAIZIB/metrics/"
                                 "results_combined_v2.csv"),
    "IXI": os.path.join(ROOT, "IR3C_experiments_V2/stage2_finetune/IXI/metrics/"
                              "results_combined_v2.csv"),
}
# Budget-matched IR3C for the transfer datasets. Same protocol as the V2 CSVs: the linear
# rows of these files reproduce the V2 linear rows exactly (max |dPSNR| = 0.0 dB).
MATCHED = {
    "OAIZIB": os.path.join(ROOT, "runs/IR3C_OAIZIB_09Sept_fulltrain_matchedinit/"
                                 "IR3C_OAIZIB_09Sept_fulltrain_matchedinit_results_per_volume.csv"),
    "IXI": os.path.join(ROOT, "runs/IR3C_IXItiny_08Sept_matchedinit/"
                              "IR3C_IXItiny_08Sept_matchedinit_results_per_volume.csv"),
}

# display name -> method selector. ``per_R`` picks one checkpoint per factor.
def method_specs(ds):
    if ds == "BraTS":
        return [("Nearest", {"m": "nearest"}),
                ("Linear", {"m": "linear"}),
                ("Cubic spline", {"m": "cubic"}),
                ("Bicubic", {"m": "bicubic"}),
                ("DCSRN", {"m": "DCSRN_mixed"}),
                ("ArSSR", {"m": "ArSSR_slowlr"}),
                ("I$^3$Net", {"m": "I3Net"}),
                ("SAINT", {"m": "SAINT"}),
                ("IR3C", {"m": "IR3_matched"})]
    mix = ({2: "DCSRNmix_x2_besttrained", 4: "DCSRNmix_x4", 8: "DCSRNmix_x8"}
           if ds == "OAIZIB" else
           {2: "DCSRNmix_x2_besttrained", 4: "DCSRNmix_x4_besttrained",
            8: "DCSRNmix_x8_besttrained"})
    return [("Nearest", {"m": "nearest"}),
            ("Linear", {"m": "linear"}),
            ("Cubic spline", {"m": "cubic"}),
            ("Bicubic", {"m": "bicubic"}),
            ("DCSRN", {"per_R": mix}),
            ("ArSSR", {"m": "ArSSR_slowlr"}),
            ("I$^3$Net", {"m": "I3Net"}),
            ("SAINT", {"m": "SAINT_rfn"}),
            ("IR3C", {"m": "IR3C_matched"})]


UNIFORM_ONLY = {"I$^3$Net", "SAINT"}          # undefined off a regular grid
METRICS = ["psnr_masked", "ssim", "nmse", "mae"]

_cache: dict = {}


def load(ds):
    """All missing-region per-volume rows of one dataset, with matched IR3C appended."""
    if ds in _cache:
        return _cache[ds]
    d = pd.read_csv(SRC[ds], low_memory=False)
    d = d[d.region == "missing"].copy()
    if ds in MATCHED:
        m = pd.read_csv(MATCHED[ds], low_memory=False)
        m = m[(m.region == "missing") & (m.method == f"IR3C_{ds}")].copy()
        m["method"] = "IR3C_matched"
        d = pd.concat([d, m], ignore_index=True, sort=False)
    _cache[ds] = d
    return d


def select(ds, spec, families=FAMILIES, Rs=TRAIN_R):
    d = load(ds)
    if "m" in spec:
        sub = d[d.method == spec["m"]]
    else:
        sub = pd.concat([d[(d.method == nm) & (d.R == R)] for R, nm in spec["per_R"].items()])
    return sub[sub.mask_family.isin(families) & sub.R.isin(Rs)]


def per_subject(ds, spec, families=FAMILIES, Rs=TRAIN_R, metric="psnr_masked"):
    """Per-subject mean of `metric` over the selected cells (Series indexed by subject)."""
    sub = select(ds, spec, families, Rs)
    if sub.empty:
        return None
    return sub.groupby("subject")[metric].mean()


def mean_sd(s):
    return (np.nan, np.nan) if s is None else (float(s.mean()), float(s.std(ddof=1)))


def paired(a, b):
    """a - b over common subjects: mean, CI half-width, Wilcoxon p, win rate, n."""
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    diff = (j.iloc[:, 0] - j.iloc[:, 1]).values
    n = len(diff)
    m = float(diff.mean())
    hw = float(stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n))
    try:
        p = float(stats.wilcoxon(diff).pvalue)
    except ValueError:
        p = 1.0
    return dict(mean=m, lo=m - hw, hi=m + hw, p=p, win=float((diff > 0).mean()), n=n)


def holm(pvals):
    """Holm-adjusted p-values, in input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ slice rejection
SR_ROOT = os.path.join(ROOT, "runs/IR3C_SliceReject_all")
SR_METHODS = ["linear", "bicubic", "ArSSR", "DCSRNmix", "I3Net_impute", "I3Net_subset",
              "SAINT_impute", "SAINT_subset", "IR3C"]
SR_LABEL = {"linear": "Linear", "bicubic": "Bicubic", "ArSSR": "ArSSR",
            "DCSRNmix": "DCSRN", "I3Net_impute": "I$^3$Net + imputation",
            "I3Net_subset": "I$^3$Net, uniform subset",
            "SAINT_impute": "SAINT + imputation", "SAINT_subset": "SAINT, uniform subset",
            "IR3C": "IR3C"}


def load_sr(ds):
    key = ("sr", ds)
    if key not in _cache:
        p = os.path.join(SR_ROOT, ds, "metrics.csv")
        _cache[key] = pd.read_csv(p) if os.path.exists(p) else None
    return _cache[key]


def sr_per_subject(ds, method, R, pattern, p, metric="psnr"):
    d = load_sr(ds)
    if d is None:
        return None
    pat = "none" if p == 0 else pattern
    s = d[(d.method == method) & (d.base_R == R) & (d.pattern == pat) & np.isclose(d.p, p)]
    return None if s.empty else s.set_index("subject")[metric]


# ------------------------------------------------------------------ LaTeX helpers
def f2(v, dec=2):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{dec}f}"


def signed(v, dec=2):
    if abs(v) < 0.5 * 10 ** (-dec):
        return f"{0:.{dec}f}"
    s = f"{v:+.{dec}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s.replace("+", "$+$")


APPENDIX = {"tab_detail.tex", "tab_lpips.tex", "tab_rejection_absolute.tex", "tab_budget.tex"}
TIGHT = {"tab_r16.tex", "tab_anchor.tex", "tab_backbone.tex", "tab_formulation.tex",
         "tab_heldout.tex", "tab_cost.tex", "tab_budget.tex", "tab_lpips.tex"}


def fix_neg_zero(text):
    """'$-$0.00' and '-0.00' print as 0.00 (a rounded zero has no sign)."""
    return text.replace("$-$0.00]", "0.00]").replace("$-$0.00,", "0.00,").replace(
        "[$-$0.00", "[0.00")


def write(name, text):
    """Write a table. Narrow tables get footnotesize and reduced column padding."""
    if name in TIGHT:
        text = text.replace("\\small\n", "\\footnotesize\\setlength{\\tabcolsep}{3pt}\n", 1)
    text = fix_neg_zero(text)
    if name in APPENDIX:                  # one-column appendix: keep tables after their text
        text = text.replace("table*}", "table}").replace("[t]", "[pos=!htbp]")
    os.makedirs(TABLES, exist_ok=True)
    p = os.path.join(TABLES, name)
    with open(p, "w") as fh:
        fh.write(text)
    print("  tables/" + name)
