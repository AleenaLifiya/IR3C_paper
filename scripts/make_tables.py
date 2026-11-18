#!/usr/bin/env python
"""Write every LaTeX table of the IR3C MedIA manuscript from the stored per-volume CSVs.

Run:  /home/aleena/miniforge3/envs/xdiff/bin/python scripts/make_tables.py

Completed experiments are computed here. Experiments that have NOT been run are written as
tables whose cells read \\tbd on a shaded row, so the manuscript shows exactly what is
missing. No pending cell is ever filled with an estimate.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TBD = r"\tbd"


def bold_idx(vals, higher=True):
    arr = np.array([np.nan if v is None else v for v in vals], float)
    if np.all(np.isnan(arr)):
        return -1
    return int(np.nanargmax(arr) if higher else np.nanargmin(arr))


def star(p):
    """Significance marker after Holm correction."""
    return r"$^{\ast}$" if p < 0.05 else ""


# =========================================================================== Table 1
def tab_datasets():
    """Dataset facts. Grids and spacings were read from the NIfTI headers; splits from
    the run manifests."""
    t = r"""\begin{table*}[t]
\caption{Datasets and subject-level partitions. ``Reconstruction axis'' is the axis along
which slices are removed; its number of slices $D$ and spacing are given. OAIZIB and
IXI-tiny models are fine-tuned on their own training partitions. Test subjects are never
used for training, early stopping or checkpoint selection.}
\label{tab:datasets}
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllllrrr@{}}
\toprule
Dataset & Anatomy, contrast & Grid (voxels) & Spacing (mm) & Reconstruction axis & Train & Val. & Test \\
\midrule
BraTS-GLI & Brain, T2-FLAIR & $240\times240\times153^{a}$ & $1.0\times1.0\times1.0$ & axial; $D=153$, 1.0\,mm & 1251 & 6 & 88 \\
OAIZIB & Knee, 3D DESS & $160\times384\times384$ & $0.70\times0.36\times0.36$ & sagittal; $D=160$, 0.70\,mm & 374 & 30 & 103 \\
IXI-tiny & Brain, T1-weighted & $83\times44\times55$ & $2.18\times4.13\times3.95$ & axial; $D=83$, 2.18\,mm$^{b}$ & 436 & 30 & 100 \\
\bottomrule
\end{tabular*}
\par\smallskip
{\footnotesize $^{a}$Central crop of the native 155 axial slices. $^{b}$The reconstruction
axis is the finest axis of this dataset, so $R=2$, 4 and 8 correspond to slice spacings of
4.36, 8.72 and 17.44\,mm.}
\end{table*}
"""
    C.write("tab_datasets.tex", t)


# =========================================================================== Table 2
UNIFORM_ROWS = ["Linear", "Bicubic", "DCSRN", "ArSSR", "I$^3$Net", "SAINT", "IR3C"]
PENDING_BASELINES = [("SA-INR", "Arbitrary-spacing INR baseline (Wang et al., 2024)")]
DIRECT_ROW = "3D U-Net, direct output"


def direct_u0(ds, R, metric):
    """A3_direct (seed 42) on U0 at factor R: the P3 BraTS scoring, or the fine-tuned run on
    OAIZIB / IXI-tiny. None while that run is pending."""
    if ds == "BraTS":
        return C.p3_per_subject("A3_direct", ["U0"], [R], metric)
    d = C.load_a3ft(ds)
    if d is None:
        return None
    sub = d[(d.mask_family == "U0") & (d.R == R)]
    return sub.groupby("subject")[metric].mean()


def tab_uniform():
    """Uniform sampling by factor. PSNR block and SSIM block; IR3C - SAINT paired row."""
    cols = [(ds, R) for ds in C.DATASETS for R in C.TRAIN_R]
    out = {}
    for metric in ("psnr_masked", "ssim"):
        table = {}
        for ds in C.DATASETS:
            specs = dict(C.method_specs(ds))
            for R in C.TRAIN_R:
                for nm in UNIFORM_ROWS:
                    s = C.per_subject(ds, specs[nm], ["U0"], [R], metric)
                    table[(nm, ds, R)] = s
        out[metric] = table

    def block(metric, dec):
        tab = out[metric]
        lines = []
        best = {c: bold_idx([tab[(nm,) + c].mean() for nm in UNIFORM_ROWS]) for c in cols}
        for i, nm in enumerate(UNIFORM_ROWS):
            cells = []
            for c in cols:
                v = tab[(nm,) + c].mean()
                s = f"{v:.{dec}f}"
                cells.append(r"\textbf{" + s + "}" if best[c] == i else s)
            name = r"\textbf{IR3C (ours)}" if nm == "IR3C" else nm
            lines.append(name + " & " + " & ".join(cells) + r" \\")
        dv = [direct_u0(ds, R, metric) for ds, R in cols]
        dcells = [TBD if v is None else f"{v.mean():.{dec}f}" for v in dv]
        lines.append((r"\pendrow " if any(v is None for v in dv) else "") + DIRECT_ROW +
                     r"$^{\dagger}$ & " + " & ".join(dcells) + r" \\")
        for nm, _ in PENDING_BASELINES:
            lines.append(r"\pendrow " + nm + " & " + " & ".join([TBD] * len(cols)) + r" \\")
        return lines

    # paired IR3C - SAINT on PSNR, Holm over the nine columns
    pr = [C.paired(out["psnr_masked"][("IR3C",) + c], out["psnr_masked"][("SAINT",) + c])
          for c in cols]
    padj = C.holm([p["p"] for p in pr])
    dl = [f"{C.signed(p['mean'])}{star(q)}" for p, q in zip(pr, padj)]
    wins = [f"{100 * p['win']:.0f}" for p in pr]

    head = (r"& \multicolumn{3}{c}{BraTS-GLI ($n=88$)} & \multicolumn{3}{c}{OAIZIB ($n=103$)}"
            r" & \multicolumn{3}{c}{IXI-tiny ($n=100$)} \\" "\n"
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" "\n"
            "Method" + " & $R=2$ & $R=4$ & $R=8$" * 3 + r" \\")
    t = [r"\begin{table*}[t]",
         r"\caption{Reconstruction on uniform sampling (U0). Values are means across test "
         r"subjects; standard deviations are given in Appendix~\ref{app:detail}. Only missing "
         r"slices inside the foreground mask are scored. Bold marks the best method per "
         r"column. The paired row reports IR3C minus SAINT in dB; an asterisk marks "
         r"$p<0.05$ (Wilcoxon signed-rank, Holm-corrected over the nine columns). "
         r"$^{\dagger}$The IR3C backbone trained to output the image directly (no anchor; "
         r"\texttt{A3\_direct} of Table~\ref{tab:formulation}, seed 42), fine-tuned on OAIZIB "
         r"and IXI-tiny with the IR3C protocol; it is not ranked. Shaded cells are pending.}",
         r"\label{tab:uniform}",
         r"\small",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}",
         r"\toprule", head, r"\midrule",
         r"\multicolumn{10}{@{}l}{\textit{PSNR (dB) $\uparrow$}} \\"]
    t += block("psnr_masked", 2)
    t += [r"\midrule", r"\multicolumn{10}{@{}l}{\textit{SSIM $\uparrow$}} \\"]
    t += block("ssim", 3)
    t += [r"\midrule",
          r"$\Delta$PSNR, IR3C $-$ SAINT & " + " & ".join(dl) + r" \\",
          r"Subjects favouring IR3C (\%) & " + " & ".join(wins) + r" \\",
          r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    C.write("tab_uniform.tex", "\n".join(t))
    return pr


# =========================================================================== Table 3
FAM_ROWS = ["Linear", "Cubic spline", "Bicubic", "DCSRN", "ArSSR", "IR3C"]


def tab_families():
    cols = C.FAMILIES + ["U1--U6"]
    t = [r"\begin{table*}[t]",
         r"\caption{PSNR (dB) by sampling family, pooled over $R\in\{2,4,8\}$. U0 is "
         r"uniform; U1--U6 are irregular (Section~\ref{sec:families}). I$^3$Net and SAINT "
         r"are not defined off a regular grid and are therefore absent. The last row of each "
         r"block is the paired difference between IR3C and the strongest applicable "
         r"comparator in that column, with its 95\% confidence interval. Bold marks the best "
         r"method per column. Standard deviations are given in Appendix~\ref{app:detail}.}",
         r"\label{tab:families}",
         r"\small",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}",
         r"\toprule",
         "Method & " + " & ".join(c if c != "U1--U6" else r"U1--U6" for c in cols) + r" \\",
         r"\midrule"]
    summary = {}
    for k, ds in enumerate(C.DATASETS):
        specs = dict(C.method_specs(ds))
        grid = {}
        for nm in FAM_ROWS:
            for c in cols:
                fams = C.NONUNIFORM if c == "U1--U6" else [c]
                grid[(nm, c)] = C.per_subject(ds, specs[nm], fams)
        if k:
            t.append(r"\midrule")
        t.append(r"\multicolumn{9}{@{}l}{\textit{" + C.DS_LABEL[ds] + r"}} \\")
        best = {c: bold_idx([grid[(nm, c)].mean() for nm in FAM_ROWS]) for c in cols}
        for i, nm in enumerate(FAM_ROWS):
            cells = []
            for c in cols:
                s = f"{grid[(nm, c)].mean():.2f}"
                cells.append(r"\textbf{" + s + "}" if best[c] == i else s)
            name = r"\textbf{IR3C (ours)}" if nm == "IR3C" else nm
            t.append(name + " & " + " & ".join(cells) + r" \\")
        t.append(r"\pendrow SA-INR & " + " & ".join([TBD] * len(cols)) + r" \\")
        dcells = []
        for c in cols:
            others = [nm for nm in FAM_ROWS if nm != "IR3C"]
            ref = max(others, key=lambda nm: grid[(nm, c)].mean())
            p = C.paired(grid[("IR3C", c)], grid[(ref, c)])
            dcells.append(f"{C.signed(p['mean'])}")
            summary[(ds, c)] = (ref, p)
        t.append(r"$\Delta$ vs.\ best comparator & " + " & ".join(dcells) + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    C.write("tab_families.tex", "\n".join(t))
    return summary


# =========================================================================== Table 4 (R=16)
def tab_r16():
    rows = ["Linear", "Cubic spline", "Bicubic", "ArSSR", "IR3C"]
    cols = [(ds, f) for ds in C.DATASETS for f in ("U0", "all")]
    vals = {}
    for ds in C.DATASETS:
        specs = dict(C.method_specs(ds))
        for nm in rows:
            vals[(nm, ds, "U0")] = C.per_subject(ds, specs[nm], ["U0"], [16])
            vals[(nm, ds, "all")] = C.per_subject(ds, specs[nm], C.FAMILIES, [16])
    best = {c: bold_idx([vals[(nm,) + c].mean() for nm in rows]) for c in cols}
    t = [r"\begin{table}[t]",
         r"\caption{Unseen sparsity, $R=16$ (never used in training). PSNR (dB), mean across "
         r"test subjects. ``All'' pools U0--U6. DCSRN and I$^3$Net use factor-specific models "
         r"and have no $R=16$ model; SAINT was not evaluated at this factor.}",
         r"\label{tab:r16}",
         r"\small",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccccc@{}}",
         r"\toprule",
         r"& \multicolumn{2}{c}{BraTS} & \multicolumn{2}{c}{OAIZIB} & \multicolumn{2}{c}{IXI} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Method & U0 & All & U0 & All & U0 & All \\", r"\midrule"]
    for i, nm in enumerate(rows):
        cells = []
        for c in cols:
            s = f"{vals[(nm,) + c].mean():.2f}"
            cells.append(r"\textbf{" + s + "}" if best[c] == i else s)
        t.append(("IR3C (ours)" if nm == "IR3C" else nm) + " & " + " & ".join(cells) + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    C.write("tab_r16.tex", "\n".join(t))
    return {c: (vals[("IR3C",) + c].mean(), max(vals[(nm,) + c].mean() for nm in rows[:-1]))
            for c in cols}


# =========================================================================== Table 5 (rejection)
PS = [0.0, 0.1, 0.2, 0.3]
COMPETITORS = [m for m in C.SR_METHODS if m != "IR3C"]


def rejection_stats(ds):
    """For every (R, pattern, p): IR3C minus the competitor with the highest cohort PSNR."""
    res = {}
    if C.load_sr(ds) is None:
        return res
    for R in (2, 4):
        for pat in ("random", "burst"):
            for p in PS:
                ir = C.sr_per_subject(ds, "IR3C", R, pat, p)
                comp = {m: C.sr_per_subject(ds, m, R, pat, p) for m in COMPETITORS}
                comp = {m: s for m, s in comp.items() if s is not None}
                best = max(comp, key=lambda m: comp[m].mean())
                res[(R, pat, p)] = dict(best=best, ir=ir.mean(), bestv=comp[best].mean(),
                                        **C.paired(ir, comp[best]))
    # Holm over the 14 distinct conditions (p = 0 shared by both patterns)
    keys = [k for k in res if not (k[2] == 0 and k[1] == "burst")]
    adj = C.holm([res[k]["p"] for k in keys])
    for k, a in zip(keys, adj):
        res[k]["padj"] = a
    for R in (2, 4):
        res[(R, "burst", 0.0)]["padj"] = res[(R, "random", 0.0)]["padj"]
    return res


SHORT = {"linear": "Lin.", "bicubic": "Bic.", "ArSSR": "ArSSR", "DCSRNmix": "DCSRN",
         "I3Net_impute": r"I$^3$Net-A", "I3Net_subset": r"I$^3$Net-B",
         "SAINT_impute": "SAINT-A", "SAINT_subset": "SAINT-B"}


def tab_rejection():
    t = [r"\begin{table*}[t]",
         r"\caption{Slice rejection. A uniform acquisition at base factor $R$ loses a fraction "
         r"of its acquired interior slices, either independently (random) or in runs of two to "
         r"three slices (burst). Each cell gives IR3C minus the strongest competitor in that "
         r"condition (PSNR, dB) with its 95\% confidence interval; the strongest competitor is "
         r"named underneath (A: rejected slices imputed linearly before inference; B: "
         r"uniform stride-$2R$ subset). An asterisk marks $p<0.05$ after Holm correction over "
         r"the conditions of one dataset. At 0\% the competitor runs in its native uniform "
         r"regime. Absolute PSNR for every method is given in Appendix~\ref{app:reject}.}",
         r"\label{tab:rejection}",
         r"\small",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllcccc@{}}",
         r"\toprule",
         r"Dataset & Base & Pattern & 0\% & 10\% & 20\% & 30\% \\", r"\midrule"]
    allres = {}
    for k, ds in enumerate(C.DATASETS):
        res = rejection_stats(ds)
        allres[ds] = res
        if k:
            t.append(r"\midrule")
        for R in (2, 4):
            for pat in ("random", "burst"):
                lab = C.DS_LABEL[ds] if (R, pat) == (2, "random") else ""
                if not res:
                    t.append(r"\pendrow " + lab + f" & $R={R}$ & {pat} & " +
                             " & ".join([TBD] * 4) + r" \\")
                    continue
                cells, names = [], []
                for p in PS:
                    r = res[(R, pat, p)]
                    cells.append(f"{C.signed(r['mean'])} [{r['lo']:.2f}, {r['hi']:.2f}]"
                                 .replace("[-", "[$-$").replace(", -", ", $-$")
                                 + star(r["padj"]))
                    names.append(r"{\scriptsize\color{black!60}" + SHORT[r["best"]] + "}")
                t.append(lab + f" & $R={R}$ & {pat} & " + " & ".join(cells) + r" \\")
                t.append(r" & & & " + " & ".join(names) + r" \\[1pt]")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    C.write("tab_rejection.tex", "\n".join(t))
    return allres


def tab_rejection_absolute():
    """Appendix: absolute PSNR of every method in every rejection condition."""
    t = []
    for ds in ("BraTS", "OAIZIB", "IXI"):
        d = C.load_sr(ds)
        conds = [(R, pat, p) for R in (2, 4) for pat, ps in (("none", [0.0]),
                 ("random", PS[1:]), ("burst", PS[1:])) for p in ps]
        hdr = ["Linear", "Bicubic", "ArSSR", "DCSRN", r"I$^3$Net-A", r"I$^3$Net-B",
               "SAINT-A", "SAINT-B", "IR3C"]
        t += [r"\begin{table*}[t]",
              r"\caption{Slice rejection on " + C.DS_LABEL[ds] + r": absolute PSNR (dB), "
              r"mean across " + str(C.DS_N[ds]) + r" test subjects. A: rejected slices "
              r"imputed linearly, then the uniform-grid model is run at base $R$. B: the "
              r"model is run at $2R$ on the uniform subset that lost fewer slices. Bold marks "
              r"the best method per row.}",
              r"\label{tab:reject-abs-" + ds.lower() + "}",
              r"\footnotesize",
              r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lll" + "c" * len(hdr) + "@{}}",
              r"\toprule",
              r"Base & Pattern & Rejected & " + " & ".join(hdr) + r" \\", r"\midrule"]
        for (R, pat, p) in conds:
            pp = 0.0 if pat == "none" else p
            vals = [C.sr_per_subject(ds, m, R, "random" if pat == "none" else pat, pp)
                    for m in C.SR_METHODS]
            means = [v.mean() for v in vals]
            b = bold_idx(means)
            cells = [(r"\textbf{%.2f}" if i == b else "%.2f") % m for i, m in enumerate(means)]
            t.append(f"$R={R}$ & {'---' if pat == 'none' else pat} & {int(round(100 * p))}\\% & "
                     + " & ".join(cells) + r" \\")
        t += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    C.write("tab_rejection_absolute.tex", "\n".join(t))


# =========================================================================== Table 6 (downstream)
DS_METHODS = [("Linear", "linear"), ("Bicubic", "bicubic"), ("DCSRN", "DCSRN_mixed"),
              ("ArSSR", "ArSSR"), ("I$^3$Net", "I3Net"), ("SAINT", "SAINT"),
              ("IR3C (ours)", "IR3_matched")]
#: reference row below the rule, never bolded: the full-budget checkpoint (Table 23)
DS_FULL = ("IR3C, full budget", "IR3")
#: P2 run: seeded from runs/IR3C_LPIPS/downstream (byte-identical reference predictions)
#: plus the budget-matched IR3C rows (IR3C_paper_TODO.md, P2).
DS_CSV = "runs/IR3C_MedIA_P2_matched_downstream/downstream_segmentation_per_volume.csv"


def tab_downstream():
    d = pd.read_csv(os.path.join(C.ROOT, DS_CSV))
    rt = d[d.method == "gt_roundtrip"]
    floor = {k: rt[f"{k}_Dice"].mean() for k in ("WT", "TC", "ET")}
    cols = [(k, R) for k in ("WT", "TC", "ET") for R in C.TRAIN_R]
    vals = {}
    for nm, m in DS_METHODS + [DS_FULL]:
        for k, R in cols:
            s = d[(d.method == m) & (d.R == R)].set_index("subject_id")[f"{k}_Dice"]
            assert len(s) == 88, (m, k, R, len(s))
            vals[(nm, k, R)] = s
    best = {c: bold_idx([vals[(nm,) + c].mean() for nm, _ in DS_METHODS]) for c in cols}
    t = [r"\begin{table*}[t]",
         r"\caption{Downstream segmentation consistency on BraTS-GLI (U0, $n=88$). A frozen "
         r"SegResNet receives the acquired T1, T1c and T2 volumes with complete or "
         r"reconstructed FLAIR. Dice is computed against the same network's prediction on the "
         r"complete input, so it measures how far a reconstruction moves a fixed pipeline, not "
         r"agreement with expert labels. Re-saving and reloading the complete volume gives "
         r"WT/TC/ET Dice of " + f"{floor['WT']:.3f}/{floor['TC']:.3f}/{floor['ET']:.3f}" +
         r", which is the pipeline's noise floor. WT: whole tumour; TC: tumour core; ET: "
         r"enhancing tumour. IR3C is the budget-matched checkpoint used in all other tables; the "
         r"last row gives the full-budget checkpoint (Table~\ref{tab:budget}) for reference and "
         r"is excluded from the bold marking.}",
         r"\label{tab:downstream}",
         r"\small",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}",
         r"\toprule",
         r"& \multicolumn{3}{c}{WT Dice $\uparrow$} & \multicolumn{3}{c}{TC Dice $\uparrow$}"
         r" & \multicolumn{3}{c}{ET Dice $\uparrow$} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         "Method" + " & $R=2$ & $R=4$ & $R=8$" * 3 + r" \\", r"\midrule"]
    for i, (nm, _) in enumerate(DS_METHODS):
        cells = []
        for c in cols:
            s = f"{vals[(nm,) + c].mean():.3f}"
            cells.append(r"\textbf{" + s + "}" if best[c] == i else s)
        t.append(nm + " & " + " & ".join(cells) + r" \\")
    nm = DS_FULL[0]
    t.append(r"\midrule")
    t.append(nm + " & " + " & ".join(f"{vals[(nm,) + c].mean():.3f}" for c in cols) + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    C.write("tab_downstream.tex", "\n".join(t))
    # paired IR3 vs linear and vs SAINT, WT
    out = {}
    for R in C.TRAIN_R:
        out[R] = dict(lin=C.paired(vals[("IR3C (ours)", "WT", R)],
                                   vals[("Linear", "WT", R)]),
                      saint=C.paired(vals[("IR3C (ours)", "WT", R)],
                                     vals[("SAINT", "WT", R)]))
        out[R]["full"] = C.paired(vals[("IR3C (ours)", "WT", R)], vals[(DS_FULL[0], "WT", R)])
    return floor, {k: {nm: vals[(nm,) + k].mean() for nm, _ in DS_METHODS + [DS_FULL]}
                   for k in cols}, out


# =========================================================================== anchor ablation
def tab_anchor():
    p = os.path.join(C.ROOT, "runs/IR3C_brats_ablation1/metrics/"
                             "IR3C_brats_ablation1_results_per_volume.csv")
    d = pd.read_csv(p, low_memory=False)
    d = d[(d.region == "missing") & d.R.isin(C.TRAIN_R)]
    names = [("nearest", "Nearest"), ("linear", "Linear"), ("cubic", "Cubic spline"),
             ("bicubic", "Bicubic")]
    per = {}
    for k, _ in names:
        for st in ("interpolation_only", "IR3C_final"):
            s = d[d.method == f"{st}_{k}"].groupby("subject").psnr_masked.mean()
            per[(k, st)] = s
    t = [r"\begin{table}[t]",
         r"\caption{Choice of anchor (BraTS-GLI, all families, $R\in\{2,4,8\}$, $n=88$). Each "
         r"model is trained from the same initial weights with identical data order; only the "
         r"interpolant used as the anchor changes. $\Delta$ is the paired difference to the "
         r"linear-anchored model with its 95\% confidence interval.}",
         r"\label{tab:anchor}",
         r"\small",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}",
         r"\toprule",
         r"Anchor & Alone & Corrected & $\Delta$ vs.\ linear anchor \\", r"\midrule"]
    info = {}
    for k, nm in names:
        a = per[(k, "interpolation_only")].mean()
        f = per[(k, "IR3C_final")].mean()
        if k == "linear":
            dcell = "---"
        else:
            pr = C.paired(per[(k, "IR3C_final")], per[("linear", "IR3C_final")])
            dcell = (f"{C.signed(pr['mean'])} [{pr['lo']:.2f}, {pr['hi']:.2f}]"
                     .replace("[-", "[$-$").replace(", -", ", $-$"))
        t.append(f"{nm} & {a:.2f} & {f:.2f} & {dcell}" + r" \\")
        info[k] = (a, f)
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    C.write("tab_anchor.tex", "\n".join(t))
    return info


# =========================================================================== formulation (P3)
P3_SEEDS = {"A0_ir3c": ["A0_ir3c", "A0_ir3c_s43", "A0_ir3c_s44"],
            "A3_direct": ["A3_direct", "A3_direct_s43", "A3_direct_s44"],
            "C2_no_positional": ["C2_no_positional", "C2_no_positional_s43",
                                 "C2_no_positional_s44"]}
# Table 9 columns: (key, families, factors, rejection)
P3_COLS = [("U0", ["U0"], C.TRAIN_R, False), ("irr", C.NONUNIFORM, C.TRAIN_R, False),
           ("U4", ["U4"], C.TRAIN_R, False), ("R16", C.FAMILIES, [16], False),
           ("rej", None, None, True)]


def _p3_col(tag, col, metric="psnr_masked"):
    _, fams, Rs, rej = col
    if rej:
        return C.p3_per_subject(tag, metric=metric, rejection=True)
    return C.p3_per_subject(tag, fams, Rs, metric)


def _p3_seedavg(arm, col, metric="psnr_masked"):
    """Per-subject value averaged over seeds, and the per-seed means."""
    tags = P3_SEEDS.get(arm, [arm])
    per = [_p3_col(t, col, metric) for t in tags]
    return pd.concat(per, axis=1).mean(axis=1), [float(x.mean()) for x in per]


def tab_formulation():
    """Formulation ablation (P3): every arm trained by p3_train_arm.py at the matched budget
    (epochs 1-10 of the 25-epoch schedule) and scored by p3_evaluate.py."""
    rows = [("Direct 3D U-Net (no anchor)", "A3_direct", r"\texttt{A3\_direct}$^{a}$"),
            ("Anchor, unbounded residual", "A1_unbounded", r"\texttt{A1\_unbounded}"),
            ("Anchor, non-zero output head", "A2_nonzero_init", r"\texttt{A2\_nonzero\_init}"),
            ("Anchor, no geometry profiles", "C2_no_positional", r"\texttt{C2\_no\_positional}"),
            ("Anchor, binary mask profile only", "C4_binary_mask", r"\texttt{C4\_binary\_mask}"),
            ("Anchor, uniform-only training", "C1_uniform_only", r"\texttt{C1\_uniform\_only}$^{b}$"),
            ("Anchor, single factor ($R=4$)", "C3_single_R", r"\texttt{C3\_single\_R}"),
            ("IR3C (full formulation)", "A0_ir3c", r"\texttt{A0\_ir3c}")]
    lin = {c[0]: _p3_col("A0_ir3c", c, "x0_psnr_masked").mean() for c in P3_COLS}
    ref = {c[0]: _p3_seedavg("A0_ir3c", c)[0] for c in P3_COLS}
    C1_best = C.p3_per_subject("C1_uniform_only@best").mean()
    lin_all = C.p3_per_subject("A0_ir3c", metric="x0_psnr_masked").mean()
    t = [r"\begin{table*}[t]",
         r"\caption{Formulation ablation on BraTS-GLI ($n=88$). Every arm uses the IR3C backbone, "
         r"optimiser, data order and matched training budget (12,510 steps) and changes one "
         r"component. PSNR (dB) over missing-slice foreground, $R\in\{2,4,8\}$, except the "
         r"$R=16$ column (all families, out of distribution) and the rejection column ($R=4$ "
         r"uniform base, 30\% of interior acquired slices rejected in bursts). Arms with three "
         r"seeds are given as the mean over seeds, with the range of the seed means in "
         r"brackets. $\Delta$ is the paired per-subject difference to IR3C on U1--U6, with its "
         r"95\% confidence interval. The IR3C row is retrained in the ablation harness and is "
         r"therefore not the same weights as the IR3C of the other tables. "
         r"$^{a}$No trained checkpoint of the direct model meets the validation SSIM "
         r"guardrail, so the checkpoint with the best validation PSNR is reported; its "
         r"training loss is computed on the unclamped output. "
         r"$^{b}$The selection rule returns the initial weights, which equal linear "
         r"interpolation exactly; the best trained checkpoint (epoch 1) gives "
         + f"{C1_best:.2f}" + r"\,dB pooled over all families and factors (linear: "
         + f"{lin_all:.2f}" + r"\,dB).}",
         r"\label{tab:formulation}",
         r"\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccccc@{}}",
         r"\toprule",
         r"Variant & Arm & Seeds & U0 & U1--U6 & U4 & $R=16$ & Rejection & "
         r"$\Delta$ vs.\ IR3C (U1--U6) \\",
         r"\midrule",
         "Linear interpolation only & --- & --- & " +
         " & ".join(f"{lin[c[0]]:.2f}" for c in P3_COLS) + r" & --- \\"]
    info = {}
    for nm, arm, lab in rows:
        nseed = len(P3_SEEDS.get(arm, [arm]))
        cells = []
        for c in P3_COLS:
            v, seeds = _p3_seedavg(arm, c)
            cell = f"{v.mean():.2f}"
            if nseed > 1:
                cell += r"\,{\scriptsize[" + f"{min(seeds):.2f}, {max(seeds):.2f}" + "]}"
            cells.append(cell)
            info[(arm, c[0])] = v.mean()
        if arm == "A0_ir3c":
            dcell = "---"
        else:
            v, _ = _p3_seedavg(arm, P3_COLS[1])
            pr = C.paired(v, ref["irr"])
            info[(arm, "d_irr")] = pr
            dcell = (f"{C.signed(pr['mean'])} [{pr['lo']:.2f}, {pr['hi']:.2f}]"
                     .replace("[-", "[$-$").replace(", -", ", $-$"))
        name = r"\textbf{" + nm + "}" if arm == "A0_ir3c" else nm
        t.append(f"{name} & {lab} & {nseed} & " + " & ".join(cells) + f" & {dcell}" + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    C.write("tab_formulation.tex", "\n".join(t))
    info["linear"] = lin
    return info


# =========================================================================== loss terms (P3)
def tab_loss():
    """Loss-term leave-one-out (Ablation B), scored on image metrics AND downstream Dice,
    because the hypothesis is that the background term matters for the second only.
    B0 = A0_ir3c seed 42. Dice comes from p3_loss_downstream.py; cells stay \tbd until then."""
    d = pd.read_csv(os.path.join(C.ROOT, DS_CSV))
    lin_wt = [d[(d.method == "linear") & (d.R == R)]["WT_Dice"].mean() for R in C.TRAIN_R]
    dd = (pd.read_csv(C.P3_DOWNSTREAM, low_memory=False)
          if os.path.exists(os.path.join(os.path.dirname(C.P3_DOWNSTREAM), "STATUS.txt"))
          and "DONE" in open(os.path.join(os.path.dirname(C.P3_DOWNSTREAM), "STATUS.txt")).read()
          else None)
    rows = [
        ("Full objective (Eq.~\\ref{eq:loss})", "A0_ir3c", r"\texttt{B0\_ir3c}", "---"),
        ("No background term", "B1_no_background", r"\texttt{B1\_no\_background}",
         r"$0.30\,\mathcal{L}_{\mathrm{C}}^{\mathrm{bg}}$"),
        ("No structural term", "B2_no_structure", r"\texttt{B2\_no\_structure}",
         r"$0.10\,(1-\mathrm{SSIM})$"),
        ("No gradient term", "B3_no_gradient", r"\texttt{B3\_no\_gradient}",
         r"$0.05\,\mathcal{L}_{\nabla_z}^{\mathrm{fg}}$"),
    ]
    lin = dict(u0=C.p3_per_subject("A0_ir3c", ["U0"], metric="x0_psnr_masked").mean(),
               irr=C.p3_per_subject("A0_ir3c", C.NONUNIFORM, metric="x0_psnr_masked").mean(),
               ssim=C.p3_per_subject("A0_ir3c", metric="x0_ssim").mean(),
               bg=C.p3_per_subject("A0_ir3c", ["U0"], metric="x0_bg_haze").mean())
    t = [r"\begin{table*}[t]",
         r"\caption{" + ("" if dd is not None else r"\pendlabel\ ") +
         r"Loss-term ablation on BraTS-GLI ($n=88$): each arm removes one term of "
         r"Eq.~(\ref{eq:loss}) and keeps everything else, including the $0.001\,\lVert\delta"
         r"\rVert_1$ term, the backbone, the initial weights, the data order and the matched "
         r"budget; one seed per arm. PSNR (dB) and SSIM are over missing-slice foreground, "
         r"$R\in\{2,4,8\}$; SSIM is pooled over all families. Background haze is the mean "
         r"predicted intensity on missing-slice background voxels, U0 ($\times10^{-3}$). WT "
         r"Dice is the downstream consistency of Table~\ref{tab:downstream} (U0, $n=88$). The "
         r"control \texttt{B0\_ir3c} is the same trained model as \texttt{A0\_ir3c} (seed 42) "
         r"in Table~\ref{tab:formulation}." +
         ("" if dd is not None else r" WT Dice of the ablation arms is pending.") + "}",
         r"\label{tab:loss}",
         r"\small",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllcccccccc@{}}",
         r"\toprule",
         r"& & & \multicolumn{2}{c}{PSNR $\uparrow$} & SSIM $\uparrow$ & Bg.\ haze $\downarrow$ & "
         r"\multicolumn{3}{c}{WT Dice $\uparrow$} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){8-10}",
         r"Variant & Arm & Removed term & U0 & U1--U6 & all & U0 & $R=2$ & $R=4$ & $R=8$ \\",
         r"\midrule",
         f"Linear interpolation only & --- & --- & {lin['u0']:.2f} & {lin['irr']:.2f} & "
         f"{lin['ssim']:.3f} & {1e3 * lin['bg']:.2f} & " +
         " & ".join(f"{v:.3f}" for v in lin_wt) + r" \\"]
    info = {}
    for nm, arm, lab, term in rows:
        u0 = C.p3_per_subject(arm, ["U0"]).mean()
        irr = C.p3_per_subject(arm, C.NONUNIFORM).mean()
        ss = C.p3_per_subject(arm, metric="ssim").mean()
        bg = C.p3_per_subject(arm, ["U0"], metric="bg_haze").mean()
        if dd is not None:
            key = f"P3_{arm}"
            wt = [dd[(dd.method == key) & (dd.R == R)]["WT_Dice"].mean() for R in C.TRAIN_R]
            wcells = [f"{v:.3f}" for v in wt]
            info[(arm, "wt")] = wt
        else:
            wcells = [TBD] * 3
        info[arm] = dict(u0=u0, irr=irr, ssim=ss, bg=bg)
        t.append(f"{nm} & {lab} & {term} & {u0:.2f} & {irr:.2f} & {ss:.3f} & {1e3 * bg:.2f} & "
                 + " & ".join(wcells) + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    C.write("tab_loss.tex", "\n".join(t))
    info["linear"] = lin
    return info


# =========================================================================== pending: held-out family
def tab_heldout():
    spec = dict(C.method_specs("BraTS"))
    t = [r"\begin{table}[t]",
         r"\caption{\pendlabel\ Generalisation to an unseen sampling family (BraTS-GLI, "
         r"$R\in\{2,4,8\}$, PSNR in dB). ``Seen'' is the reported IR3C, whose training "
         r"distribution contains all six irregular families. ``Held out'' is retrained with "
         r"that family removed. ``Uniform-only'' is trained on U0 alone. Linear and seen IR3C "
         r"are computed; the other columns are pending.}",
         r"\label{tab:heldout}",
         r"\small",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}",
         r"\toprule",
         r"Test family & Linear & Seen & Held out & Uniform-only \\", r"\midrule"]
    for f in C.NONUNIFORM:
        lv = C.per_subject("BraTS", spec["Linear"], [f]).mean()
        iv = C.per_subject("BraTS", spec["IR3C"], [f]).mean()
        t.append(f"{f} & {lv:.2f} & {iv:.2f} & " + r"\cellcolor{pendbg}" + TBD +
                 r" & \cellcolor{pendbg}" + TBD + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    C.write("tab_heldout.tex", "\n".join(t))


# =========================================================================== mask realisations
P5_CELLS = os.path.join(C.ROOT, "runs/IR3C_MedIA_P5_realisations/cells_all.csv")
P5_ROWS = [("linear", "Linear"), ("bicubic", "Bicubic"), ("ArSSR", "ArSSR"),
           ("DCSRNmix", "DCSRN"), ("IR3C", "IR3C (ours)")]


def tab_realisations():
    fams = C.NONUNIFORM
    c = pd.read_csv(P5_CELLS)
    c = c[c.k >= 0]
    g = c.groupby(["method", "family", "subject"]).psnr
    mu, sd = g.mean(), g.std(ddof=1)            # within-subject mean / SD over realisations
    t = [r"\begin{table*}[t]",
         r"\caption{Sensitivity to the mask realisation (BraTS-GLI, $R=4$, $n=88$). Each "
         r"subject receives five independent masks per family, drawn with seeds disjoint from "
         r"the training and test seeds, instead of one mask shared by all subjects. Cells are "
         r"the mean over subjects of the within-subject mean PSNR (dB); parentheses give the "
         r"mean within-subject standard deviation across the five realisations. $\Delta$ is "
         r"the paired difference between IR3C and the strongest comparator in that column, "
         r"with its 95\% confidence interval across subjects; the shared-mask row repeats it "
         r"for the single test mask of Table~\ref{tab:families} at $R=4$. ``IR3C wins'' is the "
         r"fraction of the 440 (subject, realisation) pairs in which IR3C has the highest "
         r"PSNR. Bold marks the best method per column.}",
         r"\label{tab:realisations}",
         r"\small",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}",
         r"\toprule",
         "Method & " + " & ".join(fams) + r" \\", r"\midrule"]
    best = {f: bold_idx([mu[(m, f)].mean() for m, _ in P5_ROWS]) for f in fams}
    for i, (m, nm) in enumerate(P5_ROWS):
        cells = []
        for f in fams:
            v = f"{mu[(m, f)].mean():.2f}"
            v = r"\textbf{" + v + "}" if best[f] == i else v
            cells.append(f"{v} ({sd[(m, f)].mean():.2f})")
        t.append((r"\textbf{" + nm + "}" if m == "IR3C" else nm) + " & " + " & ".join(cells) + r" \\")
    t.append(r"\midrule")
    w = c.pivot_table(index=["subject", "family", "k"], columns="method", values="psnr")
    others = [m for m, _ in P5_ROWS if m != "IR3C"]
    specs = dict(C.method_specs("BraTS"))
    shared_names = {"linear": "Linear", "bicubic": "Bicubic", "ArSSR": "ArSSR",
                    "DCSRNmix": "DCSRN", "IR3C": "IR3C"}
    d_real, ci, d_shared, wins, summary = [], [], [], [], {}
    for f in fams:
        ref = max(others, key=lambda m: mu[(m, f)].mean())
        pr = C.paired(mu[("IR3C", f)], mu[(ref, f)])
        sh = {m: C.per_subject("BraTS", specs[shared_names[m]], [f], [4]) for m, _ in P5_ROWS}
        sref = max(others, key=lambda m: sh[m].mean())
        ps = C.paired(sh["IR3C"], sh[sref])
        wf = w.xs(f, level="family")
        win = float((wf["IR3C"] > wf[others].max(axis=1)).mean())
        d_real.append(C.signed(pr["mean"]))
        ci.append(f"[{pr['lo']:.2f}, {pr['hi']:.2f}]")
        d_shared.append(f"{C.signed(ps['mean'])}")
        wins.append(f"{100 * win:.1f}\\%")
        summary[f] = dict(ref=ref, delta=pr, shared_ref=sref, shared=ps, win=win)
    t.append(r"$\Delta$ vs.\ best comparator & " + " & ".join(d_real) + r" \\")
    t.append(r"\quad 95\% CI & " + " & ".join(ci) + r" \\")
    t.append(r"$\Delta$, shared test mask & " + " & ".join(d_shared) + r" \\")
    t.append(r"IR3C wins & " + " & ".join(wins) + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    C.write("tab_realisations.tex", "\n".join(t))
    return summary


# =========================================================================== backbone
def tab_backbone():
    base = os.path.join(C.ROOT, "IR3C_Brats_Architecture_XAI_MI_V2/results")
    bench = json.load(open(os.path.join(base, "bench/bench.json")))["models"]
    arms = [("original", "MONAI U-Net (IR3C)", "abl_linear"),
            ("simplified", "Reduced bottleneck", "simplified_seed42"),
            ("depthwise", "Depthwise separable", "depthwise_seed42"),
            ("nnunet", "nnU-Net config.", "nnunet_seed42")]
    cells = {}
    for key, _, folder in arms:
        f = os.path.join(base, "eval", folder,
                         "test_cells.csv")
        e = pd.read_csv(f)
        e = e[e.R.isin(C.TRAIN_R)]
        cells[key] = (e[e.family == "U0"].groupby("subject").missing_psnr_masked.mean(),
                      e[e.family != "U0"].groupby("subject").missing_psnr_masked.mean())
    t = [r"\begin{table}[t]",
         r"\caption{Backbone capacity (BraTS-GLI, $R\in\{2,4,8\}$, $n=88$, single seed). "
         r"The anchor, input profiles, loss and training budget are identical; only the "
         r"network that predicts the correction changes. $\Delta$ is the paired difference to "
         r"the IR3C backbone on U1--U6. Parameter counts and multiply--accumulate operations "
         r"(MACs) are for one full-volume forward pass.}",
         r"\label{tab:backbone}",
         r"\small",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccccc@{}}",
         r"\toprule",
         r"Backbone & Param.\ (M) & GMACs & U0 & U1--U6 & $\Delta$ \\", r"\midrule"]
    out = {}
    for key, nm, _ in arms:
        b = bench.get(key, {})
        par = b.get("params", {}).get("total", np.nan) / 1e6
        mac = b.get("macs", {}).get("macs_total", np.nan) / 1e9
        u0, irr = cells[key]
        if key == "original":
            dcell = "---"
        else:
            pr = C.paired(irr, cells["original"][1])
            dcell = C.signed(pr["mean"])
            out[key] = pr
        t.append(f"{nm} & {par:.2f} & {mac:.0f} & {u0.mean():.2f} & {irr.mean():.2f} & "
                 f"{dcell}" + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    C.write("tab_backbone.tex", "\n".join(t))
    return out


# =========================================================================== cost
def tab_cost():
    prof = os.path.join(C.ROOT, "IR3C_experiments_V2/stage1_profiling")
    c = pd.read_csv(os.path.join(prof, "complexity_per_unit.csv"))
    r = pd.read_csv(os.path.join(prof, "fullvolume_runtime.csv"))
    rows = []
    for m, nm in (("IR3C", "IR3C (ours)"), ("DCSRN", "DCSRN"), ("ArSSR", "ArSSR"),
                  ("I3Net", r"I$^3$Net"), ("SAINT", "SAINT")):
        par = c[c.model == m].total_params.max() / 1e6
        per_R = r[r.model == m].groupby("R").recon_mean_s.mean()
        rows.append((nm, par, per_R.get(2, np.nan), per_R.get(4, np.nan), per_R.get(8, np.nan)))
    ir = rows[0]
    t = [r"\begin{table}[t]",
         r"\caption{Reconstruction time per BraTS-GLI volume ($240\times240\times153$), "
         r"measured end to end on one NVIDIA GB10 GPU at fp32 with batch size 1. Times "
         r"include patch extraction, sliding-window aggregation and output assembly, and are "
         r"averaged over test volumes and three repetitions.}",
         r"\label{tab:cost}",
         r"\small",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}",
         r"\toprule",
         r"& & \multicolumn{3}{c}{Seconds per volume} \\", r"\cmidrule(lr){3-5}",
         r"Method & Params (M) & $R=2$ & $R=4$ & $R=8$ \\", r"\midrule"]
    for nm, par, a, b, cc in rows:
        t.append(f"{nm} & {par:.2f} & {a:.2f} & {b:.2f} & {cc:.2f}" + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    C.write("tab_cost.tex", "\n".join(t))
    return rows


# =========================================================================== appendix detail
def tab_detail():
    """Per-dataset U0 table with all metrics (mean +/- sd), and family table with sd."""
    t = []
    rows = ["Nearest", "Linear", "Cubic spline", "Bicubic", "DCSRN", "ArSSR", "I$^3$Net",
            "SAINT", "IR3C"]
    mets = [("psnr_masked", r"PSNR (dB) $\uparrow$", True, 2),
            ("ssim", r"SSIM $\uparrow$", True, 4),
            ("nmse", r"NMSE $\downarrow$", False, 4),
            ("mae", r"MAE $\downarrow$", False, 4)]
    for ds in C.DATASETS:
        specs = dict(C.method_specs(ds))
        vals = {(nm, m): C.per_subject(ds, specs[nm], ["U0"], C.TRAIN_R, m)
                for nm in rows for m, *_ in mets}
        t += [r"\begin{table*}[t]",
              r"\caption{" + C.DS_LABEL[ds] + r", uniform sampling (U0), pooled over "
              r"$R\in\{2,4,8\}$. Mean $\pm$ standard deviation across " + str(C.DS_N[ds]) +
              r" test subjects, after averaging within subject. Bold marks the best method.}",
              r"\label{tab:detail-" + ds.lower() + "}",
              r"\small",
              r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
              r"\toprule", "Method & " + " & ".join(h for _, h, _, _ in mets) + r" \\",
              r"\midrule"]
        best = {m: bold_idx([vals[(nm, m)].mean() for nm in rows], hi) for m, _, hi, _ in mets}
        for i, nm in enumerate(rows):
            cells = []
            for m, _, _, dec in mets:
                mu, sd = C.mean_sd(vals[(nm, m)])
                s = f"{mu:.{dec}f} $\\pm$ {sd:.{dec}f}"
                cells.append(r"\textbf{" + s + "}" if best[m] == i else s)
            t.append(nm + " & " + " & ".join(cells) + r" \\")
        t += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]

        frows = ["Nearest", "Linear", "Cubic spline", "Bicubic", "DCSRN", "ArSSR", "IR3C"]
        fv = {(nm, f): C.per_subject(ds, specs[nm], [f]) for nm in frows for f in C.FAMILIES}
        t += [r"\begin{table*}[t]",
              r"\caption{" + C.DS_LABEL[ds] + r", PSNR (dB) by sampling family, pooled over "
              r"$R\in\{2,4,8\}$. Mean $\pm$ standard deviation across test subjects.}",
              r"\label{tab:detailfam-" + ds.lower() + "}",
              r"\footnotesize",
              r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}",
              r"\toprule", "Method & " + " & ".join(C.FAMILIES) + r" \\", r"\midrule"]
        best = {f: bold_idx([fv[(nm, f)].mean() for nm in frows]) for f in C.FAMILIES}
        for i, nm in enumerate(frows):
            cells = []
            for f in C.FAMILIES:
                mu, sd = C.mean_sd(fv[(nm, f)])
                s = f"{mu:.2f} $\\pm$ {sd:.2f}"
                cells.append(r"\textbf{" + s + "}" if best[f] == i else s)
            t.append(nm + " & " + " & ".join(cells) + r" \\")
        t += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    C.write("tab_detail.tex", "\n".join(t))


def tab_budget():
    """Matched versus full training budget, all datasets, all families, R in {2,4,8}."""
    full_specs = {"BraTS": {"m": "IR3_full"}, "OAIZIB": {"m": "IR3C_OAIZIB"},
                  "IXI": {"m": "IR3C_IXI"}}
    t = [r"\begin{table}[t]",
         r"\caption{Effect of the source training budget on IR3C (PSNR in dB, all families, "
         r"$R\in\{2,4,8\}$). ``Matched'' stops source training at 12,510 steps, the largest "
         r"budget given to any baseline, and is the model used in every comparison. ``Full'' "
         r"uses the complete 25-epoch schedule (selected at step 28,773). For OAIZIB and "
         r"IXI-tiny, the two rows differ only in the source checkpoint used to initialise "
         r"fine-tuning.}",
         r"\label{tab:budget}",
         r"\small",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}",
         r"\toprule", r"Dataset & Matched & Full & Full $-$ matched \\", r"\midrule"]
    for ds in C.DATASETS:
        m = C.per_subject(ds, dict(C.method_specs(ds))["IR3C"])
        f = C.per_subject(ds, full_specs[ds])
        pr = C.paired(f, m)
        t.append(f"{C.DS_LABEL[ds]} & {m.mean():.2f} & {f.mean():.2f} & "
                 f"{C.signed(pr['mean'])} [{pr['lo']:.2f}, {pr['hi']:.2f}]"
                 .replace("[-", "[$-$").replace(", -", ", $-$") + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    C.write("tab_budget.tex", "\n".join(t))


def tab_lpips():
    """LPIPS on U0, reused from the audited V2 LPIPS sweep (IR3C rows as archived)."""
    sys.path.insert(0, os.path.join(C.ROOT, "IR3C_journal/IR3C_journal_V2"))
    import make_tables as V2  # noqa: E402
    rows = [("Linear", "linear"), ("Bicubic", "bicubic"), ("ArSSR", "ArSSR_slowlr"),
            ("I$^3$Net", "I3Net"), ("SAINT", None), ("IR3C", None)]
    t = [r"\begin{table}[t]",
         r"\caption{LPIPS (AlexNet features, lower is better) on uniform sampling, pooled over "
         r"$R\in\{2,4,8\}$, mean across test subjects. Learned methods are not better than "
         r"classical interpolation on this metric (Section~\ref{sec:disc-metrics}). "
         r"$^{\ddagger}$Full-budget IR3C; LPIPS was not computed for the matched-budget "
         r"transfer models.}",
         r"\label{tab:lpips}", r"\small",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}",
         r"\toprule", r"Method & BraTS-GLI & OAIZIB & IXI-tiny \\", r"\midrule"]
    srcs = {"BraTS": dict(V2.brats_sources()), "OAIZIB": dict(V2.v2_sources("OAIZIB")),
            "IXI": dict(V2.v2_sources("IXI"))}
    lp = {ds: V2.load_lpips(ds) for ds in C.DATASETS}
    vals = {}
    for nm, _ in rows:
        key = {"I$^3$Net": "I3Net"}.get(nm, nm)
        for ds in C.DATASETS:
            spec = srcs[ds][key]
            s = V2.lpips_for(lp[ds], spec, ["U0"], C.TRAIN_R)
            vals[(nm, ds)] = None if s is None else s.mean()
    best = {ds: bold_idx([vals[(nm, ds)] for nm, _ in rows], False) for ds in C.DATASETS}
    for i, (nm, _) in enumerate(rows):
        cells = []
        for ds in C.DATASETS:
            v = vals[(nm, ds)]
            s = "---" if v is None else f"{v:.4f}"
            if nm == "IR3C" and ds != "BraTS" and v is not None:
                s += r"$^{\ddagger}$"
            cells.append(r"\textbf{" + s + "}" if best[ds] == i and v is not None else s)
        t.append(nm + " & " + " & ".join(cells) + r" \\")
    t += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    C.write("tab_lpips.tex", "\n".join(t))
    return vals


# =========================================================================== main
if __name__ == "__main__":
    facts = {}
    tab_datasets()
    facts["uniform_vs_saint"] = tab_uniform()
    facts["families"] = tab_families()
    facts["r16"] = tab_r16()
    facts["rejection"] = tab_rejection()
    tab_rejection_absolute()
    facts["downstream"] = tab_downstream()
    facts["anchor"] = tab_anchor()
    tab_formulation()
    tab_loss()
    tab_heldout()
    tab_realisations()
    facts["backbone"] = tab_backbone()
    facts["cost"] = tab_cost()
    tab_detail()
    tab_budget()
    facts["lpips"] = tab_lpips()

    # a plain-text digest of every number quoted in the prose, for cross-checking
    lines = []
    for (ds, R), p in zip([(d, R) for d in C.DATASETS for R in C.TRAIN_R],
                          facts["uniform_vs_saint"]):
        lines.append(f"U0 {ds} R{R}: IR3C-SAINT {p['mean']:+.3f} [{p['lo']:+.3f},{p['hi']:+.3f}]"
                     f" win {p['win']:.2f} p {p['p']:.2g}")
    for (ds, c), (ref, p) in facts["families"].items():
        lines.append(f"FAM {ds} {c}: IR3C-{ref} {p['mean']:+.3f} [{p['lo']:+.3f},{p['hi']:+.3f}]"
                     f" win {p['win']:.2f}")
    for k, v in facts["r16"].items():
        lines.append(f"R16 {k}: IR3C {v[0]:.2f} best other {v[1]:.2f}")
    for ds, res in facts["rejection"].items():
        for k, r in res.items():
            lines.append(f"SR {ds} {k}: IR3C {r['ir']:.2f} best {r['best']} {r['bestv']:.2f}"
                         f" d {r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] win {r['win']:.2f}"
                         f" padj {r['padj']:.2g}")
    floor, dvals, dpaired = facts["downstream"]
    lines.append(f"DS floor {floor}")
    for k, v in dvals.items():
        lines.append(f"DS {k}: " + ", ".join(f"{n}={x:.4f}" for n, x in v.items()))
    for R, pr in dpaired.items():
        lines.append(f"DS WT R{R}: IR3-lin {pr['lin']['mean']:+.4f} p {pr['lin']['p']:.2g}; "
                     f"IR3-SAINT {pr['saint']['mean']:+.4f} p {pr['saint']['p']:.2g}; "
                     f"matched-full {pr['full']['mean']:+.4f} p {pr['full']['p']:.2g}")
    for k, (a, f) in facts["anchor"].items():
        lines.append(f"ANCHOR {k}: raw {a:.3f} final {f:.3f}")
    for k, p in facts["backbone"].items():
        lines.append(f"BACKBONE {k}: {p['mean']:+.3f} [{p['lo']:+.3f},{p['hi']:+.3f}] win {p['win']:.2f}")
    for r in facts["cost"]:
        lines.append(f"COST {r}")
    for k, v in facts["lpips"].items():
        lines.append(f"LPIPS {k}: {v}")
    with open(os.path.join(C.PAPER, "scripts", "numbers_digest.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  scripts/numbers_digest.txt")
