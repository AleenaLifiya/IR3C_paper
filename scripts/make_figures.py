#!/usr/bin/env python
"""Write every data figure of the IR3C MedIA manuscript.

Run:  /home/aleena/miniforge3/envs/xdiff/bin/python scripts/make_figures.py

Style follows IR3C_Brats_XAI_MI/scripts/figstyle.py (Okabe-Ito in fixed order, markers as a
second encoding, recessive axes), with two changes for the Elsevier CAS layout: STIX serif
to match the journal body font, and widths set to the cas-dc column (3.30 in) and text
block (6.84 in) so figures are drawn at the size they are printed.

Rules applied to every panel
  * every plotted mean has a 95% confidence interval across subjects;
  * ticks only at values that were actually run;
  * one shared intensity window and one shared error scale per image figure;
  * no titles inside figures; the caption carries the description;
  * qualitative examples use a rule fixed before looking at images (median subject).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# ------------------------------------------------------------------ style
BLUE, VERM, GREEN, PURPLE, ORANGE, SKY = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                          "#E69F00", "#56B4E9")
GREY, INK, INK2, MUTED, GRIDC = "#8a8f98", "#1a1a1a", "#4b5563", "#6b7280", "#dde1e6"
SINGLE, DOUBLE = 3.30, 6.84

# one identity per method, used in every figure
STYLE = {
    "IR3C":    dict(color=BLUE,   marker="o", ls="-",  lw=1.8, zorder=6),
    "SAINT":   dict(color=VERM,   marker="s", ls="-",  lw=1.3, zorder=5),
    "SAINT-B": dict(color=VERM,   marker="s", ls="--", lw=1.1, zorder=5, mfc="white"),
    "I3Net":   dict(color=GREEN,  marker="^", ls="-",  lw=1.3, zorder=4),
    "I3Net-B": dict(color=GREEN,  marker="^", ls="--", lw=1.1, zorder=4, mfc="white"),
    "ArSSR":   dict(color=PURPLE, marker="D", ls="-",  lw=1.3, zorder=4),
    "DCSRN":   dict(color=ORANGE, marker="v", ls="-",  lw=1.3, zorder=3),
    "Linear":  dict(color=GREY,   marker="",  ls="-",  lw=1.2, zorder=2),
    "Bicubic": dict(color=GREY,   marker="",  ls=":",  lw=1.2, zorder=2),
    "Cubic spline": dict(color=GREY, marker="x", ls="--", lw=1.0, zorder=2),
}
LABEL = {"I3Net": "I$^3$Net", "I3Net-B": "I$^3$Net, uniform subset",
         "SAINT-B": "SAINT, uniform subset", "IR3C": "IR3C (ours)"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRIDC, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.markersize": 3.6, "legend.frameon": False, "legend.handlelength": 2.2,
    "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015, "text.color": INK, "axes.labelcolor": INK,
    "pdf.fonttype": 42,
})


def save(fig, name):
    os.makedirs(C.FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(C.FIGS, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  figs/{name}.pdf")


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            ha="right", va="bottom")


def ci(s):
    s = np.asarray(s, float)
    n = len(s)
    return float(stats.t.ppf(0.975, n - 1) * s.std(ddof=1) / np.sqrt(n))


def plot_series(ax, x, means, cis, key, label=None, dx=0.0):
    st = dict(STYLE[key])
    mfc = st.pop("mfc", st["color"])
    z = st.pop("zorder")
    ax.errorbar(np.asarray(x, float) + dx, means, yerr=cis, color=st["color"],
                marker=st["marker"] or None, ls=st["ls"], lw=st["lw"], mfc=mfc,
                mec=st["color"], mew=0.8, capsize=1.6, elinewidth=0.7, zorder=z,
                label=label or LABEL.get(key, key))


def legend_handles(keys):
    hs = []
    for k in keys:
        st = STYLE[k]
        hs.append(Line2D([], [], color=st["color"], marker=st["marker"] or None, ls=st["ls"],
                         lw=st["lw"], mfc=st.get("mfc", st["color"]), mec=st["color"],
                         mew=0.8, label=LABEL.get(k, k)))
    return hs


SPEC_KEY = {"DCSRN": "DCSRN", "ArSSR": "ArSSR", "I3Net": "I$^3$Net", "SAINT": "SAINT",
            "IR3C": "IR3C", "Linear": "Linear", "Bicubic": "Bicubic",
            "Cubic spline": "Cubic spline"}


# =========================================================================== Fig. 1
def fig_sampling():
    """Job: show the reader the acquisition problem before any method or result.

    (a) the seven frozen test-mask families at R=4 on the BraTS depth;
    (b) slice rejection from a uniform R=2 acquisition;
    (c) the four geometry profiles IR3C receives, for one irregular mask.
    """
    z = np.load(os.path.join(C.ROOT, "runs/IR3C_brats_ablation1/config/test_masks.npz"),
                allow_pickle=True)
    D = 153

    def as_ret(a):
        a = np.asarray(a)
        if a.dtype == bool or (a.size == D and set(np.unique(a)) <= {0, 1}):
            return np.flatnonzero(a)
        return np.sort(a.astype(int).ravel())

    sys.path.insert(0, os.path.join(C.ROOT, "IR3C_SliceReject"))
    sys.path.insert(0, C.ROOT)
    import sr_common as SR  # noqa: E402
    import nonuniform_masks as NM  # noqa: E402

    names = {"U0": "U0  uniform", "U1": "U1  mild jitter", "U2": "U2  strong jitter",
             "U3": "U3  random", "U4": "U4  clustered", "U5": "U5  centre-dense",
             "U6": "U6  mixed run lengths"}
    fig = plt.figure(figsize=(DOUBLE, 3.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[7, 3.6],
                          wspace=0.10, hspace=0.42, left=0.15, right=0.99, top=0.95,
                          bottom=0.10)
    ax = fig.add_subplot(gs[0, 0])
    for i, f in enumerate(C.FAMILIES):
        ret = as_ret(z[f"R4_{f}"])
        y = len(C.FAMILIES) - 1 - i
        ax.add_patch(Rectangle((-0.5, y - 0.36), D, 0.72, color="#eef1f4", lw=0))
        ax.vlines(ret, y - 0.36, y + 0.36, color=INK, lw=0.75)
        g = NM.gap_statistics(list(ret), D)
        ax.text(D + 2, y, f"{g['max_gap']:d}", va="center", ha="left", fontsize=6.5,
                color=INK2)
    ax.text(D + 2, len(C.FAMILIES) - 0.35, "max\ngap", va="bottom", ha="left",
            fontsize=6.5, color=MUTED, linespacing=0.95)
    ax.set_yticks(range(len(C.FAMILIES)))
    ax.set_yticklabels([names[f] for f in C.FAMILIES[::-1]], fontsize=7)
    ax.set_xlim(-1, D + 12)
    ax.set_ylim(-0.6, len(C.FAMILIES) - 0.4)
    ax.set_xticks([0, 50, 100, 150])
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Slice index (BraTS-GLI depth, 153 slices)")
    panel(ax, "a", x=-0.30)

    # (b) rejection
    axb = fig.add_subplot(gs[1, 0])
    base = NM.uniform_retained(D, 2)
    subj = "BraTS-GLI-00047-000"
    rows = [("Uniform, $R=2$", base, []),
            ("20% random", *SR.reject_mask(base, subj, 2, "random", 0.2)),
            ("20% burst", *SR.reject_mask(base, subj, 2, "burst", 0.2))]
    for i, (nm, keep, drop) in enumerate(rows):
        y = len(rows) - 1 - i
        axb.add_patch(Rectangle((-0.5, y - 0.36), D, 0.72, color="#eef1f4", lw=0))
        axb.vlines(keep, y - 0.36, y + 0.36, color=INK, lw=0.6)
        if drop:
            axb.vlines(drop, y - 0.36, y + 0.36, color=VERM, lw=1.0)
    axb.set_yticks(range(len(rows)))
    axb.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7)
    axb.set_xlim(-1, D + 12)
    axb.set_ylim(-0.6, len(rows) - 0.4)
    axb.set_xticks([0, 50, 100, 150])
    axb.tick_params(axis="y", length=0)
    axb.grid(False)
    axb.spines["left"].set_visible(False)
    axb.set_xlabel("Slice index")
    # colour key (ink = kept, vermillion = rejected) is stated in the caption
    panel(axb, "b", x=-0.30)

    # (c) profiles for the U3 mask, zoomed to a window
    axc = fig.add_subplot(gs[:, 1])
    ret = list(as_ret(z["R4_U3"]))
    import image_residual.masks as M  # noqa: E402
    prof = M.positional_profiles(ret, D)
    lo, hi = 40, 100
    zz = np.arange(lo, hi)
    gap = np.log1p(prof["local_gap"]) / np.log1p(160.0)
    near = np.log1p(prof["nearest"]) / np.log1p(80.0)
    off = [3.3, 2.2, 1.1, 0.0]
    series = [("known $k_z$", prof["known"], INK),
              (r"position $\alpha_z$", prof["alpha"], BLUE),
              ("gap length $g_z$", gap / max(gap[lo:hi].max(), 1e-6), VERM),
              ("distance $d_z$", near / max(near[lo:hi].max(), 1e-6), GREEN)]
    for (nm, v, col), o in zip(series, off):
        axc.step(zz, v[lo:hi] * 0.85 + o, where="mid", color=col, lw=1.1)
        axc.text(lo - 1.5, o + 0.42, nm, ha="right", va="center", fontsize=7, color=INK)
    for r in ret:
        if lo <= r < hi:
            axc.axvline(r, color="#c9ced6", lw=0.5, zorder=0)
    axc.set_xlim(lo - 0.5, hi - 0.5)
    axc.set_yticks([])
    axc.spines["left"].set_visible(False)
    axc.grid(False)
    axc.set_xlabel("Slice index (U3 mask, $R=4$)")
    panel(axc, "c", x=-0.30, y=1.0)
    axc.text(0.0, 1.0, "", transform=axc.transAxes)
    fig.subplots_adjust()
    # leave room for the profile names
    pos = axc.get_position()
    axc.set_position([pos.x0 + 0.10, pos.y0, pos.width - 0.10, pos.height])
    save(fig, "fig1_sampling")


# =========================================================================== Fig. 3
def fig_uniform():
    """Job: show how each learned method compares with linear interpolation as R grows,
    per dataset, on the uniform grid where every method is defined."""
    keys = ["DCSRN", "ArSSR", "I3Net", "SAINT", "IR3C"]
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.05), sharey=False)
    for j, (ax, ds) in enumerate(zip(axs, C.DATASETS)):
        specs = dict(C.method_specs(ds))
        for k_i, k in enumerate(keys):
            m, e = [], []
            for R in C.TRAIN_R:
                a = C.per_subject(ds, specs[SPEC_KEY[k]], ["U0"], [R])
                b = C.per_subject(ds, specs["Linear"], ["U0"], [R])
                d = (a - b).dropna()
                m.append(d.mean()); e.append(ci(d))
            plot_series(ax, [0, 1, 2], m, e, k, dx=(k_i - 2) * 0.045)
        ax.axhline(0, color=GREY, lw=0.9, zorder=1)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["2", "4", "8"])
        ax.set_xlim(-0.3, 2.3)
        ax.set_xlabel("Reduction factor $R$")
        if j == 0:
            ax.set_ylabel(r"$\Delta$PSNR vs. linear (dB)")
        ax.set_title(C.DS_LABEL[ds], loc="left", fontsize=7.5, color=INK, pad=3)
        panel(ax, "abc"[j], x=-0.04, y=1.0)
    axs[1].legend(handles=legend_handles(keys + ["Linear"]), ncol=6, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30), columnspacing=1.3)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig3_uniform")


# =========================================================================== Fig. 4
def fig_families():
    """Job: show that the IR3C margin holds on every irregular family and is largest where
    gaps are longest, alongside the methods that can be run there."""
    keys = ["Cubic spline", "DCSRN", "ArSSR", "IR3C"]
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.1), sharey=False)
    x = np.arange(len(C.FAMILIES))
    for j, (ax, ds) in enumerate(zip(axs, C.DATASETS)):
        specs = dict(C.method_specs(ds))
        for k_i, k in enumerate(keys):
            m, e = [], []
            for f in C.FAMILIES:
                a = C.per_subject(ds, specs[SPEC_KEY[k]], [f])
                b = C.per_subject(ds, specs["Linear"], [f])
                d = (a - b).dropna()
                m.append(d.mean()); e.append(ci(d))
            plot_series(ax, x, m, e, k, dx=(k_i - 1.5) * 0.07)
        ax.axhline(0, color=GREY, lw=0.9, zorder=1)
        ax.axvspan(-0.5, 0.5, color="#f3f5f7", zorder=0, lw=0)
        ax.set_xticks(x); ax.set_xticklabels(C.FAMILIES)
        ax.set_xlim(-0.5, len(C.FAMILIES) - 0.5)
        ax.set_xlabel("Sampling family")
        if j == 0:
            ax.set_ylabel(r"$\Delta$PSNR vs. linear (dB)")
        ax.set_title(C.DS_LABEL[ds], loc="left", fontsize=7.5, color=INK, pad=3)
        panel(ax, "abc"[j], x=-0.04, y=1.0)
    axs[1].legend(handles=legend_handles(keys + ["Linear"]), ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30), columnspacing=1.3)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig4_families")


# =========================================================================== Fig. 5
def fig_rejection():
    """Job: show the crossover. Absolute PSNR against the rejected fraction, per dataset,
    base factor and rejection pattern, for the methods that matter."""
    meth = [("linear", "Linear"), ("ArSSR", "ArSSR"), ("I3Net_impute", "I3Net"),
            ("SAINT_impute", "SAINT"), ("SAINT_subset", "SAINT-B"), ("IR3C", "IR3C")]
    ps = [0.0, 0.1, 0.2, 0.3]
    combos = [(2, "random"), (2, "burst"), (4, "random"), (4, "burst")]
    dss = ["BraTS", "IXI"]
    fig, axs = plt.subplots(len(dss), 4, figsize=(DOUBLE, 3.9), sharex=True)
    for i, ds in enumerate(dss):
        for j, (R, pat) in enumerate(combos):
            ax = axs[i, j]
            for k_i, (m, key) in enumerate(meth):
                mu, e = [], []
                for p in ps:
                    s = C.sr_per_subject(ds, m, R, pat, p)
                    mu.append(s.mean()); e.append(ci(s))
                plot_series(ax, np.array(ps) * 100, mu, e, key,
                            label=("SAINT + imputation" if key == "SAINT" else
                                   "I$^3$Net + imputation" if key == "I3Net" else None),
                            dx=(k_i - 2.5) * 0.35)
            ax.set_xticks([0, 10, 20, 30])
            ax.set_xlim(-2.5, 32.5)
            if i == len(dss) - 1:
                ax.set_xlabel("Rejected slices (%)")
            if j == 0:
                ax.set_ylabel(f"{C.DS_LABEL[ds]}\nPSNR (dB)")
            if i == 0:
                ax.set_title(f"$R={R}$, {pat}", fontsize=7.5, color=INK)
            panel(ax, "abcdefgh"[i * 4 + j], x=-0.06)
    hs = legend_handles(["Linear", "ArSSR", "I3Net", "SAINT", "SAINT-B", "IR3C"])
    hs[2].set_label("I$^3$Net + imputation"); hs[3].set_label("SAINT + imputation")
    fig.legend(handles=hs, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.035),
               columnspacing=1.2)
    fig.subplots_adjust(wspace=0.32, hspace=0.22, bottom=0.14)
    save(fig, "fig5_rejection")


# =========================================================================== Fig. 6
def fig_downstream():
    """Job: show that downstream consistency does not follow PSNR: SAINT is close to IR3C
    in PSNR but falls away from the complete-input segmentation as R grows."""
    import pandas as pd
    d = pd.read_csv(os.path.join(C.ROOT, "runs/IR3C_MedIA_P2_matched_downstream/"
                                         "downstream_segmentation_per_volume.csv"))
    floor = d[d.method == "gt_roundtrip"]
    meth = [("linear", "Linear"), ("DCSRN_mixed", "DCSRN"), ("ArSSR", "ArSSR"),
            ("I3Net", "I3Net"), ("SAINT", "SAINT"), ("IR3_matched", "IR3C")]
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.05))
    for j, (ax, k) in enumerate(zip(axs, ("WT", "TC", "ET"))):
        for k_i, (m, key) in enumerate(meth):
            mu, e = [], []
            for R in C.TRAIN_R:
                s = d[(d.method == m) & (d.R == R)][f"{k}_Dice"].dropna()
                mu.append(s.mean()); e.append(ci(s))
            plot_series(ax, [0, 1, 2], mu, e, key, dx=(k_i - 2.5) * 0.045)
        fl = floor[f"{k}_Dice"].mean()
        ax.axhline(fl, color=INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["2", "4", "8"])
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(0.6, 1.01)
        ax.set_xlabel("Reduction factor $R$")
        if j == 0:
            ax.set_ylabel("Dice vs. complete-input prediction")
        ax.set_title({"WT": "Whole tumour", "TC": "Tumour core",
                      "ET": "Enhancing tumour"}[k], loc="left", fontsize=7.5, pad=3)
        panel(ax, "abc"[j], x=-0.04, y=1.0)
    hs = legend_handles(["Linear", "DCSRN", "ArSSR", "I3Net", "SAINT", "IR3C"])
    hs.append(Line2D([], [], color=INK2, lw=0.7, ls=(0, (1, 1.5)),
                     label="complete input, save\u2013reload"))
    axs[1].legend(handles=hs,
                  ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.30), columnspacing=0.9)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig6_downstream")


# =========================================================================== qualitative
SR_NII = os.path.join(C.ROOT, "runs/IR3C_SliceReject_all/BraTS/nifti")
QUAL_SUBJ = "BraTS-GLI-00047-000"   # median IR3C gain over SAINT-A (R4, 30% burst) of 5


def _nii(*p):
    return np.asanyarray(nib.load(os.path.join(SR_NII, *p)).dataobj).astype(np.float32)


def _crop_box(gt, axis):
    m = (gt > 0.02).any(axis=axis)
    ys, xs = np.where(m)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def fig_qualitative():
    """Job: show what the numbers mean on one typical subject along the reconstruction
    axis, where through-plane quality is visible: uniform R=4, then the same acquisition
    after 30% burst rejection. Error is shown on missing slices only, because acquired
    slices are copied exactly by every method."""
    cols = [("linear", "Linear"), ("ArSSR", "ArSSR"), ("I3Net_impute", "I$^3$Net"),
            ("SAINT_impute", "SAINT"), ("IR3C", "IR3C (ours)")]
    conds = [("R4_none_p00", "$R=4$, uniform"), ("R4_burst_p30", "$R=4$, 30% burst")]
    masks = json.load(open(os.path.join(SR_NII, "masks", f"{QUAL_SUBJ}.json")))
    gt = _nii("GT", f"{QUAL_SUBJ}.nii.gz")
    fg = gt > 0
    D = gt.shape[2]
    # rule fixed in advance: the sagittal plane with the largest foreground area
    x_sag = int(np.argmax(fg.sum(axis=(1, 2))))
    sag_gt = gt[x_sag, :, :].T                            # rows = slice index z
    cols_fg = np.where((sag_gt > 0.02).any(axis=0))[0]
    rows_fg = np.where((sag_gt > 0.02).any(axis=1))[0]
    r0, r1 = max(rows_fg.min() - 2, 0), min(rows_fg.max() + 3, D)
    c0, c1 = max(cols_fg.min() - 2, 0), cols_fg.max() + 3
    sag_gt = sag_gt[r0:r1, c0:c1]

    data = {}
    for ck, _ in conds:
        keep = np.array(masks[ck]["keep"])
        miss = np.ones(D, bool); miss[keep] = False
        miss = miss[r0:r1]
        for m, _ in cols:
            rec = _nii(ck, m, f"{QUAL_SUBJ}.nii.gz")[x_sag, :, :].T[r0:r1, c0:c1]
            e = np.abs(rec - sag_gt)
            valid = (sag_gt > 0) & miss[:, None]
            e_show = np.where(valid, e, np.nan)
            data[(ck, m)] = (rec, e_show, float(e[valid].mean()))
        data[ck] = (keep, masks[ck]["drop"])
    allerr = np.concatenate([data[(ck, m)][1][np.isfinite(data[(ck, m)][1])]
                             for ck, _ in conds for m, _ in cols])
    vmax = float(np.percentile(allerr, 99))

    ncol = len(cols) + 1
    h = sag_gt.shape[0] / sag_gt.shape[1]
    wcol = (DOUBLE - 0.55) / ncol
    fig = plt.figure(figsize=(DOUBLE, wcol * h * 4 + 0.45))
    gs = fig.add_gridspec(4, ncol + 1, width_ratios=[1] * ncol + [0.05],
                          wspace=0.03, hspace=0.04, left=0.06, right=0.95, top=0.955,
                          bottom=0.005)
    cmap_err = plt.get_cmap("inferno").copy()
    cmap_err.set_bad("black")

    def show(r, c, img, cmap="gray", vmax_=1.0):
        a = fig.add_subplot(gs[r, c])
        im = a.imshow(img, cmap=cmap, vmin=0, vmax=vmax_, origin="lower",
                      interpolation="nearest", aspect="equal")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(False)
        return a, im

    for i, (ck, lab) in enumerate(conds):
        keep, drop = data[ck]
        a, _ = show(2 * i, 0, sag_gt)
        if i == 0:
            a.set_title("Reference", fontsize=7.5)
        a.set_ylabel(lab, fontsize=7)
        # acquired slices (ink) and rejected slices (vermillion) as ticks on the left edge
        w = sag_gt.shape[1]
        for zk in keep:
            if r0 <= zk < r1:
                a.plot([-0.5, 0.035 * w], [zk - r0, zk - r0], color="#9aa3ad", lw=0.45)
        for zd in drop:
            if r0 <= zd < r1:
                a.plot([-0.5, 0.07 * w], [zd - r0, zd - r0], color=VERM, lw=0.9)
        a2 = fig.add_subplot(gs[2 * i + 1, 0]); a2.axis("off")
        if i == 1:
            a2.text(0.02, 0.62, "left-edge ticks:", fontsize=6.2, color=INK2,
                    transform=a2.transAxes)
            a2.text(0.02, 0.44, "grey  acquired, kept", fontsize=6.2, color=INK2,
                    transform=a2.transAxes)
            a2.text(0.02, 0.26, "orange  acquired, rejected", fontsize=6.2, color=VERM,
                    transform=a2.transAxes)
        for j, (m, nm) in enumerate(cols, start=1):
            rec, e, mae = data[(ck, m)]
            a, _ = show(2 * i, j, rec)
            if i == 0:
                a.set_title(nm, fontsize=7.5, fontweight="bold" if m == "IR3C" else "normal")
            a, im = show(2 * i + 1, j, e, cmap=cmap_err, vmax_=vmax)
            a.text(0.97, 0.04, f"MAE {mae:.4f}", transform=a.transAxes, ha="right",
                   va="bottom", color="white", fontsize=6)
        cax = fig.add_subplot(gs[2 * i + 1, ncol])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0.4)
        cb.set_label("|error|", fontsize=6.5)
    save(fig, "fig7_qualitative")
    return dict(x_sag=x_sag, vmax=vmax, n_drop=len(masks["R4_burst_p30"]["drop"]),
                mae={f"{ck}:{m}": data[(ck, m)][2] for ck, _ in conds for m, _ in cols})


# =========================================================================== placeholder
def fig_pending_gap():
    """Placeholder panel layout for the pending error-versus-local-gap analysis. Draws
    the axes and the analysis plan only; no data."""
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.0))
    for j, (ax, xl) in enumerate(zip(axs, ("Local gap length $g_z$ (slices)",
                                           "Distance to nearest kept slice $d_z$ (slices)"))):
        ax.set_xlim(0, 24 if j == 0 else 12)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel(xl)
        ax.set_ylabel("Per-slice PSNR (dB)" if j == 0 else "")
        ax.grid(False)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#fff3e0",
                               edgecolor="#e0a050", lw=0.8, ls="--", zorder=0))
        ax.text(0.5, 0.5, "PENDING\nper-slice errors binned by gap geometry,\n"
                          "all methods, U0--U6 and rejection masks pooled",
                transform=ax.transAxes, ha="center", va="center", fontsize=7,
                color="#9a5b12")
        panel(ax, "ab"[j], x=-0.04)
    fig.subplots_adjust(wspace=0.18)
    save(fig, "fig8_gap_pending")


if __name__ == "__main__":
    fig_sampling()
    fig_uniform()
    fig_families()
    fig_rejection()
    fig_downstream()
    info = fig_qualitative()
    fig_pending_gap()
    with open(os.path.join(C.PAPER, "scripts", "figure_provenance.json"), "w") as fh:
        json.dump(dict(qualitative_subject=QUAL_SUBJ, **info), fh, indent=1)
