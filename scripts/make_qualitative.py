"""
IR3C_MedIA/scripts/make_qualitative.py -- qualitative figures beyond Fig. 7.

    fig9_cubes          3-D cut-away cubes, 30% burst rejection at R=4, all three datasets
    fig10_irregular     reformats along the reconstruction axis on irregular masks
                        (U4 at R=4, U3 at R=8), all three datasets
    fig11_seg           downstream failure mode: background haze and segmentation, BraTS

Sources (read-only):
    runs/IR3C_SliceReject_all/<DS>/nifti        archived showcase reconstructions (5/dataset)
    runs/IR3C_MedIA_P10_qual_irregular/<DS>     inference-only irregular-mask reconstructions
    runs/IR3C_LPIPS/downstream, runs/brats_ir3c_downstream_20260907_122522,
    runs/IR3C_MedIA_P2_matched_downstream       archived restored FLAIR + predictions
    runs/IR3C_MedIA_P9_bgmask_downstream        background-masked re-segmentation

Subjects are chosen by rules fixed before looking at the images (stated in each caption).
The reconstruction axis is vertical in every panel.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import nibabel as nib
import numpy as np
import pandas as pd

import common as C
import make_figures as F
from make_figures import DOUBLE, INK, INK2, VERM, save

SRA = os.path.join(C.ROOT, "runs/IR3C_SliceReject_all")
P10 = os.path.join(C.ROOT, "runs/IR3C_MedIA_P10_qual_irregular")
DS_LABEL = {"BraTS": "BraTS-GLI (T2-FLAIR)", "OAIZIB": "OAIZIB (knee DESS)",
            "IXI": "IXI-tiny (T1w)"}
RECON_AXIS = {"BraTS": 2, "OAIZIB": 0, "IXI": 0}      # through-plane axis of the NIfTI
CMAP_ERR = plt.get_cmap("inferno").copy()
CMAP_ERR.set_bad("black")


# ------------------------------------------------------------------ data access
def load_sra(ds, *parts):
    """Archived showcase volume as (X, Y, Z) with Z = reconstruction axis, plus spacing."""
    img = nib.load(os.path.join(SRA, ds, "nifti", *parts))
    v = np.asanyarray(img.dataobj).astype(np.float32)
    sp = np.asarray(img.header.get_zooms()[:3], float)
    ax = RECON_AXIS[ds]
    order = [a for a in range(3) if a != ax] + [ax]
    return np.transpose(v, order), sp[order]


def sra_mask(ds, subj, cond):
    m = json.load(open(os.path.join(SRA, ds, "nifti", "masks", f"{subj}.json")))[cond]
    return list(m["keep"]), list(m["drop"])


def showcase_subject(ds, cond="R4_burst_p30", rival="SAINT_impute"):
    """Rule fixed in advance (as for Fig. 7): among the five showcase subjects, the one
    with the median IR3C gain over the rival in this condition."""
    m = pd.read_csv(os.path.join(SRA, ds, "metrics.csv"))
    subs = sorted(os.listdir(os.path.join(SRA, ds, "nifti", "GT")))
    subs = [s.replace(".nii.gz", "") for s in subs]
    m = m[(m.cond == cond) & m.subject.isin(subs)]
    g = (m[m.method == "IR3C"].set_index("subject").psnr
         - m[m.method == rival].set_index("subject").psnr).sort_values()
    return g.index[len(g) // 2], g


def fg_box(gt, thr=0.02, pad=2):
    idx = np.where(gt > thr)
    lo = [max(int(i.min()) - pad, 0) for i in idx]
    hi = [min(int(i.max()) + pad + 1, n) for i, n in zip(idx, gt.shape)]
    return lo, hi


# ------------------------------------------------------------------ cube drawing
COS, SIN, DEPTH = np.cos(np.deg2rad(32)), np.sin(np.deg2rad(32)), 0.55


def draw_cube(ax, vol, sp, box, corner, cmap="gray", vmin=0.0, vmax=1.0, ticks=None):
    """Cut-away block with its corner at `corner` = (x1, y0, z0).

    The block spans x in [xa, x1], y in [y0, yb], z in [za, z0]; its three visible faces
    are the interior planes y = y0 (front, coronal for brain), x = x1 (right) and z = z0
    (top, a slice orthogonal to the reconstruction axis).  Physical spacing is respected;
    depth is drawn obliquely (cabinet projection).
    """
    (xa, ya, za), (xb, yb, zb) = box
    x1, y0, z0 = corner
    sx, sy, sz = sp
    W, Dp, H = (x1 - xa) * sx, (yb - y0) * sy * DEPTH, (z0 - za) * sz
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="antialiased",
              resample=True)
    # front face: y = y0, u = x, v = z
    front = vol[xa:x1, y0, za:z0].T
    tf = Affine2D.from_values(sx, 0, 0, sz, 0, 0)
    ax.imshow(front, extent=(0, front.shape[1], 0, front.shape[0]),
              transform=tf + ax.transData, **kw)
    # top face: z = z0, u = x, v = y
    top = vol[xa:x1, y0:yb, z0].T
    tt = Affine2D.from_values(sx, 0, sy * DEPTH * COS, sy * DEPTH * SIN, 0, H)
    ax.imshow(top, extent=(0, top.shape[1], 0, top.shape[0]),
              transform=tt + ax.transData, **kw)
    # right face: x = x1, u = y, v = z
    right = vol[x1, y0:yb, za:z0].T
    tr = Affine2D.from_values(sy * DEPTH * COS, sy * DEPTH * SIN, 0, sz, W, 0)
    ax.imshow(right, extent=(0, right.shape[1], 0, right.shape[0]),
              transform=tr + ax.transData, **kw)
    # edges
    ox, oy = Dp * COS, Dp * SIN
    e = dict(color="#c9ced6", lw=0.45)
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], **e)
    ax.plot([0, ox, W + ox, W], [H, H + oy, H + oy, H], **e)
    ax.plot([W, W + ox, W + ox], [0, oy, H + oy], **e)
    if ticks is not None:                     # slice pattern on the left edge of the front
        keep, drop = ticks
        L = 0.045 * W
        for z in keep:
            if za <= z < z0:
                ax.plot([-L, 0], [(z - za + 0.5) * sz] * 2, color="#9aa3ad", lw=0.35)
        for z in drop:
            if za <= z < z0:
                ax.plot([-1.8 * L, 0], [(z - za + 0.5) * sz] * 2, color=VERM, lw=0.7)
    ax.set_xlim(-0.10 * W, W + ox + 0.02 * W)
    ax.set_ylim(-0.02 * H, H + oy + 0.02 * H)
    ax.set_aspect("equal")
    ax.axis("off")


def corner_for(gt, keep, box):
    """Rule: in-plane corner at the foreground centroid; top face at the missing slice in
    the centre of the longest gap that lies inside the foreground (where the methods
    differ most), restricted to the middle band (45-80%) of the foreground extent along
    the reconstruction axis, so the block shows most of the anatomy."""
    lo, hi = box
    idx = np.array(np.where(gt > 0.02))
    cx, cy = int(np.round(idx[0].mean())), int(np.round(idx[1].mean()))
    k = sorted(keep)
    zmin = lo[2] + 0.45 * (hi[2] - lo[2])
    zmax = lo[2] + 0.80 * (hi[2] - lo[2])
    best, z0 = -1, None
    for a, b in zip(k[:-1], k[1:]):
        if b - a > 1 and zmin <= (a + b) / 2 <= zmax:
            if b - a > best:
                best, z0 = b - a, (a + b) // 2
    return (cx, cy, z0), best


# ------------------------------------------------------------------ Fig. 9 cubes
CUBE_COLS = [("linear", "Linear"), ("SAINT_impute", "SAINT + imputation"),
             ("ArSSR", "ArSSR"), ("IR3C", "IR3C (ours)")]


def fig_cubes(cond="R4_burst_p30"):
    """Job: show the whole reconstructed volume, not one plane, after heavy rejection.
    Image row: reference and four methods.  Error row: the acquired input (what every
    method receives) and |error| on missing slices for the same four methods."""
    info = {}
    nrow = 2 * 3
    ncol = 1 + len(CUBE_COLS)
    fig = plt.figure(figsize=(DOUBLE, 7.6))
    gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1] * ncol + [0.045],
                          wspace=0.02, hspace=0.06, left=0.045, right=0.955, top=0.965,
                          bottom=0.01)
    for d, ds in enumerate(("BraTS", "OAIZIB", "IXI")):
        subj, gains = showcase_subject(ds, cond)
        gt, sp = load_sra(ds, "GT", f"{subj}.nii.gz")
        keep, drop = sra_mask(ds, subj, cond)
        box = fg_box(gt)
        corner, gap = corner_for(gt, keep, box)
        D = gt.shape[2]
        miss = np.ones(D, bool)
        miss[keep] = False
        recs = {m: load_sra(ds, cond, m, f"{subj}.nii.gz")[0] for m, _ in CUBE_COLS}
        errs, mae = {}, {}
        valid = (gt > 0) & miss[None, None, :]
        for m, _ in CUBE_COLS:
            e = np.abs(recs[m] - gt)
            mae[m] = float(e[valid].mean())
            errs[m] = np.where(valid, e, 0.0)     # zeros, not NaN: resampling spreads NaN
        vmax = float(np.percentile(np.concatenate([errs[m][valid] for m, _ in CUBE_COLS]),
                                   99))
        xs = np.where(miss[None, None, :], 0.0, gt)
        r = 2 * d
        a = fig.add_subplot(gs[r, 0])
        draw_cube(a, gt, sp, box, corner, ticks=(keep, drop))
        a.text(-0.06, 0.5, DS_LABEL[ds], transform=a.transAxes, rotation=90, ha="right",
               va="center", fontsize=7, color=INK)
        if d == 0:
            a.set_title("Reference", fontsize=7.5)
        a = fig.add_subplot(gs[r + 1, 0])
        draw_cube(a, xs, sp, box, corner, ticks=(keep, drop))
        a.text(0.5, -0.02, "acquired input", transform=a.transAxes, ha="center", va="top",
               fontsize=6.3, color=INK2)
        for j, (m, nm) in enumerate(CUBE_COLS, start=1):
            a = fig.add_subplot(gs[r, j])
            draw_cube(a, recs[m], sp, box, corner)
            if d == 0:
                a.set_title(nm, fontsize=7.5, fontweight="bold" if m == "IR3C" else "normal")
            a = fig.add_subplot(gs[r + 1, j])
            draw_cube(a, errs[m], sp, box, corner, cmap=CMAP_ERR, vmax=vmax)
            a.text(0.5, -0.02, f"MAE {mae[m]:.4f}", transform=a.transAxes, ha="center",
                   va="top", fontsize=6.3, color=INK2)
        cax = fig.add_subplot(gs[r + 1, ncol])
        sm = plt.cm.ScalarMappable(cmap=CMAP_ERR, norm=plt.Normalize(0, vmax))
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.outline.set_linewidth(0.4)
        cb.set_label("|error|", fontsize=6)
        info[ds] = dict(subject=subj, corner=[int(c) for c in corner], gap=int(gap),
                        n_drop=len(drop), vmax=vmax, mae=mae,
                        gain_vs_saint=gains.round(3).to_dict())
    save(fig, "fig9_cubes")
    return info


# ------------------------------------------------------------------ Fig. 10 irregular
IRR_COLS = [("linear", "Linear"), ("bicubic", "Bicubic"), ("DCSRNmix", "DCSRN"),
            ("ArSSR", "ArSSR"), ("IR3C", "IR3C (ours)")]


def p10_load(ds, *parts):
    v = np.load(os.path.join(P10, ds, *parts)).astype(np.float32)
    return v


def p10_subject(ds, cond="R4_U4"):
    """Rule: median IR3C gain over the best other method in this condition, of five."""
    g = pd.read_csv(os.path.join(P10, "gate.csv"))
    g = g[(g.ds == ds) & (g.cond == cond)]
    p = g.pivot_table(index="subject", columns="method", values="psnr")
    gain = (p["IR3C"] - p.drop(columns="IR3C").max(axis=1)).sort_values()
    return gain.index[len(gain) // 2], gain


def fig_irregular():
    """Job: show irregular sampling, where SAINT and I3Net cannot run, on all three
    datasets.  One reformat per dataset along the reconstruction axis through the
    foreground centroid; for each of two masks an image row and an error row."""
    conds = [("R4_U4", "U4 clustered, $R=4$"), ("R8_U3", "U3 random, $R=8$")]
    ncol = 1 + len(IRR_COLS)
    blocks = []
    for ds in ("BraTS", "OAIZIB", "IXI"):
        subj, gain = p10_subject(ds)
        gt = p10_load(ds, "GT", f"{subj}.npy")
        masks = json.load(open(os.path.join(P10, ds, "masks.json")))
        # plane: through the foreground centroid, orthogonal to the first in-plane axis
        fg = gt > 0.02
        x = int(np.argmax(fg.sum(axis=(1, 2))))
        sl = gt[x].T                                    # rows = reconstruction axis
        rows = np.where(fg[x].any(axis=0))[0]
        cols_ = np.where(fg[x].any(axis=1))[0]
        r0, r1 = max(rows.min() - 1, 0), rows.max() + 2
        c0, c1 = max(cols_.min() - 1, 0), cols_.max() + 2
        blocks.append((ds, subj, gain, gt, masks, x, (r0, r1, c0, c1)))

    # physical aspect of each reformat (row spacing = reconstruction axis)
    SP = {"BraTS": (1.0, 1.0), "OAIZIB": (0.70, 0.3646), "IXI": (2.18, 3.95)}
    fig = plt.figure(figsize=(DOUBLE, 8.4))
    heights = []
    for ds, _, _, gt, _, x, (r0, r1, c0, c1) in blocks:
        h = (r1 - r0) * SP[ds][0] / ((c1 - c0) * SP[ds][1])
        heights += [h] * 4
    gs = fig.add_gridspec(12, ncol + 1, width_ratios=[1] * ncol + [0.05],
                          height_ratios=heights, wspace=0.03, hspace=0.05, left=0.075,
                          right=0.945, top=0.975, bottom=0.005)
    info = {}
    for b, (ds, subj, gain, gt, masks, x, (r0, r1, c0, c1)) in enumerate(blocks):
        ref = gt[x].T[r0:r1, c0:c1]
        asp = SP[ds][0] / SP[ds][1]
        info[ds] = dict(subject=subj, plane=x, mae={})
        for ci, (ck, lab) in enumerate(conds):
            keep = masks[ck]
            miss = np.ones(gt.shape[2], bool)
            miss[keep] = False
            mrow = miss[r0:r1]
            data, errs = {}, []
            for m, _ in IRR_COLS:
                p = os.path.join(P10, ds, ck, m, f"{subj}.npy")
                if not os.path.exists(p):
                    data[m] = None
                    continue
                rec = np.load(p).astype(np.float32)[x].T[r0:r1, c0:c1]
                valid = (ref > 0) & mrow[:, None]
                e = np.abs(rec - ref)
                data[m] = (rec, np.where(valid, e, np.nan), float(e[valid].mean()))
                errs.append(e[valid])
            vmax = float(np.percentile(np.concatenate(errs), 99))
            r = 4 * b + 2 * ci

            def show(rr, cc, img, cmap="gray", vm=1.0):
                a = fig.add_subplot(gs[rr, cc])
                a.imshow(img, cmap=cmap, vmin=0, vmax=vm, origin="lower", aspect=asp,
                         interpolation="nearest")
                a.set_xticks([]); a.set_yticks([]); a.grid(False)
                for s in a.spines.values():
                    s.set_visible(False)
                return a

            a = show(r, 0, ref)
            w = ref.shape[1]
            for z in keep:
                if r0 <= z < r1:
                    a.plot([-0.5, 0.05 * w], [z - r0, z - r0], color="#9aa3ad", lw=0.4)
            a.set_ylabel(lab, fontsize=6.3)
            if ci == 0:
                a.text(-0.30, 0.0, DS_LABEL[ds], transform=a.transAxes, rotation=90,
                       ha="right", va="center", fontsize=7, color=INK)
            if b == 0 and ci == 0:
                a.set_title("Reference", fontsize=7.5)
            a2 = fig.add_subplot(gs[r + 1, 0]); a2.axis("off")
            for j, (m, nm) in enumerate(IRR_COLS, start=1):
                if data[m] is None:
                    for rr in (r, r + 1):
                        aa = fig.add_subplot(gs[rr, j]); aa.axis("off")
                    fig.axes[-2].text(0.5, 0.5, "no $\\times$8 model", ha="center",
                                      va="center", fontsize=6, color=INK2,
                                      transform=fig.axes[-2].transAxes)
                    continue
                rec, e, mae = data[m]
                a = show(r, j, rec)
                if b == 0 and ci == 0:
                    a.set_title(nm, fontsize=7.5,
                                fontweight="bold" if m == "IR3C" else "normal")
                a = show(r + 1, j, e, cmap=CMAP_ERR, vm=vmax)
                a.text(0.97, 0.04, f"{mae:.4f}", transform=a.transAxes, ha="right",
                       va="bottom", color="white", fontsize=5.8)
                info[ds]["mae"][f"{ck}:{m}"] = mae
            cax = fig.add_subplot(gs[r + 1, ncol])
            sm = plt.cm.ScalarMappable(cmap=CMAP_ERR, norm=plt.Normalize(0, vmax))
            cb = fig.colorbar(sm, cax=cax)
            cb.ax.tick_params(labelsize=5, length=1.5)
            cb.outline.set_linewidth(0.4)
    save(fig, "fig10_irregular")
    return info


# ------------------------------------------------------------------ Fig. 11 segmentation
DSR = {"linear": "brats_ir3c_downstream_20260907_122522",
       "I3Net": "brats_ir3c_downstream_20260907_122522",
       "SAINT": "IR3C_LPIPS/downstream",
       "IR3_matched": "IR3C_MedIA_P2_matched_downstream"}
SHOW = os.path.join(C.ROOT, "runs/IR3C_MedIA_P9_bgmask_downstream/showcase")
LAB_COL = {1: "#0072B2", 2: "#009E73", 3: "#E69F00"}     # NCR, oedema, enhancing


def _nii_arr(p):
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)


def fig_seg():
    """Job: show why a reconstruction with reasonable PSNR can move the segmenter.
    Top row: voxels of the head-surrounding air that the restored FLAIR makes nonzero
    (orange), which the segmenter's nonzero z-scoring then treats as tissue.
    Bottom row: the frozen segmenter's prediction from each FLAIR."""
    info = json.load(open(os.path.join(SHOW, "showcase.json")))
    S, R = info["subject"], info["R"]
    import sys
    sys.path.insert(0, os.path.join(C.ROOT, "downstream_ir3c"))
    import ds_common as DC  # noqa: E402
    raw = _nii_arr(DC.modality_paths(S)["t2f"])
    bg = raw <= 0
    run = lambda rd, *a: os.path.join(C.ROOT, "runs", rd, *a)       # noqa: E731
    ref_lab = _nii_arr(run("IR3C_MedIA_P2_matched_downstream", "pred_seg", S,
                           f"{S}_original_complete_R0-pred-seg.nii.gz"))
    cols = [("ref", "Complete FLAIR"), ("linear", "Linear"), ("I3Net", "I$^3$Net"),
            ("SAINT", "SAINT"), ("IR3_matched", "IR3C (ours)"),
            ("SAINT_bg0", "SAINT, background\nset to zero")]
    vols, labs = {}, {}
    for m, _ in cols:
        if m == "ref":
            vols[m], labs[m] = raw, ref_lab
        elif m == "SAINT_bg0":
            vols[m] = _nii_arr(os.path.join(SHOW, f"{S}_SAINT_R{R}_bgzero-t2f.nii.gz"))
            labs[m] = _nii_arr(os.path.join(SHOW, f"{S}_SAINT_R{R}_bgzero-pred-seg.nii.gz"))
        else:
            vols[m] = _nii_arr(run(DSR[m], "recon_nifti", S, f"{S}_R{R}_U0_{m}-t2f.nii.gz"))
            labs[m] = _nii_arr(run(DSR[m], "pred_seg", S, f"{S}_{m}_R{R}-pred-seg.nii.gz"))
    # rule: the reconstructed (non-acquired) axial slice with the largest reference tumour
    recon_z = np.where(np.abs(vols["SAINT"] - raw).max(axis=(0, 1)) > 0)[0]
    area = (ref_lab[:, :, recon_z] > 0).sum(axis=(0, 1))
    z = int(recon_z[np.argmax(area)])

    def dice(a, b):
        a, b = a > 0, b > 0
        return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)

    wt = {m: float(dice(labs[m], ref_lab)) for m, _ in cols}
    frac = {m: float((vols[m][bg] > 0).mean()) for m, _ in cols}
    box = np.where((raw[:, :, z] > 0))
    x0, x1 = max(box[0].min() - 12, 0), box[0].max() + 13
    y0, y1 = max(box[1].min() - 12, 0), box[1].max() + 13
    hi = float(np.percentile(raw[raw > 0], 99.5))

    fig, axs = plt.subplots(2, len(cols), figsize=(DOUBLE, 2.75),
                            gridspec_kw=dict(wspace=0.03, hspace=0.05))
    for j, (m, nm) in enumerate(cols):
        v = vols[m][x0:x1, y0:y1, z].T
        haze = (bg[x0:x1, y0:y1, z] & (vols[m][x0:x1, y0:y1, z] > 0)).T
        a = axs[0, j]
        a.imshow(v, cmap="gray", vmin=0, vmax=hi, origin="lower", interpolation="nearest")
        ov = np.zeros(haze.shape + (4,))
        ov[haze] = (0.835, 0.369, 0.0, 0.9)
        a.imshow(ov, origin="lower", interpolation="nearest")
        a.set_title(nm, fontsize=7, fontweight="bold" if m == "IR3_matched" else "normal")
        a.text(0.03, 0.03, f"air $>0$: {100 * frac[m]:.0f}%", transform=a.transAxes,
               fontsize=5.8, color="white", va="bottom")
        a = axs[1, j]
        a.imshow(raw[x0:x1, y0:y1, z].T, cmap="gray", vmin=0, vmax=hi, origin="lower",
                 interpolation="nearest")
        lab = labs[m][x0:x1, y0:y1, z].T
        ov = np.zeros(lab.shape + (4,))
        for k, c in LAB_COL.items():
            rgb = tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
            ov[lab == k] = rgb + (0.6,)
        a.imshow(ov, origin="lower", interpolation="nearest")
        a.text(0.03, 0.03, f"WT Dice {wt[m]:.3f}", transform=a.transAxes, fontsize=5.8,
               color="white", va="bottom")
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp_ in a.spines.values():
            sp_.set_visible(False)
    axs[0, 0].set_ylabel("restored FLAIR", fontsize=6.5)
    axs[1, 0].set_ylabel("segmentation", fontsize=6.5)
    from matplotlib.patches import Patch
    hs = [Patch(color="#D55E00", label="air voxel made nonzero"),
          Patch(color=LAB_COL[1], label="necrotic core"),
          Patch(color=LAB_COL[2], label="oedema"),
          Patch(color=LAB_COL[3], label="enhancing tumour")]
    fig.legend(handles=hs, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               fontsize=6.5, handlelength=1.2)
    save(fig, "fig11_seg")
    return dict(subject=S, R=R, z=z, wt_dice=wt, air_nonzero=frac)


if __name__ == "__main__":
    out = dict(cubes=fig_cubes(), irregular=fig_irregular(), seg=fig_seg())
    json.dump(out, open(os.path.join(C.PAPER, "scripts", "qualitative_provenance.json"), "w"),
              indent=1, default=float)
