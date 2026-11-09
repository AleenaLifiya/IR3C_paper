# IR3C — Medical Image Analysis manuscript (draft 2)

Elsevier CAS double-column (`cas-dc`), author–year references (`cas-model2-names`).
Rewritten from `IR3C_paper_draft1/` (which is left untouched).

```bash
./compile.sh          # regenerate all tables + figures from archived results, then typeset
./compile.sh --tex    # typeset only
```

Output: `main.pdf` and a copy `IR3C_MedIA_draft2.pdf`.

## Layout

| Path | What |
|---|---|
| `main.tex` | manuscript (front matter, highlights, graphical abstract, body, declarations, appendix) |
| `references.bib` | draft-1 bibliography + 11 added entries |
| `scripts/common.py` | data layer: loaders, aggregation (within subject first), paired stats, Holm |
| `scripts/make_tables.py` | writes every `tables/*.tex`; also `scripts/numbers_digest.txt` (every number quoted in the prose) |
| `scripts/make_figures.py` | writes `figs/fig1,3–8_*.{pdf,png}`; `scripts/figure_provenance.json` |
| `figs/fig_architecture3d.*` | copied from `IR3C_journal/IR3C_journal_V2/` (TikZ source included) |

## Conventions (apply everywhere)

* IR3C = **budget-matched** model in every comparison: BraTS `IR3_matched` (step 12,510);
  OAIZIB / IXI fine-tuned from it (`runs/*_matchedinit`). Full budget only in Table 23.
  *Exception*: downstream segmentation (Table 6) — archived recons are full-budget; marked †, re-run pending.
* Region = foreground voxels of missing slices. Aggregation = cells averaged within subject, then mean ± sd across subjects.
* Slice rejection = `runs/IR3C_SliceReject_all/` (matched), **not** the older full-init `IR3C_SliceReject/` used by draft 1.
* CAS `table` takes `[pos=...]`, not `[t]`/`[H]` (plain options are silently ignored).

## Corrections relative to draft 1

1. IXI rejection numbers came from the older full-init run; now matched-budget (`IR3C_SliceReject_all`).
2. BraTS rejection arm is complete and is now reported (draft 1 said rejection was IXI-only).
3. IXI/OAIZIB IR3C rows now use matched-init fine-tunes (IXI −0.09 dB vs draft 1, OAIZIB ±0.00).
4. Dataset table: explicit spacing + reconstruction axis (IXI recon axis = axis 0, 83 slices, 2.18 mm; OAIZIB = sagittal, 0.70 mm).
5. Downstream recons are full-budget IR3C — now disclosed.
6. SAINT is a single multi-factor model; DCSRN and I3Net are per-factor.
7. Uniform results reported per R (SAINT wins R=2 on all datasets; IR3C wins R=8 on all) rather than only pooled.
8. Qualitative figure rebuilt: sagittal reformats (through-plane visible), matched IR3C, missing-slice-only error, subject chosen by a fixed rule (median gain).

## Pending experiments (shaded `tbd` cells + orange boxes in the PDF)

Set `\showpendingfalse` in `main.tex` to hide the orange boxes.

| # | Experiment | Fills | Cost / status |
|---|---|---|---|
| P1 | OAIZIB slice rejection (`IR3C_SliceReject_all/run_all.sh`) | Table 5 OAIZIB rows, Fig. 5 row | ~2.5 GPU-days; smoke passed |
| P2 | Downstream seg with matched IR3C | Table 6 shaded row | inference only, ~2 h |
| P3 | Formulation ablation (HPC package: A0–A3, C1–C3) + new binary-mask-only arm + seeds 43/44 for A0, A3, C2 | Table 9; "3D U-Net direct" rows of Table 2 | HPC |
| P4 | Leave-one-family-out training (6 models) | Table 10 "Held out" | 6 trainings |
| P5 | 5 subject-specific mask realisations per family, R=4 | Table 11 | inference only |
| P6 | Per-slice error vs gap length / distance | Fig. 8 | re-scoring only (store per-slice SE) |
| P7 | Labelled downstream: hold out ~150 labelled BraTS training subjects, retrain IR3C + SAINT + ArSSR | Table 12 | most costly |
| P8 | SA-INR baseline, all datasets | Table 2, Table 3 SA-INR rows | train + eval |

## Author-side fields (orange `[...]` in PDF)

Authors, affiliations, ORCID, email, CRediT, competing interests, funding, code URL, and the
**Elsevier generative-AI declaration** (mandatory; ChatGPT/Claude were used in drafting).

## Check before submission

* Verify bibliography details for `remedios2026eclare` and `wohlrapp2026downstream` (carried over from draft 1, not re-verified).
* Abstract is ~330 words; check the current MedIA guide-for-authors limit.
* Fig. 2 (TikZ) says "positional profiles"; the text says "geometry profiles".
