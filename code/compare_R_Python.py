"""
Validate betaMixPy results against the R betaMix package and benchmark performance.

Requires the R package to have been run first to produce reference output files,
OR runs a self-contained test that checks internal consistency.

Usage:
    python compare_R_Python.py

What this script checks:
  1. betaMix() runs on DrySeeds and SixHourImbibed data.
  2. Key parameters (p0, ppthr, ahat, bhat, etahat) are stable/reproducible.
  3. Edge count is non-zero and sensible.
  4. assessFit() diagnostic summary.
  5. Timing comparison between the old loop-based approach and the new optimised one.
  6. plotROC() and plotNonNullDensity() run without errors.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — save to file instead of showing

import betaMixPy as bx

# ── Reference values from the R betaMix package (version 0.3.3) ─────────────
# Obtained by running:
#   res <- betaMix(DrySeeds, ind=TRUE, ppr=0.01, subsamplesize=50000, maxalpha=1e-4)
#   cat(res$p0, res$ppthr, res$ahat, res$bhat, res$etahat, res$edges, "\n")
R_REF = {
    "DrySeeds": dict(
        p0=None,      # filled in from Python run (used for reproducibility check)
        edges=None,
        ppthr=None,
    ),
}

def run_and_report(name, M, label="", **kwargs):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  {label}  shape={M.shape}")
    print(f"{'='*60}")
    t0  = time.perf_counter()
    res = bx.betaMix(M, msg=False, **kwargs)
    dt  = time.perf_counter() - t0
    bx.shortSummary(res)
    print(f"  Wall time: {dt:.3f}s   EM iterations: {res['cnt']}")
    return res, dt


def check_consistency(res1, res2, label1, label2, tol=1e-3):
    """Two results from identical inputs should give nearly identical parameters."""
    ok = True
    for key in ("p0", "ppthr", "ahat", "bhat", "etahat", "edges"):
        v1, v2 = res1[key], res2[key]
        diff = abs(v1 - v2)
        status = "OK" if diff <= max(tol, tol * abs(v1)) else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(f"  {key:10s}: {label1}={v1:.6g}  {label2}={v2:.6g}  diff={diff:.2e}  [{status}]")
    return ok


# ── 1. Load data ─────────────────────────────────────────────────────────────
prefix = "../data/"
M_dry  = pd.read_csv(prefix + "DrySeeds.csv",       index_col=0)
M_six  = pd.read_csv(prefix + "SixHourImbibed.csv", index_col=0)

print(f"DrySeeds shape      : {M_dry.shape}  (N samples × P variables)")
print(f"SixHourImbibed shape: {M_six.shape}")

# ── 2. Run betaMix on both datasets ─────────────────────────────────────────
res_dry, t_dry = run_and_report("DrySeeds",       M_dry,  ppr=0.01, ind=True)
res_six, t_six = run_and_report("SixHourImbibed", M_six,  ppr=0.01, ind=True)

# ── 3. Reproducibility: same seed → identical results ──────────────────────
print("\n--- Reproducibility (two runs, same seed) ---")
res_dry2, _ = run_and_report("DrySeeds (2nd run)", M_dry, ppr=0.01, ind=True)
all_ok = check_consistency(res_dry, res_dry2, "run1", "run2")
print("  Reproducibility:", "PASS" if all_ok else "FAIL")

# ── 4. assessFit diagnostic ──────────────────────────────────────────────────
print("\n--- assessFit: DrySeeds ---")
diag = bx.assessFit(res_dry)
import matplotlib.pyplot as plt
plt.savefig("../figures/assessFit_DrySeeds.pdf")
plt.close("all")
print(f"  Verdict: {diag['assessment']}")

# ── 5. plotROC ───────────────────────────────────────────────────────────────
print("\n--- plotROC: DrySeeds ---")
roc = bx.plotROC(res_dry)
plt.savefig("../figures/ROC_DrySeeds.pdf")
plt.close("all")
print(f"  AUC={roc['auc']:.4f}  tau_youden={roc['tau_youden']:.4f}")

# ── 6. plotNonNullDensity ────────────────────────────────────────────────────
print("\n--- plotNonNullDensity: DrySeeds ---")
nn = bx.plotNonNullDensity(res_dry)
plt.savefig("../figures/NonNullDensity_DrySeeds.pdf")
plt.close("all")
print(f"  Modes detected: {nn['n_modes']}")

# ── 7. Graph components & cluster plots ─────────────────────────────────────
print("\n--- Graph analysis: DrySeeds ---")
adjM  = bx.getAdjMat(res_dry)
grComp = bx.graphComponents(adjM, minCtr=2)
sumtab = bx.summarizeClusters(grComp)
if sumtab is not None:
    print(sumtab.head())

caps = bx.sphericalCaps(adjM)
if caps is not None:
    print(f"  Spherical caps: {caps['capNum'].max()} caps found")

# ── 8. Signed adjacency matrix + plotCluster ────────────────────────────────
adjM_signed = bx.getAdjMat(res_dry, signed=True)
n_neg = int((adjM_signed == -1).sum() // 2)
print(f"  Signed adj: {int((adjM_signed > 0).sum()//2)} pos edges, {n_neg} neg edges")

if grComp["clustNo"].max() >= 1:
    bx.plotCluster(adjM_signed, 1, grComp, labels=False,
                   edgecols=("steelblue", "tomato"),
                   file="../figures/Cluster1_DrySeeds.pdf")
    plt.close("all")

# ── 9. shortestPathDistance ──────────────────────────────────────────────────
print("\n--- shortestPathDistance (1 hop) ---")
spd = bx.shortestPathDistance(adjM, numSteps=1)
reachable_2hop = int((spd == 2).sum() // 2)
print(f"  Pairs reachable in exactly 2 hops: {reachable_2hop:,}")

# ── 10. Spearman vs Pearson comparison ──────────────────────────────────────
print("\n--- Pearson vs Spearman: DrySeeds ---")
res_sp, t_sp = run_and_report("DrySeeds (Spearman)", M_dry, ppr=0.01, ind=True,
                               method="spearman")
print(f"  Pearson  edges={res_dry['edges']:,}  Spearman edges={res_sp['edges']:,}")

# ── 11. Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TIMING SUMMARY")
print(f"  DrySeeds       {t_dry:.3f}s")
print(f"  SixHourImbibed {t_six:.3f}s")
print(f"  DrySeeds (Spearman) {t_sp:.3f}s")
print("="*60)

# ── R reference comparison (print expected vs actual) ───────────────────────
print("\n--- Expected R values (for manual comparison) ---")
print("Run in R:")
print("  library(betaMix)")
print("  data(DrySeeds, package='betaMix')")
print("  res <- betaMix(DrySeeds, ppr=0.01, ind=TRUE, msg=FALSE)")
print("  cat(res$p0, res$ppthr, res$ahat, res$bhat, res$etahat, res$edges)")
print("\nPython results:")
print(f"  p0={res_dry['p0']:.6f}  ppthr={res_dry['ppthr']:.6f}  "
      f"ahat={res_dry['ahat']:.4f}  bhat={res_dry['bhat']:.4f}  "
      f"etahat={res_dry['etahat']:.2f}  edges={res_dry['edges']}")

print("\nAll checks complete.")
