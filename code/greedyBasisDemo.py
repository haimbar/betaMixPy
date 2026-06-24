"""
Demonstration of greedyBasis() on the DrySeeds metabolomics dataset.

Run from the code/ directory:
    PYTHONPATH=.. python3 greedyBasisDemo.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import betaMixPy as bx

M   = pd.read_csv("../data/DrySeeds.csv", index_col=0)
res = bx.betaMix(M, ppr=0.01, ind=True, msg=False)
adjM = bx.getAdjMat(res)

print("=== strategy='min_degree' (classical MIS, non-redundant basis) ===")
bf_min = bx.greedyBasis(adjM, strategy="min_degree", labels=M.columns)
bx.summarizeBasis(bf_min)

print()
print("=== strategy='max_degree' (hub-first MIS) ===")
bf_max = bx.greedyBasis(adjM, strategy="max_degree", labels=M.columns)
bx.summarizeBasis(bf_max)

print()
print("=== Top-5 non-basis metabolites by number of communities (min_degree) ===")
non_basis = bf_min[~bf_min["isBasis"]].copy()
top = non_basis.nlargest(5, "nCommunities")[["degree", "nCommunities", "communities"]]
for name, row in top.iterrows():
    comm_names = [M.columns[c] for c in row["communities"]]
    print(f"  {name:<22s}  deg={int(row['degree']):2d}  "
          f"nCommunities={int(row['nCommunities'])}  "
          f"communities: {', '.join(comm_names)}")
