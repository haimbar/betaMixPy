"""
Example: betaMixPy on metabolomics data (Dry Seeds and Six-Hour Imbibed).

M convention: N rows = samples, P columns = variables (matches R betaMix).
The CSV files have samples in rows, so NO transpose is needed.
"""
import pandas as pd
import betaMixPy as bx
import matplotlib.pyplot as plt

# --- Dry Seeds ---
M0 = pd.read_csv("../data/DrySeeds.csv", index_col=0)
print("DrySeeds shape (N, P):", M0.shape)

res0 = bx.betaMix(M0, subsamplesize=50000, ind=True, ppr=0.01)
bx.shortSummary(res0)

adjM0 = bx.getAdjMat(res0)
df0   = bx.graphComponents(adjM0, minCtr=2)

bx.plotBitmapCC(adjM0, file="../figures/DrySeeds.pdf")
plt.close("all")

# Find cluster containing GABA (index 44 in the variable list)
gaba_idx = list(M0.columns).index("GABA") if "GABA" in M0.columns else 44
clst0 = int(df0.iloc[gaba_idx]["clustNo"])
bx.plotCluster(adjM0, clst0, df0, labels=True, nodecol="#DDFF00",
               file="../figures/Gaba0.pdf")
plt.close("all")

# --- Six-Hour Imbibed ---
M1 = pd.read_csv("../data/SixHourImbibed.csv", index_col=0)
print("SixHourImbibed shape (N, P):", M1.shape)

res1 = bx.betaMix(M1, subsamplesize=50000, ind=True, ppr=0.01)
bx.shortSummary(res1)

adjM1 = bx.getAdjMat(res1)
df1   = bx.graphComponents(adjM1, minCtr=2)

bx.plotBitmapCC(adjM1, file="../figures/SixHourImbibed.pdf")
plt.close("all")

gaba_idx = list(M1.columns).index("GABA") if "GABA" in M1.columns else 44
clst1 = int(df1.iloc[gaba_idx]["clustNo"])
bx.plotCluster(adjM1, clst1, df1, labels=True, nodecol="#00FFDD",
               file="../figures/Gaba1.pdf")
plt.close("all")
