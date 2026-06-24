"""
Example: betaMixPy on simulated hub-structured data.

M convention: N rows = samples, P columns = variables (matches R betaMix).
The CSV files have samples as rows, so NO transpose is needed.
"""
import pandas as pd
import betaMixPy as bx
import matplotlib.pyplot as plt

prefix = "../data/"
files = [
    "cls_0200_1000_03_0050",
    "hub_0500_0500_03_0025",
    "hub_0200_1000_09_0020",
    "rbk_0500_0500_03",
]

i = 3
# Samples in rows, variables in columns (no transpose needed)
M = pd.read_csv(prefix + files[i] + ".csv", index_col=0)
print("Data shape (N, P):", M.shape)

res = bx.betaMix(M)
bx.shortSummary(res)

bx.plotFittedBetaMix(res)  # , file="../figures/fitted.pdf"
plt.close("all")

adjM = bx.getAdjMat(res)
df1  = bx.graphComponents(adjM, minCtr=2)

bx.plotBitmapCC(adjM, file="../figures/CorMatEst.pdf")
plt.close("all")

bx.plotCluster(adjM, 1, df1, labels=True, nodecol="#DDFF00")
# , file="../figures/Cluster.pdf"
plt.close("all")
