import pandas as pd
import betaMixPy as bx
import matplotlib.pyplot as plt
prefix = "../data/"
files = ["cls_0200_1000_03_0050", "hub_0500_0500_03_0025", "hub_0200_1000_09_0020",
            "rbk_0500_0500_03"]
i = 3
M = pd.read_table(prefix+files[i]+".csv", sep=",", index_col=0)
M = M.transpose()
N = M.shape[0]
sineAngleMat, z_j, m0, p0, ahat, bhat, etahat, ppthr, P, edges, bmax, N = bx.betaMix(M)
bx.shortSummary(P, N, bmax, ahat, bhat, etahat, ppthr, edges, p0)
bx.plotFittedBetaMix(z_j, p0, ahat, bhat, etahat, ppthr, bmax)#, file="../figures/fitted.pdf")
plt.close('all')
adjM = bx.getAdjMat(sineAngleMat, ppthr)
df1 = bx.graphComponents(adjM, minCtr=2, labels=M.index)
bx.plotBitmapCC(adjM, ppthr, file="../figures/CorMatEst.pdf")
plt.close('all')
bx.plotCluster(adjM, 1, df1, labels=True, nodecol="#DDFF00F0")#, file="../figures/Cluster.pdf")
plt.close('all')