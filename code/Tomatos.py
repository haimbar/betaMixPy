import pandas as pd
import betaMixPy as bx
import matplotlib.pyplot as plt
M0 = pd.read_table("../data/DrySeeds.csv", sep=",", index_col=0)
M0 = M0.transpose()
N0 = M0.shape[0]
sineAngleMat, z_j, m0, p0, ahat, bhat, etahat, ppthr, P, edges, bmax = bx.betaMix(M0,
        subsamplesize=50000, ind=True, ppr=0.01)
adjM0 = bx.getAdjMat(sineAngleMat, ppthr)
df0 = bx.graphComponents(adjM0, minCtr=2, labels=M0.index)
bx.plotBitmapCC(adjM0, ppthr, file="../figures/DrySeeds.pdf")
plt.close('all')
clst0 = int(df0.iloc[[44]]['clustNo'])
bx.plotCluster(adjM0, clst0, df0, labels=True, nodecol="#DDFF00F0", file="../figures/Gaba0.pdf")
plt.close('all')

#np.where(df1.index == "GABA") # index=44

M1 = pd.read_table("../data/SixHourImbibed.csv", sep=",", index_col=0)
M1 = M1.transpose()
N1 = M1.shape[0]
sineAngleMat, z_j, m0, p0, ahat, bhat, etahat, ppthr, P, edges, bmax = bx.betaMix(M1,
        subsamplesize=50000, ind=True, ppr=0.01)
adjM1 = bx.getAdjMat(sineAngleMat, ppthr)
df1 = bx.graphComponents(adjM1, minCtr=2, labels=M1.index)
bx.plotBitmapCC(adjM1, ppthr, file="../figures/SixHourImbibed.pdf")
plt.close('all')
clst1 = int(df1.iloc[[44]]['clustNo'])
bx.plotCluster(adjM1, clst1, df1, labels=True, nodecol="#00FFDDF0", file="../figures/Gaba1.pdf")
plt.close('all')
