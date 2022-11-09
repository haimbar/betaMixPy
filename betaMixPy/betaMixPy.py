"""
A Python package to find strong correlations among P variables, each with N observations.
"""
import math
import statistics as stat
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
from scipy.stats import beta as betadist
from scipy.special import psi ,polygamma
from scipy.optimize import root, Bounds, least_squares
from scipy.sparse import diags



def MLEfun(params, *args):
    """The log-likelihood function of the beta-mixture distrubtion, used to obtain
    the maximum likelihood estimates of the non-null component (a, b).

    params -- the beta distribution parameters to be estimated.
    args -- z: the sine-squared of the angle between pairs of vectors; 
            m0: the posterior null probability of each pair.
    """
    a, b = params
    z, m0 = args
    sm = sum(1-m0)
    if(sm < 1e-10):
        return(1, 1)
    return np.array(sm*(psi(a)-psi(a+b))-sum((1-m0)*np.log(z)), sm*(psi(b)-psi(a+b))-sum((1-m0)*np.log(1-z)))


def etafun(eta, z, m0):
    """The log-likelihood function of the null component of beta-mixture distrubtion
    when the parameter eta is not assumed to be known and equal to (N-1)/2.

    eta -- the unknown parameter of the beta distribution (the other one is equal to 0.5).
    z -- the sine-squared of the angle between pairs of vectors.
    m0 -- the posterior null probability of each pair.
    """
    if(sum(m0) < 1e-10):
        return 1
    return (psi(eta) - psi(eta+0.5) - sum(m0*np.log(z))/sum(m0))


def shortSummary(P, N, bmax, ahat, bhat, etahat, ppthr, edges, p0):
    print("No. Nodes:", P, "\nPossible no. edges:", int(P*(P-1)/2), "\nNo. edges detected =", edges,
    "\nDetection threshold:", round(ppthr, 2), "\nPr(null):", round(p0, 4),
     "\nNonnull support: [0,", round(bmax, 2),")\nNonnull parameters (a, b, eta): (", round(ahat, 2), ", ", round(bhat, 2), ", ", round(etahat, 2),")\n")


def plotFittedBetaMix(z, p0, ahat, bhat, etahat, ppthr, bmax, file=""):
    """Plot the histogram of z (the sine-square of the angles between pairs of
    predictors and the fitted mixture distribution (log-scale).

    z -- the sine-squared of the angle between pairs of vectors.
    p0 -- the probability of the null component.
    ahat, bhat -- the parameters of the nonnull component.
    etahat -- the first parameter of the null component (the second is 0.5).
    ppthr -- the threshold for dclaring a strong correlations.
    bmax -- the upper bound of the support of the non-null component.
    file -- the file name to save to plot ("" means show on the screen)
    """
    x = np.arange(0.001, 1-0.001, 0.0001)
    y0 = betadist.pdf(x, etahat, 0.5)*p0
    y1 = np.zeros(len(y0))
    y1[np.where(x < bmax)] = betadist.pdf(x[np.where(x < bmax)], ahat, bhat, scale=bmax)*(1-p0)
    plt.plot(x, y0, color='cyan', lw=4)
    plt.plot(x, y1, color='magenta', lw=4)
    plt.plot(x, y0+y1, color='navy', lw=2, linestyle='dashed')
    plt.ylim(1e-6, max(y0+y1)*2)
    plt.hist(z, bins=150, range=(0, 1), log=True, density=True, ec='white', color='silver', bottom=1e-6)
    plt.axvline(ppthr, color='r', lw=3, linestyle='dotted')
    if file == "":
        plt.show()
    else:
        plt.savefig(file)


def plotBitmapCC(adjM, thr, clusterInfo=pd.DataFrame(), orderByCluster=False, showMinDegree=0, file=""):
    """Show the connectivity in the network as a grid (bitmap).

    adjM -- the adjacency matrix.
    thr -- the threshold below which an angle between two vectors is smaller than expected by chance.
    clusterInfo -- output from the graphComponents function.
    orderByCluster -- use the original order of nodes (default) or re-order by clusters.
    showMinDegree -- the minimum number of neighbors a node should have in order to be included in the plot.
    file -- the output file ("" means show on the screen)
    """
    adjM = abs(adjM)
    if clusterInfo.size > 0:
        orderByCluster = True
    if orderByCluster == True:
        if clusterInfo.size > 0:
            clusterInfo = graphComponents(adjM, thr)
        clusterInfo.sort_values(by=['clustNo', 'distCenter'], inplace=True, ascending=[False, False])
        nodeOrder = clusterInfo['rnum']
    else:
        nodeOrder = range(adjM.shape[0])
    adjM = adjM[nodeOrder,:][:,nodeOrder]
    showNodes = np.where(np.sum(adjM, axis=0) >= showMinDegree)[0]
    adjM = adjM[showNodes,:][:,showNodes]
    plt.imshow(adjM)
    if file == "":
        plt.show()
    else:
        plt.savefig(file)


def plotDegCC(degs, ccs, highlightNodes=[], file=""):
    """Plot the degree of each node vs. the correlation coeficient times the degree.

    degs -- the node degrees.
    ccs -- the node correlation coeficients.
    highlightNodes -- numeric values of node numbers which should be highlighted.
    file -- the output file ("" means show on the screen)
    """
    plt.scatter(degs, ccs*degs, c="thistle", s=10)
    for item in highlightNodes:
        plt.scatter(degs[item], ccs[item]*degs[item], c="green", s=30)
    plt.xlabel("degree")
    plt.ylabel("CC*degree")
    plt.grid() # color='r', linestyle='-', linewidth=2
    x = np.linspace(0,max(degs),50)
    plt.plot(x, x)
    if file == "":
        plt.show()
    else:
        plt.savefig(file)


def getAdjMat(sineAngleMat, thr):
    """Construct the adjacency matrix for the network. Could contain 0 (no correlation),
    1 or -1 (strong positive or negative correlations).

    sineAngleMat -- the matrix of pairwise sine of angles.
    thr -- the threshold below which the sine of an angle between two vectors is smaller than expected by chance.
    """
    adjM = np.sign(sineAngleMat)
    adjM[np.where(sineAngleMat**2 >= thr)] = 0
    np.fill_diagonal(adjM, 0)
    return(adjM)


def clusteringCoef(A):
    """Calculate the clustering coefficient of each node.

    A -- an adjacency matrix.
    """
    A = abs(A)
    rsum = np.sum(A, axis=1)
    P = A.shape[0]
    cc = [0]*P
    for i in range(P):
        if rsum[i] > 1:
            nbrs = np.where(A[i,] == 1)[0]
            Atmp = A[nbrs,:][:, nbrs] 
            cc[i] = 0.5*np.sum(Atmp)/(rsum[i]*(rsum[i]-1)/2)
    return(np.array(cc))


def graphComponents(A, minCtr=5, wgt=1, labels=list([])):
    """Find graph components from an adjacency Matrix. For each node, find 
    the degree and clustering coefficient (CC). Then, calculate a centrality
    measure (wgt\*CC+1)\*deg. For wgt=0, it's just the degree. Note that
    setting wgt=1 means that we assign a higher value to nodes that not only
    have many neighbors, but the neighbors are highly interconnected.
    For example, suppose we have two components with k nodes, one has a star shape,
    and the other is a complete graph. With wgt=0 both graphs will get the
    same value, but with wgt=1 the complete graph will be picked by the
    algorithm first. Setting wgt to a negative value gives CC\*deg as
    the centrality measure.
    The function returns a data frame with node labels, CC, degree, cluster
    number, whether a node is the center of a cluster, tthe number of edges from
    the node to nodes in the same cluster, the number of edges from the node
    to nodes NOT in the same cluster, and a standardized Manhattan distance
    to the central node.
    
    A -- an adjacency matrix.
    minCtr -- the minimum number of nodes in order to define a component.
    wgt -- weight parameter used to define the centrality measure (see above).
    """
    A = abs(A)
    Vn = A.shape[1]
    deg = np.sum(A, axis=0)
    CC = np.array(clusteringCoef(A))
    ctrs = (wgt*CC+1)*deg
    if wgt < 0:
        ctrs = CC*deg
    clNum = 1
    if(len(labels) < Vn):
        labels = list(range(Vn))
    iscenter = np.array([0]*Vn)
    clustNo = np.array([0]*Vn)
    intEdges = np.array([0]*Vn)
    extEdges = np.array([0]*Vn)
    distCenter = np.array([0]*Vn)
    clustered = np.where(deg < 1)[0]
    while len(clustered) < Vn:
        notInCluster = np.setdiff1d(range(Vn), clustered) 
        if (max(ctrs[notInCluster]) < minCtr):
            df = pd.DataFrame({'rnum': range(Vn),
                'deg': deg, 'CC': CC, 'ctrs': ctrs, 'clustNo': clustNo, 'iscenter': iscenter,
                'intEdges': intEdges, 'extEdges': extEdges, 'distCenter': distCenter
            }, index=labels)
            return df
        ctrnode = notInCluster[np.where(ctrs[notInCluster] == max(ctrs[notInCluster]))][0]
        unarr = np.union1d(ctrnode, np.where(A[ctrnode,] != 0)[0])
        nbrs = np.setdiff1d(unarr, clustered)
        if len(nbrs) > minCtr:
            iscenter[ctrnode] = 1
            clustNo[np.union1d(ctrnode, nbrs).astype(int)] = clNum
            intEdges[nbrs] = np.sum(A[nbrs,:][:, nbrs], 1)
            if len(nbrs) < Vn:
                extEdges[nbrs] = np.sum(A[nbrs,:][:, np.setdiff1d(range(Vn), nbrs)], 1)
            else:
                extEdges[nbrs] = 0
            for i in range(len(nbrs)):
                distCenter[nbrs[i]] = stat.mean(np.minimum(A[ctrnode,:], A[nbrs[i],:]))
            clNum = clNum+1
        else:
            nbrs = []
        clustered = np.union1d(clustered, np.union1d(ctrnode, nbrs))
    df = pd.DataFrame({'rnum': range(Vn),
        'deg': deg, 'CC': CC, 'ctrs': ctrs, 'clustNo': clustNo, 'iscenter': iscenter,
        'intEdges': intEdges, 'extEdges': extEdges, 'distCenter': distCenter
    }, index=labels)
    return df


def sphericalCaps(A, graphCom):
    """Returns a data frame with all the points within a spherical cap around a central node.
    A point is close to the center if the sine-square of the angle between it and
    the cap's center is less than the threshold found by betaMix.

    A -- the adjacency matrix.
    graphCom -- the output (data frame) from the graphComponents function.
    """
    ctrs = np.where(graphCom['iscenter'] != 0)[0]
    if len(ctrs) == 0:
        return None
    df = pd.DataFrame(columns=['clustNo','Node'])
    for i in range(len(ctrs)):
        inCap = np.where(A[ctrs[i],:] != 0)[0]
        cnum = int(graphCom.iloc[ctrs[i]]['clustNo'])
        df2 = pd.DataFrame({'clustNo': [cnum]*len(inCap), 'Node': inCap})
        df = pd.concat([df, df2], sort=False)
    df.index = range(df.shape[0])
    return df


def plotCluster(adjM, clNo, grComp, labels=True, nodecol="#00FF00F0", edgecols="#AEAEAE05", file=""):
    """Plot the network representation of a cluster.

    adjM -- the adjacency matrix.
    clNo -- the selected cluster number.
    grComp -- the output (data frame) from the graphComponents function.
    labels -- whether to show the node names or not.
    nodecol -- the colors for the nodes.
    edgecols -- the colors for the edges.
    file -- the output file ("" means show on the screen)
    """
    if (max(grComp['clustNo']) < clNo) or clNo < 1:
        print("Invalid clustar number. Must be between 1 and " + str(max(grComp['clustNo'])))
        return None
    nodes = np.where(grComp['clustNo'] == clNo)[0]
    if len(nodes) < 2:
        print(str(len(nodes)) + ' nodes.')
        return None
    df = pd.DataFrame(adjM[nodes,:][:,nodes])
    df.index = grComp.index[nodes]
    df.columns = grComp.index[nodes]
    G = nx.from_pandas_adjacency(abs(df))
    nx.draw(G, with_labels=labels, linewidths=0.5, edge_color=edgecols, node_color=nodecol, style="dashed", alpha=0.5)
    if file == "":
        plt.show()
    else:
        plt.savefig(file)


def betaMix(M, tol=1e-5, calcAcc=1e-9, ppr=0.01, maxalpha=1e-4, mxcnt=200, ahat=8, bhat=2,
            subsamplesize=50000, seed=912469, ind=True, msg=True):
    """Fit a two-component beta mixture model to the matrix of all pairwise correlations.
    
    M -- A matrix with P rows (variables) and N columns (observations), which is what numpy's corrcoef expects as input.
    tol -- The convergence threshold for the EM algorithm (default= the maximum of user's input and 1/(P(P-1)/2)).
    calcAcc -- The calculation accuracy threshold (to avoid values greater than 1 when calling asin) Default=1e-9.
    ppr -- The null posterior probability threshold (default=0.01). 
    maxalpha -- The probability of Type I error (default=1e-4). For a large P, 0.01/(P*(P-1)/2) is used.
    mxcnt -- The maximum number of EM iterations (default=200).
    ahat -- The initial value for the first parameter of the nonnull beta distribution (default=8).
    bhat -- The initial value for the second parameter of the nonnull beta distribution (default=2).
    subsamplesize -- If greater than 20000, take a random sample of size subsamplesize to fit the model. Otherwise, use all the data  (default=50000).
    seed -- The random seed to use if selecting a subset with the subsamplesize parameter (default=912469).
    ind -- Whether the N samples should be assumed to be independent (default=True).
    msg -- Whether to print intermediate output messages (default=True).

    Returns the following:
    sineAngleMat -- A PxP matrix with the sine of the angles between pairs of vectors.
    z_j -- The statistics z_j=sin^2(angles).
    m0 -- The posterior null probabilities.
    p0 -- The estimated probability of the null component.
    ahat -- The estimated first parameter of the nonnull beta component.
    bhat -- The estimated second parameter of the nonnull beta component.
    etahat -- If the samples are not assumed to be independent, this corresponds to the effective sample size, ESS=2*etahat+1.
    ppthr -- The estimated posterior probability threshold, under which all the z_j correspond to nonnull edges.
    P -- The number of vectors.
    edges -- The number of edges detected in the graph.
    bmax -- The estimated right-hand side of the support of the non-null component.
    """
    P, N = M.shape
    maxalpha = min(maxalpha, 0.01/(P*(P-1)/2))
    etahat = (N-1)/2
    corM = np.corrcoef(M)
    nas = np.where(np.isnan(corM))
    if len(nas[0]) > 0:
        print("Got ", len(nas[0]), " NANs in the correlation matrix. Replacing with zeros.")
        corM[nas] = 0
    sineAngleMat = np.sin(np.arccos(corM))
    z_j = (sineAngleMat[np.triu_indices_from(sineAngleMat, k=1)])**2
    tooSmall = np.where(z_j < calcAcc)
    if len(tooSmall) > 0:
        z_j[tooSmall] = calcAcc
    tooLarge = np.where(z_j > 1-calcAcc)
    if len(tooLarge) > 0:
        z_j[tooLarge] = 1-calcAcc
    z_jall = z_j
    if len(z_j) > subsamplesize and subsamplesize >= 20000:
        rs = RandomState(MT19937(SeedSequence(seed)))
        z_j = np.random.choice(z_jall, subsamplesize)
    z_j.sort()
    p0 = min(1, len(np.where(z_j > betadist.ppf(0.1, etahat, 0.5)))/(0.9*len(z_j)))
    tol = max(tol, 1/len(z_j))
    if msg:
        print("Fitting the model...\n")
    p0f0 = p0*betadist.pdf(z_j, etahat, 0.5)
    p1f1 = (1-p0)*betadist.pdf(z_j, ahat, bhat)
    m0 = p0f0/(p0f0+p1f1)
    p0new = stat.mean(m0)
    bmax = 1
    inNonNullSupport = np.where(z_j <= bmax)
    cnt = 0
    while np.abs(p0-p0new) > tol and cnt < mxcnt :
        cnt = cnt+1
        p0 = p0new
        if ind == False:
            etahat = (root(etafun, x0=[(N-1)/2], args=(z_j, m0)).x)[0]
        bmax = z_j[min(np.where(z_j > betadist.ppf(100/len(z_j), etahat, 0.5))[0])]
        inNonNullSupport = np.where(z_j < bmax)
        outNonNullSupport = np.where(z_j >= bmax)
        ahat, bhat = least_squares(MLEfun, (ahat, bhat), args=(z_j[inNonNullSupport]/bmax, m0[inNonNullSupport]), bounds=([0.1, 0.1], [1000, 1000])).x
        p0f0 = p0*betadist.pdf(z_j, etahat, 0.5)
        p1f1[outNonNullSupport] = 0
        p1f1[inNonNullSupport] = (1-p0)*betadist.pdf(z_j[inNonNullSupport], ahat, bhat, scale=bmax)
        m0 = p0f0/(p0f0+p1f1)
        p0new = stat.mean(m0)
    if len(z_jall) > subsamplesize:
        z_j = z_jall
    z_j.sort()
    inNonNullSupport = np.where(z_j < bmax)[0]
    p0f0 = p0*betadist.pdf(z_j, etahat, 0.5)
    p1f1 = np.zeros(len(p0f0))
    p1f1[inNonNullSupport] = (1-p0)*betadist.pdf(z_j[inNonNullSupport], ahat, bhat, scale=bmax)
    m0 = p0f0/(p0f0+p1f1)
    p0 = stat.mean(m0)
    ppthr = betadist.ppf(maxalpha, etahat, 0.5)
    critPP = np.where(m0 > ppr)[0]
    if len(critPP) > 0:
        ppthr = max(min(z_j[critPP]), ppthr)
    nonnull = np.where(z_j < ppthr)[0]
    edges = len(nonnull)
    return sineAngleMat, z_j, m0, p0, ahat, bhat, etahat, ppthr, P, edges, bmax

