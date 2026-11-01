# betaMixPy

Python implementation of the **betaMix** method for detecting edges (significant pairwise associations) in gene co-expression and other multivariate networks.

The method is built on the result of Frankl & Maehara (1990): two random vectors in a high-dimensional space are nearly orthogonal with high probability.  Pairwise correlations equal the cosine of the angle between column vectors, so the statistic $z_j = 1 - r_{ij}^2 \in (0,1]$ follows a $\mathrm{Beta}((N-1)/2,\,1/2)$ null distribution.  An EM algorithm fits a two-component Beta mixture to all $P(P-1)/2$ pairs simultaneously, and a threshold is derived that simultaneously controls a frequentist (Type I error) and a Bayesian (posterior probability) false-positive rate.

## Installation

```
pip install betaMixPy
```

## Quick start

```python
import pandas as pd
import betaMixPy as bx

# M is N×P: samples in rows, variables (genes/metabolites) in columns
M   = pd.read_csv("my_data.csv", index_col=0)
res = bx.betaMix(M, ppr=0.01, ind=True)    # fit the mixture model

bx.shortSummary(res)       # parameter estimates + edge count
bx.assessFit(res)          # 2×2 diagnostic panel

adjM  = bx.getAdjMat(res)                     # P×P adjacency matrix
clust = bx.graphComponents(adjM, minCtr=2)    # disjoint clusters
bx.plotBitmapCC(adjM, clust)                  # bitmap of adjacency
bx.plotCluster(adjM, 1, clust, labels=True)   # network plot of cluster 1
```

## Input convention

`betaMix()` expects **M as N×P** — N samples in rows, P variables in columns — matching numpy's `corrcoef(M, rowvar=False)` and the R `betaMix` package.

> **Upgrading from ≤ 0.1.4?** Remove any `.T` transpose and update code that unpacked the previous tuple return value. See [NEWS.md](NEWS.md) for the full list of breaking changes.

## Key functions

### Fitting and diagnostics

| Function | Description |
|---|---|
| `betaMix(M, ...)` | Fit the two-component Beta mixture; returns a `dict` |
| `shortSummary(res)` | Print parameter estimates and edge count |
| `assessFit(res)` | 2×2 panel: mixture histogram, null Q-Q, non-null Q-Q, ROC |
| `plotFittedBetaMix(res)` | Fitted mixture density on log scale |
| `plotROC(res)` | Model-based ROC; returns AUC and Youden-index threshold |
| `plotNonNullDensity(res)` | Kernel density of non-null z_j values |

### Network analysis

| Function | Description |
|---|---|
| `getAdjMat(res, ...)` | Build P×P adjacency matrix; `signed=True` for −1/0/1 |
| `graphComponents(A, ...)` | Disjoint clusters via centrality-based assignment |
| `sphericalCaps(A, ...)` | Hub-and-spoke clustering (greedy, degree-ordered) |
| `greedyBasis(A, strategy, ...)` | **Overlapping communities** via maximal independent set |
| `summarizeClusters(ci)` | Five-number summary per cluster |
| `summarizeBasis(bi)` | Summary for a `greedyBasis()` result |
| `collapsedGraph(res, ci)` | Adjacency matrix reduced to cluster centres |
| `shortestPathDistance(A, numSteps)` | Multi-hop distances in the network |
| `clusteringCoef(A)` | Vectorised clustering coefficients |

### Visualisation

| Function | Description |
|---|---|
| `plotBitmapCC(A, ...)` | Bitmap heatmap of adjacency, coloured by cluster |
| `plotCluster(A, n, ci, ...)` | Force-directed network plot of cluster `n` |
| `plotDegCC(res, ci)` | Degree vs clustering coefficient scatter |

## `betaMix()` parameters

| Parameter | Default | Meaning |
|---|---|---|
| `M` | required | N×P DataFrame or array |
| `ind` | `True` | Independent samples? `False` estimates an effective sample size ν |
| `maxalpha` | `1e-4` | Per-pair frequentist Type I error bound (use `1e-6` for P > 1000) |
| `ppr` | `0.05` | Posterior null probability threshold for calling an edge |
| `method` | `"pearson"` | `"pearson"` or `"spearman"` |
| `subsamplesize` | `50000` | Pairs subsampled for EM; ≥ 20 000 required (ignored if smaller) |
| `bmax` | `0.999` | Upper bound of non-null support |
| `seed` | `912469` | Random seed for subsampling reproducibility |
| `msg` | `False` | Print EM iteration progress |

**Returns** a dict with: `angleMat`, `z_j`, `m0`, `p0`, `ahat`, `bhat`, `etahat`, `ppthr`, `nodes`, `edges`, `bmax`, `N`, `cnt`, `method`.

## Overlapping communities with `greedyBasis()`

Unlike `graphComponents` (disjoint) and `sphericalCaps` (hub-centric), `greedyBasis` finds a **maximal independent set (MIS)**: a set of mutually non-adjacent representative nodes.  Each remaining node is then assigned to every community whose centre it is adjacent to, giving overlapping membership.

```python
adjM  = bx.getAdjMat(res)
basis = bx.greedyBasis(adjM, strategy="min_degree", labels=M.columns)
# columns: node, isBasis, basisOrder, degree, cc, nCovered, nCommunities, communities

bx.summarizeBasis(basis)

# Nodes in multiple communities simultaneously:
bridges = basis[basis["nCommunities"] > 1]

# Members of the community centred on node 5:
community5 = basis[basis["communities"].apply(lambda c: 5 in c)]
```

Two strategies:

| Strategy | Selects first | Basis size | Community size | Overlap |
|---|---|---|---|---|
| `"min_degree"` (default) | Least-connected node | Larger | Smaller | Higher |
| `"max_degree"` | Most-connected hub | Smaller | Larger | Lower |

**Guaranteed properties**: no two basis nodes share an edge (independence); every non-basis node has at least one basis neighbour (maximality).

## Choosing the right clustering function

| | `graphComponents` | `sphericalCaps` | `greedyBasis` |
|---|---|---|---|
| Membership | Disjoint | Overlapping | Overlapping |
| Basis nodes adjacent? | Yes | Yes | **No** (independent set) |
| Selection criterion | Max degree × cc | Max degree | Min or max degree |
| Best for | General clusters | Hub-spoke networks | Non-redundant representatives |

## What's new in 0.2.x

See [NEWS.md](NEWS.md) for the full changelog.  Highlights:

- **0.2.1**: `getAdjMat` edge count now matches `betaMix()`; fixed a
  `plotFittedBetaMix` crash; unused dependencies removed; full test suite

- **`greedyBasis()`** — overlapping community detection via MIS
- **Input convention**: M is now N×P (remove any `.T` in existing code)
- **`betaMix()` returns a `dict`** instead of a tuple
- **`method="spearman"`** for rank-based correlation
- **`assessFit()`**, **`plotROC()`**, **`plotNonNullDensity()`** — new diagnostics
- **Vectorised `clusteringCoef()`** — approximately 10–80× faster
- Analytic Jacobian for MLE; `brentq` for eta estimation — matching R's solvers
- Python is approximately 3× faster than R on large benchmark datasets

## See also

- R package: [github.com/haimbar/betaMix](https://github.com/haimbar/betaMix)
- LLM-friendly function reference: [llms.txt](llms.txt)
- Version history: [NEWS.md](NEWS.md)
- Additional examples and data: [code/](code/), [data/](data/), [doc/](doc/)

## References

Bar, H. & Wells, M. T. (2023). On graphical models and convex geometry. *Computational Statistics & Data Analysis*, 187, 107800. https://doi.org/10.1016/j.csda.2023.107800

Frankl, P. & Maehara, H. (1990). Some geometric applications of the beta distribution. *Annals of the Institute of Statistical Mathematics*, 42(3), 463–474.
