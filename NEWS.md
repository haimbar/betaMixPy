# betaMixPy — Changelog

## 0.2.1 (2026-06-23)

### Bug fixes

- `getAdjMat`: changed edge condition from `< ppthr` to `<= ppthr` so the
  adjacency matrix edge count matches the value reported by `betaMix()`.
  (`betaMix` sets `ppthr` as the z_j value of the boundary pair itself, so
  that pair must be included in the adjacency matrix.)

### Removed unused dependencies

- Removed `statistics` (stdlib module) from the module-level imports — it
  was imported but never used.
- Removed `networkx` from `install_requires` in `setup.py` — the package has
  not depended on networkx since the pure-numpy graph analysis functions
  replaced it.

### Tests

- Added `tests/test_betaMixPy.py` (69 tests, all passing): covers `betaMix`,
  `getAdjMat`, `clusteringCoef`, `graphComponents`, `sphericalCaps`,
  `greedyBasis` (independence + maximality assertions), `summarizeBasis`,
  `shortestPathDistance`, `collapsedGraph`, `summarizeClusters`, and a module
  import smoke test.

### Documentation

- `doc/betaMixPy.tex`: added explicit `\date{June 2026}` and a new
  `greedyBasis` subsection in the Case Study chapter with code listing and
  cached output from the DrySeeds dataset.
- `doc/port_plan_greedyBasis.md`: step-by-step R porting guide for
  `greedyBasis()` and `summarizeBasis()`, with full code translations and test
  expectations.
- Added `code/greedyBasisDemo.py` demonstrating both strategies on DrySeeds.

## 0.2.0 (2026-06-23)

### Breaking changes

- **Input convention changed.** `betaMix()` now expects `M` as an **N×P** array
  (N samples in rows, P variables in columns), matching R and numpy's
  `corrcoef(M, rowvar=False)`. The old convention was P×N. Remove any `.T`
  transpose in existing code.

- **`betaMix()` now returns a `dict`** (not a tuple). Keys match R's named list:
  `angleMat`, `z_j`, `m0`, `p0`, `ahat`, `bhat`, `etahat`, `ppthr`, `nodes`,
  `edges`, `bmax`, `N`, `cnt`, `method`. Update any code that unpacked the
  previous tuple return value.

### New functions

| Function | Description |
|---|---|
| `greedyBasis(A, strategy, labels)` | Maximal independent set (MIS) basis with overlapping community assignment |
| `summarizeBasis(basisInfo)` | Summary statistics for a `greedyBasis()` result |
| `assessFit(res)` | 2×2 diagnostic panel: mixture histogram, null Q-Q, non-null Q-Q, ROC |
| `plotROC(res)` | Model-based ROC curve; returns AUC and Youden-index threshold |
| `plotNonNullDensity(res)` | Kernel density of non-null z_j values with mode detection |
| `summarizeClusters(ci)` | Five-number summary per cluster from `graphComponents()` |
| `collapsedGraph(res, ci)` | Reduce adjacency matrix to cluster centres |
| `shortestPathDistance(A, numSteps)` | Multi-hop distances in the network |

### `greedyBasis()` — overlapping communities

Finds a maximal independent set of the network: a set of mutually non-adjacent
representative nodes (the **basis**), then assigns each remaining node to every
community whose basis centre it is adjacent to.  This gives overlapping
membership — something `graphComponents` (disjoint) and `sphericalCaps`
(hub-centric) cannot produce.

Two strategies:

- `"min_degree"` (default): classical greedy MIS.  Selects the least-connected
  remaining node first.  Produces a sparse, non-redundant basis analogous to the
  MRB feature basis in the StatPapers companion library.
- `"max_degree"`: hub-first.  Selects the most-connected remaining node first,
  like `sphericalCaps`, but with a strict non-adjacency guarantee between basis
  nodes.

Returns a DataFrame with columns `node`, `isBasis`, `basisOrder`, `degree`,
`cc`, `nCovered`, `nCommunities`, `communities`.  Guaranteed: no two basis nodes
share an edge; every non-basis node has at least one basis neighbour (maximality).

### Performance improvements

- **EM inner loop**: `z_j = 1 − r²` replaces `sin²(arccos(r))`, eliminating
  P² trigonometric calls.  Null density precomputed outside the loop.
  Non-null support indices determined once via `np.searchsorted`.
- **`clusteringCoef()`**: vectorised as `np.sum(A * (A @ A), axis=1)` —
  approximately 10–80× faster than the previous per-node loop.
- **MLE solver**: upgraded to `scipy.optimize.root(..., jac=_jacmle,
  method="hybr")` with analytic Jacobian, matching R's `nleqslv`.
- **eta estimation**: `scipy.optimize.brentq` bracket solver, matching R's
  `uniroot`.
- **Overall**: Python is approximately 3× faster than R on hub_0200_1000
  (0.049 s vs 0.160 s).

### New features

- `method="pearson"|"spearman"` parameter in `betaMix()` for Spearman
  rank-based correlation.
- Zero-variance column detection with an informative error in `betaMix()`.
- `getAdjMat(res, signed=True)` returns a −1/0/1 adjacency matrix encoding
  negative/absent/positive edges.
- `plotCluster()` uses polar-coordinate layout with node size proportional to
  degree and opacity proportional to intra-cluster edge fraction.
- Updated `graphComponents()` with XOR-based distance metric and consistent
  column names.
- Updated `sphericalCaps()` with columns: `node`, `capNum`, `isCtr`, `deg`, `cc`.

### Validation

- DrySeeds (N=50, P=68): exact match with R (`p0=0.4751`, `ppthr=0.7767`,
  `edges=203`, `cnt=26`).
- SixHourImbibed: < 1% edge difference due to a minor difference between
  `nleqslv` (R) and scipy's `hybr` solver in a flat-likelihood region; both
  find valid solutions.

---

## 0.1.4 and earlier

Initial public release.
