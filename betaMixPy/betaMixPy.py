"""
betaMixPy: A Python package to find strong correlations among P variables,
each with N observations, using a two-component beta mixture model.

Convention: M is an N×P matrix (N rows = samples, P columns = variables),
matching the R betaMix package convention.
"""
import warnings
import statistics as stat

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.optimize import brentq, root
from scipy.special import psi, polygamma
from scipy.stats import beta as betadist, kstest


# ---------------------------------------------------------------------------
# Private helpers (precomputed-sum scalar arithmetic for speed)
# ---------------------------------------------------------------------------

def _MLEfun(par, sm_val, swt_logz, swt_log1mz):
    """Score equations for Beta(a,b) nonnull component.
    All O(n) sums are precomputed; this is O(1) scalar arithmetic per call.
    """
    a, b = par
    dab = psi(a + b)
    return np.array([
        sm_val * (psi(a) - dab) - swt_logz,
        sm_val * (psi(b) - dab) - swt_log1mz,
    ])


def _jacmle(par, sm_val, swt_logz=None, swt_log1mz=None):
    """Jacobian of _MLEfun (extra kwargs absorbed so fsolve can forward args)."""
    a, b = par
    tab = polygamma(1, a + b)
    return sm_val * np.array([
        [polygamma(1, a) - tab, -tab],
        [-tab, polygamma(1, b) - tab],
    ])


def _etafun(eta, sum_m0, sum_m0_logz):
    """Score equation for null eta.  sum_m0 and sum_m0_logz precomputed."""
    return -(psi(eta) - psi(eta + 0.5)) * sum_m0 + sum_m0_logz


def _ppoints(n):
    """Plotting positions (1:n - 0.5)/n, equivalent to R's ppoints(n)."""
    return (np.arange(1, n + 1) - 0.5) / n


# ---------------------------------------------------------------------------
# Core EM fitting
# ---------------------------------------------------------------------------

def betaMix(M, tol=1e-4, calcAcc=1e-9, ppr=0.05, maxalpha=1e-4,
            mxcnt=200, ahat=8, bhat=3, subsamplesize=50000, seed=912469,
            bmax=0.999, ind=True, msg=True, method="pearson"):
    """Fit a two-component beta mixture model to all pairwise correlations.

    M -- N×P matrix: N rows = samples, P columns = variables.
    tol -- EM convergence threshold (default 1e-4).
    calcAcc -- numerical accuracy guard (default 1e-9).
    ppr -- null posterior probability threshold (default 0.05).
    maxalpha -- Type I error bound (default 1e-4).
    mxcnt -- max EM iterations (default 200).
    ahat, bhat -- initial nonnull Beta parameters (default 8, 3).
    subsamplesize -- subsample size when P*(P-1)/2 > this (default 50000).
    seed -- RNG seed for subsampling (default 912469).
    bmax -- right endpoint of nonnull support (default 0.999).
    ind -- assume independent samples (default True).
    msg -- print progress (default True).
    method -- "pearson" (default) or "spearman".

    Returns a dict with keys:
      angleMat  -- P×P Pearson/Spearman correlation matrix.
      z_j       -- sorted z_j = 1 - r^2 statistics.
      m0        -- posterior null probabilities.
      p0        -- estimated null proportion.
      ahat, bhat -- fitted nonnull Beta parameters.
      etahat    -- effective sample-size parameter (ESS = 2*etahat+1).
      ppthr     -- detection threshold on z_j scale.
      nodes     -- number of variables P.
      edges     -- number of detected edges.
      bmax      -- right endpoint of nonnull support.
      N         -- sample size.
      cnt       -- number of EM iterations.
      method    -- correlation method used.
    """
    if not (0 < bmax < 1):
        raise ValueError(f"betaMix: 'bmax' must be in (0, 1); got {bmax}")
    if not (0 < calcAcc < 0.5):
        raise ValueError(f"betaMix: 'calcAcc' must be in (0, 0.5); got {calcAcc}")

    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError("betaMix: M must be a 2-D array")
    N, P = M.shape
    if N < 2:
        raise ValueError(f"betaMix: M must have >= 2 rows (samples); got N={N}")
    if P < 2:
        raise ValueError(f"betaMix: M must have >= 2 columns (variables); got P={P}")

    etahat = (N - 1) / 2.0

    # Warn about zero-variance columns
    col_std = np.std(M, axis=0, ddof=1)
    const_cols = np.where(col_std == 0)[0]
    if len(const_cols) > 0:
        warnings.warn(
            f"betaMix: {len(const_cols)} constant column(s) at index "
            f"{const_cols.tolist()}; their correlations are set to 0"
        )

    if msg:
        print("Generating the z_ij statistics...")

    # Correlation matrix P×P (rowvar=False: each column is a variable)
    if method == "spearman":
        from scipy.stats import rankdata
        ranked = np.apply_along_axis(rankdata, 0, M)
        corM = np.corrcoef(ranked, rowvar=False)
    else:
        corM = np.corrcoef(M, rowvar=False)

    if np.any(np.isnan(corM)):
        n_nans = int(np.sum(np.isnan(corM)))
        warnings.warn(f"betaMix: {n_nans} NaN(s) in correlation matrix; replaced with 0")
        corM = np.nan_to_num(corM, nan=0.0)

    # Store corM directly (sin^2(arccos(r)) == 1-r^2; avoids P^2 trig calls)
    angleMat = corM

    # Lower-triangle statistics: z_j = 1 - r^2, clipped to [calcAcc, 1-calcAcc]
    lower = np.tril_indices(P, k=-1)
    z_j = np.clip(1.0 - corM[lower] ** 2, calcAcc, 1.0 - calcAcc)

    z_jall = np.array([], dtype=float)
    if len(z_j) > subsamplesize and subsamplesize >= 20000:
        z_jall = np.sort(z_j)
        rng = np.random.default_rng(seed)
        z_j = np.sort(rng.choice(z_jall, subsamplesize, replace=False))
    else:
        z_j = np.sort(z_j)

    tol = max(tol, 1.0 / len(z_j))

    # O(log n) initial p0 via binary search on sorted z_j
    thr0 = betadist.ppf(0.1, etahat, 0.5)
    idx0 = len(z_j) - np.searchsorted(z_j, thr0, side="left")
    p0 = min(1.0, idx0 / (0.9 * len(z_j)))

    if msg:
        print("Fitting the model...")

    # Precompute nonnull support once (z_j sorted, bmax constant)
    nns_count = int(np.searchsorted(z_j, bmax, side="left"))
    nns = np.arange(nns_count)
    z_j_nns = z_j[nns] / bmax

    # Interior mask and log values for MLEfun (constant throughout EM)
    inc = np.where((z_j_nns > 1e-6) & (z_j_nns < 1.0 - 1e-6))[0]
    log_z   = np.log(z_j_nns[inc])
    log1mz  = np.log(1.0 - z_j_nns[inc])

    # Initialise m0; m0 outside nns stays 1 throughout
    m0 = np.ones(len(z_j))
    if len(nns) > 0:
        dbeta0_nns = betadist.pdf(z_j[nns], etahat, 0.5)
        p0f0_n = p0 * dbeta0_nns
        p1f1_n = (1.0 - p0) * betadist.pdf(z_j_nns, ahat, bhat)
        denom = p0f0_n + p1f1_n
        if np.any(denom == 0):
            warnings.warn("betaMix E-step (init): zero denominator; treating as null")
        m0[nns] = np.where(denom > 0, p0f0_n / denom, 1.0)

    p0new = p0 - 10.0 * tol
    cnt = 0
    while abs(p0 - p0new) > tol and cnt < mxcnt:
        cnt += 1
        p0 = p0new

        # --- eta update (ind=False only) ---
        if not ind:
            sum_m0      = float(np.sum(m0))
            sum_m0_logz = float(np.sum(m0 * np.log(z_j)))
            try:
                lo_val = _etafun(1.0, sum_m0, sum_m0_logz)
                hi_val = _etafun((N - 1) / 2.0, sum_m0, sum_m0_logz)
                if lo_val * hi_val < 0:
                    etahat = brentq(_etafun, 1.0, (N - 1) / 2.0,
                                    args=(sum_m0, sum_m0_logz))
                    if len(nns) > 0:
                        dbeta0_nns = betadist.pdf(z_j[nns], etahat, 0.5)
            except Exception as e:
                warnings.warn(f"betaMix: eta estimation failed ({e})")

        # --- M-step: update ahat, bhat ---
        m_nns = m0[nns]
        sm_val = float(np.sum(1.0 - m_nns))
        if not np.any(np.isnan(m_nns)) and sm_val >= 1e-10:
            wt         = 1.0 - m_nns[inc]
            swt_logz   = float(np.sum(wt * log_z))
            swt_log1mz = float(np.sum(wt * log1mz))
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    sol = root(_MLEfun, [ahat, bhat],
                               args=(sm_val, swt_logz, swt_log1mz),
                               jac=_jacmle, method="hybr",
                               options={"xtol": 1.49e-8})
                if sol.success or np.allclose(sol.fun, 0, atol=1e-6):
                    ahat = float(np.clip(sol.x[0], 0.5, 1000.0))
                    bhat = float(np.clip(sol.x[1], 0.5, 1000.0))
            except Exception as e:
                warnings.warn(f"betaMix: ahat/bhat estimation failed ({e})")

        # --- E-step: reuse dbeta0_nns; m0 outside nns stays 1 ---
        if len(nns) > 0:
            p0f0_n = p0 * dbeta0_nns
            p1f1_n = (1.0 - p0) * betadist.pdf(z_j_nns, ahat, bhat)
            denom  = p0f0_n + p1f1_n
            if np.any(denom == 0):
                warnings.warn("betaMix E-step: zero denominator; treating as null")
            m0[nns] = np.where(denom > 0, p0f0_n / denom, 1.0)

        # Efficient p0 update: m0 outside nns is always 1
        p0new = (float(np.sum(m0[nns])) + len(z_j) - len(nns)) / len(z_j)

    # Expand to full z_jall if we subsampled
    if len(z_jall) > 0:
        z_j = z_jall

    # Final E-step on full z_j
    nns_f_count = int(np.searchsorted(z_j, bmax, side="left"))
    nns_f = np.arange(nns_f_count)
    m0 = np.ones(len(z_j))
    if len(nns_f) > 0:
        z_j_nns_f = z_j[nns_f] / bmax
        p0f0_n = p0 * betadist.pdf(z_j[nns_f], etahat, 0.5)
        p1f1_n = (1.0 - p0) * betadist.pdf(z_j_nns_f, ahat, bhat)
        denom  = p0f0_n + p1f1_n
        if np.any(denom == 0):
            warnings.warn("betaMix final E-step: zero denominator; treating as null")
        m0[nns_f] = np.where(denom > 0, p0f0_n / denom, 1.0)

    p0 = float(np.mean(m0))
    ppthr = betadist.ppf(maxalpha, etahat, 0.5)
    critPP = np.where(m0 < ppr)[0]
    if len(critPP) > 0:
        ppthr = float(np.max(z_j[critPP]))
    edges = int(np.sum(z_j <= ppthr))

    if msg:
        print("Done.")

    return dict(
        angleMat=angleMat, z_j=z_j, m0=m0, p0=p0,
        ahat=ahat, bhat=bhat, etahat=etahat,
        ppthr=ppthr, nodes=P, edges=edges,
        bmax=bmax, N=N, cnt=cnt, method=method,
    )


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------

def getAdjMat(res, ppthr=None, signed=False, nodes=None):
    """Build the adjacency matrix from a fitted betaMix result.

    res    -- dict returned by betaMix().
    ppthr  -- detection threshold; defaults to res['ppthr'].
    signed -- if True, -1 for negative correlations (default False).
    nodes  -- list of node indices (0-based) for a sparse submatrix (default None).

    Returns a P×P numpy array (or submatrix when nodes is given).
    """
    if ppthr is None:
        ppthr = res["ppthr"]
    corM = res["angleMat"]
    P = corM.shape[0]

    if nodes is not None:
        nodes = np.asarray(nodes, dtype=int)
        if np.any(nodes < 0) or np.any(nodes >= P):
            raise ValueError(f"getAdjMat: 'nodes' must be in [0, {P-1}]")
        # Find all neighbors of the requested nodes
        rows = corM[nodes, :]        # shape (len(nodes), P)
        is_edge = (1.0 - rows ** 2) < ppthr  # bool, shape (len(nodes), P)
        nbr_mask = np.any(is_edge, axis=0)
        selected = np.sort(np.union1d(nodes, np.where(nbr_mask)[0]))
        corM = corM[np.ix_(selected, selected)]

    A = (1.0 - corM ** 2) < ppthr
    if signed:
        A = A.astype(float)
        neg = (corM < 0) & (A > 0)
        A[neg] = -1.0
    else:
        A = A.astype(float)
    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------------------------
# Graph / network analysis
# ---------------------------------------------------------------------------

def clusteringCoef(A):
    """Clustering coefficient of each node.  Vectorised via matrix multiply.

    A -- adjacency matrix (N×N numpy array, 0/1).

    Returns array of length N.
    """
    A = np.abs(np.asarray(A, dtype=float))
    deg = A.sum(axis=1)
    # tri[i] = number of closed triangles through node i (each counted twice)
    tri = np.sum(A * (A @ A), axis=1)
    denom = deg * (deg - 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tri / denom, 0.0)


def graphComponents(A, minCtr=5, wgt=1):
    """Find graph components from an adjacency matrix.

    For each node compute degree and clustering coefficient (CC). The centrality
    measure is (wgt*CC+1)*deg; for wgt=0 it's plain degree; for wgt<0 it's CC*deg.

    A      -- adjacency matrix (P×P numpy array).
    minCtr -- min centrality to consider a node a cluster centre (default 5).
    wgt    -- weight for the centrality measure (default 1).

    Returns a DataFrame with columns:
      labels, degree, cc, ctr, clustNo, iscenter, intEdges, extEdges, distCenter.
    """
    A = np.abs(np.asarray(A, dtype=float))
    Vn = A.shape[0]
    deg = A.sum(axis=1)
    CC  = clusteringCoef(A)
    ctrs = (wgt * CC + 1.0) * deg if wgt >= 0 else CC * deg

    labels     = list(range(Vn))
    clustNo    = np.zeros(Vn, dtype=int)
    iscenter   = np.zeros(Vn, dtype=int)
    intEdges   = np.zeros(Vn, dtype=float)
    extEdges   = np.zeros(Vn, dtype=float)
    distCenter = np.zeros(Vn, dtype=float)

    clNum    = 1
    clustered = set(np.where(deg < 1)[0].tolist())

    while len(clustered) < Vn:
        not_in = np.array([i for i in range(Vn) if i not in clustered])
        if len(not_in) == 0 or ctrs[not_in].max() < minCtr:
            break
        ctrnode = int(not_in[np.argmax(ctrs[not_in])])
        # Candidate cluster = center + its neighbors, excluding already clustered
        raw_nbrs = np.where(A[ctrnode, :] != 0)[0]
        nbrs = np.array(
            sorted(set(np.append(raw_nbrs, ctrnode).tolist()) - clustered),
            dtype=int,
        )
        if len(nbrs) > 1 and len(nbrs) > minCtr:
            iscenter[ctrnode] = 1
            clustNo[nbrs] = clNum
            intEdges[nbrs] = A[np.ix_(nbrs, nbrs)].sum(axis=1)
            not_nbrs = np.array([i for i in range(Vn) if i not in set(nbrs.tolist())], dtype=int)
            if len(not_nbrs) > 0:
                extEdges[nbrs] = A[np.ix_(nbrs, not_nbrs)].sum(axis=1)
            for ni in nbrs:
                # XOR distance: fraction of neighbors where center and node differ
                distCenter[ni] = float(np.mean(A[ctrnode, :] != A[ni, :]))
            clNum += 1
            clustered.update(nbrs.tolist())
        else:
            clustered.add(ctrnode)

    return pd.DataFrame({
        "labels":      labels,
        "degree":      deg,
        "cc":          CC,
        "ctr":         ctrs,
        "clustNo":     clustNo,
        "iscenter":    iscenter,
        "intEdges":    intEdges,
        "extEdges":    extEdges,
        "distCenter":  distCenter,
    })


def sphericalCaps(A):
    """Find hub structures (spherical caps) in the network graph.

    Greedily selects high-degree nodes as cap centres. Each centre and all its
    direct neighbours form one cap. Each node can be a centre at most once.

    A -- adjacency matrix (P×P numpy array).

    Returns a DataFrame with columns: node, capNum, isCtr, deg, cc.
    """
    A   = np.abs(np.asarray(A, dtype=float))
    P   = A.shape[0]
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    CC  = clusteringCoef(A)

    possible = np.where(deg > 0)[0]
    if len(possible) == 0:
        print("No edges!")
        return None

    orddeg  = list(np.argsort(deg)[::-1])
    capNum  = 1
    chunks  = []
    orddeg_set = set(orddeg)

    while orddeg and deg[orddeg[0]] > 0:
        ctr  = orddeg[0]
        nbrs = np.where(A[ctr, :] != 0)[0].tolist()
        remove = set([ctr] + nbrs)
        orddeg = [n for n in orddeg if n not in remove]
        for nd, is_c in [(ctr, 1)] + [(n, 0) for n in nbrs]:
            chunks.append({
                "node":   nd,
                "capNum": capNum,
                "isCtr":  is_c,
                "deg":    deg[nd],
                "cc":     CC[nd],
            })
        capNum += 1
        if not orddeg:
            break

    if not chunks:
        return None
    df = pd.DataFrame(chunks)
    df.index = range(len(df))
    return df


def greedyBasis(A, strategy="min_degree", labels=None):
    """Find a maximal independent set (MIS) and derive overlapping communities.

    Repeatedly picks the node with the smallest or largest current degree in the
    remaining subgraph, adds it to the basis, then removes it and all its active
    neighbours.  After the basis is found, each node is assigned to every
    community whose centre it is adjacent to, giving overlapping membership —
    something graphComponents and sphericalCaps cannot produce.

    strategy="min_degree" (default)
        Classical greedy MIS: selects the least-connected remaining node first.
        Produces a sparse, high-coverage basis of mutually non-adjacent
        representatives.  Used in the StatPapers MRB method as a
        non-redundant feature basis.
    strategy="max_degree"
        Hub-first: selects the most-connected remaining node first.
        Like sphericalCaps but with a strict non-adjacency guarantee
        between basis nodes — the most influential hubs without redundancy.

    A        -- adjacency matrix (P×P numpy array, 0/1 or signed).
    strategy -- "min_degree" (default) or "max_degree".
    labels   -- optional list/array of P node labels (used as DataFrame index).

    Returns a DataFrame with one row per node and columns:
      node         -- integer node index (0-based).
      isBasis      -- True if the node is in the MIS.
      basisOrder   -- selection order 1, 2, …; 0 for non-basis nodes.
      degree       -- original node degree in A.
      cc           -- clustering coefficient.
      nCovered     -- size of this node's community (basis only; 0 otherwise).
      nCommunities -- number of communities this node belongs to.
      communities  -- list of basis-node indices whose community includes this node.

    Key properties guaranteed by construction:
      * No two basis nodes share an edge (independent set).
      * Every non-basis node is adjacent to at least one basis node (maximal).
      * A non-basis node belongs to as many communities as it has basis neighbours
        (overlapping membership).
    """
    if strategy not in ("min_degree", "max_degree"):
        raise ValueError("greedyBasis: strategy must be 'min_degree' or 'max_degree'")

    A         = np.abs(np.asarray(A, dtype=float))
    np.fill_diagonal(A, 0.0)
    P         = A.shape[0]
    adj_bool  = A > 0                               # (P, P) bool
    orig_deg  = adj_bool.sum(axis=1).astype(int)
    CC        = clusteringCoef(A)

    # ── Phase 1: greedy MIS ───────────────────────────────────────────────
    # Maintain current-subgraph degrees and update them after each removal,
    # so argmin/argmax always reflects the live subgraph — O(P²) total,
    # fast enough for typical betaMixPy sizes (P ≤ few thousand).
    deg       = orig_deg.astype(float).copy()
    active    = np.ones(P, dtype=bool)
    is_basis  = np.zeros(P, dtype=bool)
    basis_ord = np.zeros(P, dtype=int)
    pick_fn   = np.argmin if strategy == "min_degree" else np.argmax

    order = 0
    while active.any():
        active_idx = np.where(active)[0]
        pick = active_idx[pick_fn(deg[active_idx])]

        order += 1
        is_basis[pick]  = True
        basis_ord[pick] = order

        # Nodes to remove: pick itself + its currently active neighbours
        nbrs_active = np.where(adj_bool[pick] & active)[0]
        deact_mask  = np.zeros(P, dtype=bool)
        deact_mask[pick]        = True
        deact_mask[nbrs_active] = True

        # Update degrees of nodes that will remain active
        remain_idx = np.where(active & ~deact_mask)[0]
        deact_idx  = np.where(deact_mask)[0]
        if len(remain_idx) > 0 and len(deact_idx) > 0:
            loss = adj_bool[np.ix_(remain_idx, deact_idx)].sum(axis=1)
            deg[remain_idx] -= loss

        active[deact_mask] = False

    basis_nodes = np.where(is_basis)[0]

    # ── Phase 2: overlapping community assignment ─────────────────────────
    # Each node belongs to all communities of the basis nodes it is adjacent
    # to in the ORIGINAL adjacency matrix.  A basis node belongs to its own
    # community only (since no two basis nodes are adjacent).
    communities = [[] for _ in range(P)]
    n_covered   = np.zeros(P, dtype=int)

    for b in basis_nodes:
        all_nbrs   = np.where(adj_bool[b])[0]
        n_covered[b] = 1 + len(all_nbrs)          # b covers itself + all adj nodes
        communities[b].append(int(b))
        for m in all_nbrs:
            communities[m].append(int(b))

    n_communities = np.array([len(c) for c in communities], dtype=int)

    df = pd.DataFrame({
        "node":         np.arange(P),
        "isBasis":      is_basis,
        "basisOrder":   basis_ord,
        "degree":       orig_deg,
        "cc":           CC,
        "nCovered":     n_covered,
        "nCommunities": n_communities,
        "communities":  communities,
    })

    if labels is not None:
        lbl = list(labels)
        if len(lbl) == P:
            df.index = lbl
        else:
            warnings.warn(
                f"greedyBasis: len(labels)={len(lbl)} != P={P}; labels ignored"
            )

    return df


def summarizeBasis(basisInfo):
    """Print and return summary statistics for a greedyBasis result.

    basisInfo -- DataFrame returned by greedyBasis().

    Returns a dict with: n_basis, basis_fraction, mean_community_size,
    overlap_stats (mean/max nCommunities for non-basis nodes),
    isolated_nodes.
    """
    bi = basisInfo
    P          = len(bi)
    n_basis    = int(bi["isBasis"].sum())
    n_edges    = int(bi["degree"].sum()) // 2
    n_isolated = int((bi["degree"] == 0).sum())
    non_basis  = bi[~bi["isBasis"]]

    mean_cov  = float(bi.loc[bi["isBasis"], "nCovered"].mean()) if n_basis > 0 else 0.0
    max_cov   = int(bi["nCovered"].max())
    overlap   = non_basis["nCommunities"]
    mean_ovlp = float(overlap.mean()) if len(overlap) > 0 else 0.0
    max_ovlp  = int(overlap.max()) if len(overlap) > 0 else 0
    n_shared  = int((overlap > 1).sum())

    print(f"Nodes: {P}   Edges: {n_edges}   Isolated: {n_isolated}")
    print(f"Basis size (MIS): {n_basis}  ({100*n_basis/P:.1f}% of nodes)")
    print(f"Mean community size: {mean_cov:.1f}   Max: {max_cov}")
    print(f"Non-basis nodes in >1 community: {n_shared} / {len(non_basis)} "
          f"(mean {mean_ovlp:.2f}, max {max_ovlp})")

    return dict(
        n_basis=n_basis,
        basis_fraction=n_basis / P,
        mean_community_size=mean_cov,
        max_community_size=max_cov,
        n_shared_nodes=n_shared,
        mean_n_communities=mean_ovlp,
        max_n_communities=max_ovlp,
        n_isolated=n_isolated,
    )


def summarizeClusters(clustersInfo):
    """Print and return summary statistics for each cluster.

    clustersInfo -- DataFrame returned by graphComponents().

    Returns a matrix with cluster, nodes, degree fivenum, pctInCluster fivenum.
    """
    ci = clustersInfo
    print("Num of nodes:", len(ci))
    print("Num of edges:", int(ci["degree"].sum() / 2))
    n_clusters = int(ci["clustNo"].max())
    print("Num of clusters:", n_clusters)
    print("Num of unclustered nodes:", int((ci["clustNo"] == 0).sum()))

    pct = np.where(ci["degree"] > 0,
                   ci["intEdges"] / ci["degree"], 0.0)
    if n_clusters == 0:
        return None

    rows = []
    for cnum in range(1, n_clusters + 1):
        idx = ci["clustNo"] == cnum
        d = ci["degree"][idx].values
        p = pct[idx]
        row = [cnum, idx.sum(),
               np.min(d), np.percentile(d, 25), np.median(d),
               np.percentile(d, 75), np.max(d),
               np.min(p), np.percentile(p, 25), np.median(p),
               np.percentile(p, 75), np.max(p)]
        rows.append(row)

    tab = np.array(rows)
    cols = ["Cluster", "Nodes",
            "degMin", "degQ25", "degMedian", "degQ75", "degMax",
            "pctMin", "pctQ25", "pctMedian", "pctQ75", "pctMax"]
    return pd.DataFrame(tab, columns=cols)


def collapsedGraph(A, clustersInfo):
    """Collapse clusters into their centres; return a weighted adjacency matrix.

    A            -- adjacency matrix (P×P numpy array).
    clustersInfo -- DataFrame returned by graphComponents().

    Returns a square numpy array of dimension (unclustered nodes + clusters).
    """
    ci = clustersInfo
    n_clusters = int(ci["clustNo"].max())
    not_in = np.where(ci["clustNo"] == 0)[0]
    ni = len(not_in)
    dim = ni + n_clusters
    collA = np.zeros((dim, dim), dtype=float)

    # Unclustered × unclustered block
    if ni > 0:
        collA[:ni, :ni] = (A[np.ix_(not_in, not_in)] > 0).astype(float)

    cluster_idx = [np.where(ci["clustNo"] == k)[0] for k in range(1, n_clusters + 1)]

    for i in range(n_clusters):
        Ci = cluster_idx[i]
        if ni > 0:
            Atmp = A[np.ix_(not_in, Ci)]
            collA[ni + i, :ni] = Atmp.sum(axis=1)
        for j in range(i + 1, n_clusters):
            collA[ni + i, ni + j] = A[np.ix_(Ci, cluster_idx[j])].sum()

    return collA + collA.T


def shortestPathDistance(AdjMat, numSteps=0):
    """Shortest-path distances in the network up to numSteps hops beyond direct edges.

    AdjMat   -- adjacency matrix (P×P numpy array).
    numSteps -- additional hops beyond direct neighbours (default 0 = return AdjMat).

    Returns a P×P float array; entry (i,j) is the shortest-path distance
    (1 = direct edge, 2 = two-hop, ...).  Unreachable pairs remain 0.
    """
    if not (isinstance(numSteps, (int, np.integer)) and numSteps >= 0):
        raise ValueError(f"shortestPathDistance: 'numSteps' must be a non-negative int; got {numSteps}")
    if numSteps == 0:
        return AdjMat.copy()

    B        = (AdjMat != 0).astype(float)
    minDist  = B.copy()                   # 1 for direct edges
    frontier = B.astype(bool)
    visited  = B.astype(bool)
    np.fill_diagonal(visited, True)

    for step in range(numSteps):
        nextFrontier = (frontier.astype(float) @ B) > 0  # bool matrix
        nextFrontier[visited] = False
        if not np.any(nextFrontier):
            break
        minDist[nextFrontier] = step + 2
        visited  = visited | nextFrontier
        frontier = nextFrontier

    return minDist


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def shortSummary(betaMixObj):
    """Print a short summary of the fitted model.

    betaMixObj -- dict returned by betaMix().
    """
    r = betaMixObj
    print(f"Nonnull support = [0, {r['bmax']:.3g}]")
    print(f"ahat = {r['ahat']:.3g},  bhat = {r['bhat']:.3g}")
    print(f"etahat = {r['etahat']:.3g}")
    print(f"Post. Pr. threshold = {r['ppthr']:.3g}")
    print(f"Sample size = {r['N']}")
    print(f"No. nodes = {r['nodes']:,}")
    print(f"Max no. edges = {r['nodes'] * (r['nodes'] - 1) // 2:,}")
    print(f"No. edges detected = {r['edges']:,}")
    print(f"p0 = {r['p0']:.4f}")
    print(f"EM iterations = {r['cnt']}")
    print(f"Correlation method = {r['method']}")


# ---------------------------------------------------------------------------
# Goodness-of-fit diagnostics
# ---------------------------------------------------------------------------

def assessFit(betaMixObj, yLim=5, qq_purity=0.9):
    """Produce a 2×2 diagnostic plot and print a goodness-of-fit assessment.

    betaMixObj -- dict returned by betaMix().
    yLim       -- y-axis max for the fitted-mixture panel (default 5).
    qq_purity  -- m0 threshold for 'pure' null/nonnull Q-Q subsets (default 0.9).

    Returns a dict with KS test results, mode count, and overall assessment.
    """
    r = betaMixObj
    z_j    = r["z_j"]
    m0     = r["m0"]
    ahat   = r["ahat"]
    bhat   = r["bhat"]
    etahat = r["etahat"]
    bmax   = r["bmax"]
    ppthr  = r["ppthr"]
    edges  = r["edges"]
    p0     = r["p0"]

    # Hard assignment
    null_idx    = np.where(m0 >= 0.5)[0]
    nonnull_idx = np.where(m0 < 0.5)[0]
    n_null    = len(null_idx)
    n_nonnull = len(nonnull_idx)

    # KS tests
    ks_null = ks_nonnull = None
    if n_null >= 2:
        ks_null = kstest(z_j[null_idx], lambda x: betadist.cdf(x, etahat, 0.5))
    z_nn_raw = z_j[nonnull_idx] / bmax if n_nonnull > 0 else np.array([])
    z_nn = z_nn_raw[(z_nn_raw > 1e-6) & (z_nn_raw < 1 - 1e-6)]
    if len(z_nn) >= 2:
        ks_nonnull = kstest(z_nn, lambda x: betadist.cdf(x, ahat, bhat))

    # Pure subsets for conditional Q-Q
    pure_null_idx    = np.where(m0 >= qq_purity)[0]
    pure_nonnull_idx = np.where(m0 <= 1.0 - qq_purity)[0]
    n_null_pure = len(pure_null_idx)
    z_nn_pure_raw = z_j[pure_nonnull_idx] / bmax if len(pure_nonnull_idx) > 0 else np.array([])
    z_nn_pure = z_nn_pure_raw[(z_nn_pure_raw > 1e-6) & (z_nn_pure_raw < 1 - 1e-6)]
    n_nonnull_pure = len(z_nn_pure)

    ks_null_pure = ks_nonnull_pure = None
    null_D_pure  = nonnull_D_pure  = np.nan

    if n_null_pure >= 2:
        z_np   = np.sort(z_j[pure_null_idx])
        p_low  = betadist.cdf(z_np[0], etahat, 0.5)
        if p_low < 1.0 - 1e-15:
            cdf_cond = lambda x: (betadist.cdf(x, etahat, 0.5) - p_low) / (1.0 - p_low)
            ks_null_pure = kstest(z_np, cdf_cond)
            null_D_pure  = float(ks_null_pure.statistic)

    if n_nonnull_pure >= 2:
        z_np   = np.sort(z_nn_pure)
        p_high = betadist.cdf(z_np[-1], ahat, bhat)
        if p_high > 1e-15:
            cdf_cond = lambda x: betadist.cdf(x, ahat, bhat) / p_high
            ks_nonnull_pure = kstest(z_np, cdf_cond)
            nonnull_D_pure  = float(ks_nonnull_pure.statistic)

    # Multimodality in nonnull
    n_modes = np.nan
    peaks_above_thr = []
    if n_nonnull >= 10:
        dens_nn = _kernel_density(z_j[nonnull_idx])
        peak_thr = 0.10 * dens_nn["y"].max()
        dy  = dens_nn["y"]
        nd  = len(dy)
        is_peak = np.concatenate([[False],
                                  (dy[1:-1] > dy[:-2]) & (dy[1:-1] > dy[2:]),
                                  [False]])
        peaks_above_thr = np.where(is_peak & (dy >= peak_thr))[0]
        n_modes = len(peaks_above_thr)

    left_tail_frac = float(np.mean(z_j[nonnull_idx] < 0.05)) if n_nonnull > 0 else 0.0
    heavy_left = left_tail_frac > 0.20

    null_D    = float(ks_null.statistic)    if ks_null    is not None else np.nan
    nonnull_D = float(ks_nonnull.statistic) if ks_nonnull is not None else np.nan
    null_dev    = not np.isnan(null_D)    and null_D    > 0.05
    nonnull_dev = not np.isnan(nonnull_D) and nonnull_D > 0.05
    multimodal  = not np.isnan(n_modes) and n_modes > 1
    no_signal   = edges == 0
    extreme_p0  = p0 < 0.01

    n_issues = sum([null_dev, nonnull_dev, multimodal, heavy_left, no_signal, extreme_p0])
    assessment = ("Good fit" if n_issues == 0
                  else "Acceptable fit" if n_issues == 1
                  else "Deviations detected")

    # 2×2 plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel 1: fitted mixture
    plt.sca(axes[0, 0])
    _plot_fitted_panel(betaMixObj, yLim)
    axes[0, 0].set_title("Fitted mixture")

    # Panel 2: Q-Q null
    ax = axes[0, 1]
    use_pure_null = n_null_pure >= 10
    if use_pure_null or n_null >= 2:
        if use_pure_null:
            z_sorted  = np.sort(z_j[pure_null_idx])
            n_qq      = n_null_pure
            p_low_qq  = min(betadist.cdf(z_sorted[0], etahat, 0.5), 1 - 1e-14)
            q_theo    = betadist.ppf(p_low_qq + _ppoints(n_qq) * (1 - p_low_qq),
                                     etahat, 0.5)
            d_show    = null_D_pure
            note      = f"m0>={qq_purity:.2f} (n={n_null_pure} of {n_null})"
        else:
            z_sorted  = np.sort(z_j[null_idx])
            n_qq      = n_null
            q_theo    = betadist.ppf(_ppoints(n_qq), etahat, 0.5)
            d_show    = null_D
            note      = f"m0>=0.50 fallback (n={n_null})"
        col = "tomato" if (not np.isnan(d_show) and d_show > 0.05) else "steelblue"
        ax.scatter(q_theo, z_sorted, c=col, s=3, alpha=0.6)
        lo = min(q_theo[0], z_sorted[0]); hi = max(q_theo[-1], z_sorted[-1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
        ax.set_xlabel(f"Theoretical Beta({etahat:.2f}, 0.5)")
        ax.set_ylabel("Sample quantiles (null)")
        ax.set_title(f"Q-Q: Null  (D={'NA' if np.isnan(d_show) else f'{d_show:.4f}'})")
        ax.text(0.5, 0.02, note, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7, color="grey")
    else:
        ax.text(0.5, 0.5, "Insufficient null\nobservations", ha="center", va="center")
        ax.set_title("Q-Q: Null (no data)")

    # Panel 3: Q-Q nonnull
    ax = axes[1, 0]
    use_pure_nn = n_nonnull_pure >= 10
    if use_pure_nn or len(z_nn) >= 2:
        if use_pure_nn:
            z_sorted  = np.sort(z_nn_pure)
            n_qq      = n_nonnull_pure
            p_high_qq = max(betadist.cdf(z_sorted[-1], ahat, bhat), 1e-14)
            q_theo    = betadist.ppf(_ppoints(n_qq) * p_high_qq, ahat, bhat)
            d_show    = nonnull_D_pure
            note      = f"m0<={1-qq_purity:.2f} (n={n_nonnull_pure} of {len(z_nn)})"
        else:
            z_sorted  = np.sort(z_nn)
            n_qq      = len(z_nn)
            q_theo    = betadist.ppf(_ppoints(n_qq), ahat, bhat)
            d_show    = nonnull_D
            note      = f"m0<0.50 fallback (n={n_qq})"
        col = "tomato" if (not np.isnan(d_show) and d_show > 0.05) else "steelblue"
        ax.scatter(q_theo, z_sorted, c=col, s=3, alpha=0.6)
        lo = min(q_theo[0], z_sorted[0]); hi = max(q_theo[-1], z_sorted[-1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
        ax.set_xlabel(f"Theoretical Beta({ahat:.2f}, {bhat:.2f})")
        ax.set_ylabel(f"Sample quantiles (non-null, z_j/{bmax:.3f})")
        ax.set_title(f"Q-Q: Non-null  (D={'NA' if np.isnan(d_show) else f'{d_show:.4f}'})")
        ax.text(0.5, 0.02, note, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7, color="grey")
    else:
        ax.text(0.5, 0.5, "Insufficient non-null\nobservations", ha="center", va="center")
        ax.set_title("Q-Q: Non-null (no data)")

    # Panel 4: ROC
    plt.sca(axes[1, 1])
    roc = _plot_roc_panel(betaMixObj)

    plt.tight_layout()

    # Verbal assessment
    print("=== assessFit: Goodness-of-Fit Summary ===")
    print(f"  Hard-assigned null: {n_null},  non-null: {n_nonnull}")
    _ksd = "NA" if np.isnan(null_D) else f"{null_D:.4f}"
    _tag = "NOTABLE - D > 0.05" if null_dev else "OK"
    print(f"  Null component   KS D = {_ksd}  [{_tag}]")
    if null_dev:
        print("    Hint: null deviation may indicate dependent samples; try ind=False.")
    _ksd = "NA" if np.isnan(nonnull_D) else f"{nonnull_D:.4f}"
    _tag = "NOTABLE - D > 0.05" if nonnull_dev else "OK"
    print(f"  Non-null component   KS D = {_ksd}  [{_tag}]")
    if nonnull_dev:
        print("    Hint: non-null deviation may indicate multimodal/heavy-tailed distribution.")
    print(f"  Conditional Q-Q (m0>={qq_purity:.2f} / m0<={1-qq_purity:.2f}):")
    print(f"    Null D = {'NA' if np.isnan(null_D_pure) else f'{null_D_pure:.4f}'} (n={n_null_pure}),  "
          f"Non-null D = {'NA' if np.isnan(nonnull_D_pure) else f'{nonnull_D_pure:.4f}'} (n={n_nonnull_pure})")
    if not np.isnan(n_modes):
        _tag = "NOTABLE - multimodal" if multimodal else "OK"
        print(f"  Non-null modes: {int(n_modes)}  [{_tag}]")
    print(f"  Heavy left tail (z_j<0.05): {100*left_tail_frac:.1f}%  "
          f"[{'NOTABLE' if heavy_left else 'OK'}]")
    if no_signal:
        print("  No edges detected  [WARNING: no non-null signal found]")
    print(f"  p0 = {p0:.4f}  "
          f"[{'WARNING: ill-conditioned null' if extreme_p0 else 'OK'}]")
    print(f"  Overall verdict: >>> {assessment} <<<")
    print("==========================================")

    return dict(
        ks_null=ks_null, ks_nonnull=ks_nonnull,
        ks_null_pure=ks_null_pure, ks_nonnull_pure=ks_nonnull_pure,
        n_null=n_null, n_nonnull=n_nonnull,
        n_null_pure=n_null_pure, n_nonnull_pure=n_nonnull_pure,
        n_modes=n_modes, assessment=assessment,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _kernel_density(x, n=512):
    """Simple Gaussian KDE returning a dict with x/y arrays."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x)
    xs  = np.linspace(x.min(), x.max(), n)
    return {"x": xs, "y": kde(xs)}


def _plot_fitted_panel(betaMixObj, yLim=5):
    """Draw the fitted-mixture histogram on the current axes (no show/save)."""
    r = betaMixObj
    z_j, p0, ahat, bhat, etahat, bmax, ppthr = (
        r["z_j"], r["p0"], r["ahat"], r["bhat"], r["etahat"], r["bmax"], r["ppthr"])
    ccc = np.linspace(0.001, 0.999, 1000)
    ax  = plt.gca()
    ax.hist(z_j, bins=300, density=True, color="silver", edgecolor="grey",
            range=(0, 1), label="_hist")
    ax.add_patch(mpatches.Rectangle(
        (0, 0), ppthr, yLim, color="#FF7F50", alpha=0.20, zorder=3))
    ax.plot(ccc, p0 * betadist.pdf(ccc, etahat, 0.5),
            color="green", lw=2, ls="--", label="Null")
    ax.plot(ccc, (1 - p0) * betadist.pdf(ccc / bmax, ahat, bhat) / bmax,
            color="red", lw=1, label="Non-null")
    ax.plot(ccc,
            p0 * betadist.pdf(ccc, etahat, 0.5)
            + (1 - p0) * betadist.pdf(ccc / bmax, ahat, bhat) / bmax,
            color="blue", lw=2, label="Mixture")
    ax.set_ylim(0, yLim)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$z_j = \sin^2(\theta)$")
    ax.legend(fontsize=7)


def _plot_roc_panel(betaMixObj):
    """Draw the model-based ROC on the current axes; return ROC summary dict."""
    r  = betaMixObj
    tau = np.linspace(0, 1, 2001)
    fpr = betadist.cdf(tau, r["etahat"], 0.5)
    tpr = betadist.cdf(np.minimum(tau, r["bmax"]) / r["bmax"], r["ahat"], r["bhat"])
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    j_idx = int(np.argmax(tpr - fpr))
    tau_y, fpr_y, tpr_y = tau[j_idx], fpr[j_idx], tpr[j_idx]
    fpr_c = betadist.cdf(r["ppthr"], r["etahat"], 0.5)
    tpr_c = betadist.cdf(min(r["ppthr"], r["bmax"]) / r["bmax"], r["ahat"], r["bhat"])
    ax = plt.gca()
    ax.plot(fpr, tpr, "steelblue", lw=2)
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    ax.axvline(betadist.cdf(r["bmax"], r["etahat"], 0.5), ls=":", color="0.7", lw=1)
    ax.scatter([fpr_c], [tpr_c], c="orange", zorder=5, s=60)
    ax.scatter([fpr_y], [tpr_y], c="tomato", zorder=5, s=60)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC  (AUC = {auc:.4f})")
    ax.legend([
        f"Current  τ={r['ppthr']:.3g}",
        f"Youden   τ={tau_y:.3g}",
    ], handles=[
        mpatches.Patch(color="orange"), mpatches.Patch(color="tomato")
    ], fontsize=7, loc="lower right")
    return dict(tau=tau, fpr=fpr, tpr=tpr, auc=auc,
                tau_youden=tau_y, tau_current=r["ppthr"])


def plotFittedBetaMix(betaMixObj, yLim=5, file=""):
    """Plot the z_j histogram with the fitted mixture overlay and significance region.

    betaMixObj -- dict returned by betaMix().
    yLim       -- y-axis maximum (default 5).
    file       -- output filename; "" means show on screen (default).
    """
    plt.figure(figsize=(8, 5))
    _plot_fitted_panel(betaMixObj, yLim)
    plt.title("Fitted beta mixture")
    if file == "":
        plt.show()
    else:
        plt.savefig(file)


def plotROC(betaMixObj, file=""):
    """Plot the model-based ROC curve.

    betaMixObj -- dict returned by betaMix().
    file       -- output filename; "" means show on screen (default).

    Returns a dict: tau, fpr, tpr, auc, tau_youden, tau_current.
    """
    r = betaMixObj
    plt.figure(figsize=(6, 6))
    roc = _plot_roc_panel(betaMixObj)
    tau_y = roc["tau_youden"]
    fpr_y = betadist.cdf(tau_y, r["etahat"], 0.5)
    tpr_y = betadist.cdf(min(tau_y, r["bmax"]) / r["bmax"], r["ahat"], r["bhat"])
    fpr_c = betadist.cdf(r["ppthr"], r["etahat"], 0.5)
    tpr_c = betadist.cdf(min(r["ppthr"], r["bmax"]) / r["bmax"], r["ahat"], r["bhat"])
    print("=== plotROC: Model-based ROC ===")
    print(f"  AUC = {roc['auc']:.4f}")
    print(f"  Current  τ = {r['ppthr']:.4g}  FPR = {fpr_c:.4f}  TPR = {tpr_c:.4f}  edges = {r['edges']}")
    print(f"  Youden   τ = {tau_y:.4g}  FPR = {fpr_y:.4f}  TPR = {tpr_y:.4f}")
    print("  To apply: getAdjMat(res, ppthr=roc['tau_youden'])")
    print("================================")
    if file == "":
        plt.show()
    else:
        plt.savefig(file)
    return roc


def plotNonNullDensity(betaMixObj, file=""):
    """Plot the kernel density of non-null z_j with fitted Beta overlay.

    betaMixObj -- dict returned by betaMix().
    file       -- output filename; "" means show on screen (default).

    Returns a dict: n_modes, peaks_x.
    """
    r = betaMixObj
    z_j, m0, ahat, bhat, bmax = (r["z_j"], r["m0"], r["ahat"], r["bhat"], r["bmax"])
    nonnull_idx = np.where(m0 < 0.5)[0]
    n_nonnull   = len(nonnull_idx)

    plt.figure(figsize=(7, 5))
    if n_nonnull < 10:
        plt.text(0.5, 0.5, "Insufficient non-null\nobservations",
                 ha="center", va="center")
        plt.title("Non-null density (no data)")
        if file == "":
            plt.show()
        else:
            plt.savefig(file)
        return dict(n_modes=np.nan, peaks_x=np.array([]))

    dens = _kernel_density(z_j[nonnull_idx])
    dy   = dens["y"]
    nd   = len(dy)
    peak_thr = 0.10 * dy.max()
    is_peak  = np.concatenate([[False],
                                (dy[1:-1] > dy[:-2]) & (dy[1:-1] > dy[2:]),
                                [False]])
    peaks_above = np.where(is_peak & (dy >= peak_thr))[0]
    n_modes = len(peaks_above)

    xgrid     = np.linspace(1e-4, bmax - 1e-4, 512)
    beta_dens = betadist.pdf(xgrid / bmax, ahat, bhat) / bmax

    plt.plot(dens["x"], dens["y"], color="steelblue", lw=2, label="Kernel density")
    plt.plot(xgrid, beta_dens, color="firebrick", lw=2, ls="--",
             label=f"Beta({ahat:.2f}, {bhat:.2f})/bmax")
    if len(peaks_above) > 1:
        for px in dens["x"][peaks_above]:
            plt.axvline(px, color="orange", ls=":", lw=1.5)
    plt.title(f"Non-null density  ({n_modes} mode{'s' if n_modes != 1 else ''})")
    plt.xlabel(r"$z_j$")
    plt.ylabel("Density")
    plt.legend(fontsize=8)

    if file == "":
        plt.show()
    else:
        plt.savefig(file)

    return dict(n_modes=n_modes, peaks_x=dens["x"][peaks_above])


def plotBitmapCC(AdjMat, clusterInfo=None, orderByCluster=False,
                 showMinDegree=0, file=""):
    """Bitmap (image) plot of the adjacency matrix.

    AdjMat         -- adjacency matrix (P×P numpy array).
    clusterInfo    -- DataFrame from graphComponents(); if provided, cluster order is used.
    orderByCluster -- order nodes by cluster (default False; True if clusterInfo given).
    showMinDegree  -- hide nodes with degree < this value (default 0 = show all).
    file           -- output file; "" = show on screen (default).
    """
    A = np.abs(np.asarray(AdjMat, dtype=float))
    if clusterInfo is not None:
        orderByCluster = True
    if orderByCluster:
        if clusterInfo is None:
            clusterInfo = graphComponents(A)
        order = np.argsort(
            clusterInfo["clustNo"].values * 1e9 + clusterInfo["distCenter"].values
        )
        A = A[np.ix_(order, order)]
    show = np.where(A.sum(axis=0) >= showMinDegree)[0]
    A    = A[np.ix_(show, show)]
    plt.figure(figsize=(7, 7))
    plt.imshow(A, interpolation="none", cmap="Greys")
    if file == "":
        plt.show()
    else:
        plt.savefig(file)


def plotDegCC(betaMixObj, clusterInfo=None, highlightNodes=None, file=""):
    """Plot node degree vs CC×degree.

    betaMixObj     -- dict returned by betaMix().
    clusterInfo    -- DataFrame from graphComponents(); computed on the fly if None.
    highlightNodes -- list of node indices to mark in red (default None).
    file           -- output file; "" = show on screen (default).
    """
    if clusterInfo is None:
        A = getAdjMat(betaMixObj)
        clusterInfo = graphComponents(A)
    deg = clusterInfo["degree"].values
    cc  = clusterInfo["cc"].values
    if len(deg) == 0 or deg.max() == 0:
        warnings.warn("plotDegCC: no edges in graph; nothing to plot")
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(deg, deg * cc, c="thistle", s=10)
    if highlightNodes is not None:
        hn = np.asarray(highlightNodes, dtype=int)
        plt.scatter(deg[hn], (deg * cc)[hn], c="red", s=30)
    x = np.linspace(0, deg.max(), 50)
    plt.plot(x, x, color="seagreen", lw=2)
    plt.xlabel("degree")
    plt.ylabel("CC×degree")
    plt.grid(True)
    if file == "":
        plt.show()
    else:
        plt.savefig(file)


def plotCluster(AdjMat, clustNo, clusterInfo=None, labels=False,
                nodecol="blue", labelsize=9, figtitle=None,
                edgecols=("grey", "grey"), file=""):
    """Plot a cluster network in a polar-coordinate layout.

    Node radius = distance from cluster centre.
    Node size   = proportional to degree.
    Node opacity = fraction of edges within the cluster.
    The centre node is marked with a black ring.

    AdjMat      -- signed or unsigned adjacency matrix (P×P numpy array).
    clustNo     -- which cluster to plot.
    clusterInfo -- DataFrame from graphComponents(); computed on the fly if None.
    labels      -- show node indices as text (default False).
    nodecol     -- node fill colour (default "blue").
    labelsize   -- font size for labels (default 9).
    figtitle    -- figure title (default None).
    edgecols    -- (pos_color, neg_color) tuple or single string (default grey).
    file        -- output file; "" = show on screen (default).
    """
    if clusterInfo is None:
        clusterInfo = graphComponents(np.abs(AdjMat), minCtr=2, wgt=0)

    if isinstance(edgecols, str):
        edgecols = (edgecols, edgecols)

    n_clusters = int(clusterInfo["clustNo"].max())
    if clustNo < 1 or clustNo > n_clusters:
        print(f"Invalid cluster number. Must be between 1 and {n_clusters}")
        return

    ids = np.where(clusterInfo["clustNo"].values == clustNo)[0]
    if len(ids) < 2:
        print(f"{len(ids)} node(s) in cluster {clustNo}; nothing to plot")
        return

    tmpA   = AdjMat[np.ix_(ids, ids)]
    ci_sub = clusterInfo.iloc[ids].reset_index(drop=True)
    deg    = ci_sub["degree"].values
    dc     = ci_sub["distCenter"].values
    inE    = ci_sub["intEdges"].values

    max_dc  = max(dc.max(), 0.1)
    max_deg = max(deg.max(), 1)
    rads = np.round(10.0 * dc / max_dc).astype(int)

    # Assign angular positions per radial ring
    thetas = np.zeros(len(ids))
    intvls = np.searchsorted(np.arange(1, 11), rads)
    for iv in np.unique(intvls):
        pts = np.where(intvls == iv)[0]
        thetas[pts] = (3 * iv * np.pi / max(intvls.max(), 0.01)
                       + np.linspace(0, 1.9 * np.pi, len(pts)))

    sizes   = np.maximum(0.3, deg / max_deg) * 300
    opacity = np.where(deg > 0, 0.25 + inE / deg, 0.25)
    opacity = opacity / max(opacity.max(), 1e-9)

    xs = rads * np.cos(thetas)
    ys = rads * np.sin(thetas)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    if figtitle:
        ax.set_title(figtitle)

    # Draw edges
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(tmpA[i, j]) == 1:
                col = edgecols[1] if tmpA[i, j] < 0 else edgecols[0]
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                        color=col, lw=0.5, zorder=1)

    # Draw nodes
    from matplotlib.colors import to_rgba
    base_rgba = np.array(to_rgba(nodecol))
    for i in range(n):
        rgba = base_rgba.copy()
        rgba[3] = opacity[i]
        ax.scatter(xs[i], ys[i], s=sizes[i], color=[rgba], zorder=2)
        if labels:
            ax.text(xs[i], ys[i], str(ids[i]), fontsize=labelsize,
                    ha="center", va="bottom", zorder=3)

    # Mark centre
    ctr = np.where(ci_sub["iscenter"].values == 1)[0]
    if len(ctr) > 0:
        c = ctr[0]
        ax.scatter(xs[c], ys[c], s=sizes[c], facecolors="none",
                   edgecolors="black", linewidths=2, zorder=4)

    if file == "":
        plt.show()
    else:
        plt.savefig(file)
