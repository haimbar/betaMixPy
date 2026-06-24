# Port plan: greedyBasis → R betaMix package

This document describes how to port `greedyBasis()` from betaMixPy to the R
package at `/Users/haim/Projects/betaMix`.

---

## Target file

Create a new file `R/greedyBasis.R`.  Export the function via roxygen2 `@export`.

---

## R function signature

```r
greedyBasis <- function(A, strategy = "min_degree", labels = NULL)
```

`A` — square numeric adjacency matrix (P×P).  
`strategy` — `"min_degree"` (default) or `"max_degree"`.  
`labels` — optional character vector of length P; used as row names of the
returned data frame.

Returns a `data.frame` with one row per node and columns:
`node`, `isBasis`, `basisOrder`, `degree`, `cc`, `nCovered`,
`nCommunities`, `communities` (stored as a list-column).

---

## Python → R translation

| Python | R equivalent |
|---|---|
| `np.zeros(P, dtype=bool)` | `logical(P)` |
| `np.where(cond)[0]` | `which(cond) - 1L` (0-based → 1-based) |
| `adj_bool.sum(axis=1)` | `rowSums(adj_bool)` |
| `np.argmin(deg[active_idx])` | `which.min(deg[active])` |
| `np.ix_(remain_idx, deact_idx)` | standard R matrix indexing: `A[remain, deact]` |
| `rowSums(adj_bool[remain, deact])` | same |
| `np.where(adj_bool[b])[0]` | `which(adj_bool[b, ]) - 1L` |
| `clusteringCoef(A)` | `clusteringCoef(A)` (already in `R/clusteringCoef.R`) |
| list-column `communities` | use `I(lapply(...))` when constructing the data frame |

---

## Phase 1 — greedy MIS

```r
# Initialise
deg     <- rowSums(adj_bool)             # current-subgraph degrees (numeric)
active  <- rep(TRUE, P)
is_basis   <- logical(P)
basis_ord  <- integer(P)
pick_fn <- if (strategy == "min_degree") which.min else which.max
order_cnt  <- 0L

while (any(active)) {
  active_idx <- which(active)            # 1-based
  local_pick <- pick_fn(deg[active_idx]) # index within active_idx
  pick       <- active_idx[local_pick]

  order_cnt <- order_cnt + 1L
  is_basis[pick]  <- TRUE
  basis_ord[pick] <- order_cnt

  # Active neighbours of pick
  nbrs_active <- which(adj_bool[pick, ] & active)
  deact        <- c(pick, nbrs_active)

  # Update degrees of nodes that will remain active
  remain <- which(active & !(seq_len(P) %in% deact))
  if (length(remain) > 0 && length(deact) > 0) {
    loss <- rowSums(adj_bool[remain, deact, drop = FALSE])
    deg[remain] <- deg[remain] - loss
  }

  active[deact] <- FALSE
}
basis_nodes <- which(is_basis)           # 1-based
```

---

## Phase 2 — overlapping community assignment

```r
communities <- vector("list", P)
for (i in seq_len(P)) communities[[i]] <- integer(0)
n_covered <- integer(P)

for (b in basis_nodes) {
  all_nbrs    <- which(adj_bool[b, ])    # all neighbours in original A
  n_covered[b] <- 1L + length(all_nbrs)
  communities[[b]] <- c(communities[[b]], b)
  for (m in all_nbrs) {
    communities[[m]] <- c(communities[[m]], b)
  }
}

n_communities <- lengths(communities)
```

---

## Return value

```r
df <- data.frame(
  node         = seq_len(P),
  isBasis      = is_basis,
  basisOrder   = basis_ord,
  degree       = rowSums(adj_bool),
  cc           = clusteringCoef(A),
  nCovered     = n_covered,
  nCommunities = n_communities,
  row.names    = if (!is.null(labels)) labels else NULL,
  stringsAsFactors = FALSE
)
df$communities <- I(communities)   # list-column
df
```

---

## roxygen2 skeleton

```r
#' Find a maximal independent set and derive overlapping communities
#'
#' @param A Square numeric adjacency matrix (P×P).
#' @param strategy \code{"min_degree"} (default) or \code{"max_degree"}.
#' @param labels Optional character vector of length P used as row names.
#'
#' @return A \code{data.frame} with one row per node and columns
#'   \code{node}, \code{isBasis}, \code{basisOrder}, \code{degree},
#'   \code{cc}, \code{nCovered}, \code{nCommunities}, \code{communities}
#'   (list-column of integer vectors).
#'
#' @details
#' Guarantees: (1) no two basis nodes share an edge (independence);
#' (2) every non-basis node is adjacent to at least one basis node
#' (maximality).  A non-basis node with \code{nCommunities > 1} genuinely
#' bridges multiple communities.
#'
#' @examples
#' data(DrySeeds)
#' res  <- betaMix(DrySeeds, ppr = 0.01, ind = TRUE, msg = FALSE)
#' adjM <- getAdjMat(res)
#' bf   <- greedyBasis(adjM, strategy = "min_degree",
#'                     labels = colnames(DrySeeds))
#' summarizeBasis(bf)
#'
#' @export
greedyBasis <- function(A, strategy = "min_degree", labels = NULL) { ... }
```

---

## NAMESPACE / DESCRIPTION changes

`NAMESPACE`:
```
export(greedyBasis)
export(summarizeBasis)
```

`DESCRIPTION` (no new package dependencies — uses only base R).

---

## Companion function: `summarizeBasis`

Port `summarizeBasis()` similarly.  R equivalent of the Python:
```r
summarizeBasis <- function(basisInfo) {
  bi          <- basisInfo
  P           <- nrow(bi)
  n_basis     <- sum(bi$isBasis)
  n_isolated  <- sum(bi$degree == 0)
  non_basis   <- bi[!bi$isBasis, ]
  mean_cov    <- if (n_basis > 0) mean(bi$nCovered[bi$isBasis]) else 0
  max_cov     <- max(bi$nCovered)
  ovlp        <- non_basis$nCommunities
  n_shared    <- sum(ovlp > 1)

  cat(sprintf("Nodes: %d   Edges: %d   Isolated: %d\n",
              P, sum(bi$degree) %/% 2L, n_isolated))
  cat(sprintf("Basis size (MIS): %d  (%.1f%% of nodes)\n",
              n_basis, 100 * n_basis / P))
  cat(sprintf("Mean community size: %.1f   Max: %d\n", mean_cov, max_cov))
  cat(sprintf("Non-basis nodes in >1 community: %d / %d  (mean %.2f, max %d)\n",
              n_shared, nrow(non_basis),
              if (length(ovlp) > 0) mean(ovlp) else 0,
              if (length(ovlp) > 0) max(ovlp)  else 0))

  invisible(list(
    n_basis            = n_basis,
    basis_fraction     = n_basis / P,
    mean_community_size = mean_cov,
    max_community_size = max_cov,
    n_shared_nodes     = n_shared,
    n_isolated         = n_isolated
  ))
}
```

---

## Testing in R

```r
# After adding the functions:
library(betaMix)
data(DrySeeds)
res  <- betaMix(DrySeeds, ppr = 0.01, ind = TRUE, msg = FALSE)
adjM <- getAdjMat(res)
bf   <- greedyBasis(adjM, strategy = "min_degree")
summarizeBasis(bf)

# Independence check:
basis_idx <- which(bf$isBasis)
stopifnot(all(adjM[basis_idx, basis_idx] == 0))

# Maximality check:
non_basis <- which(!bf$isBasis)
stopifnot(all(sapply(non_basis, function(i) length(bf$communities[[i]]) >= 1)))
```

Expected output (matching Python):
- Basis size: 29 (42.6 % of nodes) for `strategy = "min_degree"`
- No two basis nodes adjacent
- All non-basis nodes have at least one community
