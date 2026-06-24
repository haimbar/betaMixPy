"""
Test suite for betaMixPy.

Run with:
    cd /Users/haim/Projects/betaMixPy
    python -m pytest tests/ -v

Reference values (DrySeeds, ind=True, ppr=0.01) verified against R betaMix 0.3.3.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import betaMixPy as bx

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dry_seeds():
    return pd.read_csv(os.path.join(DATA_DIR, "DrySeeds.csv"), index_col=0)


@pytest.fixture(scope="module")
def dry_res(dry_seeds):
    return bx.betaMix(dry_seeds, ppr=0.01, ind=True, msg=False)


@pytest.fixture(scope="module")
def adj_mat(dry_res):
    return bx.getAdjMat(dry_res)


@pytest.fixture
def path_adj():
    """5-node path graph: 0–1–2–3–4 (no triangles)."""
    A = np.zeros((5, 5))
    for i in range(4):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


@pytest.fixture
def triangle_adj():
    """Complete 3-node graph: all clustering coefficients = 1."""
    A = np.array([[0., 1., 1.],
                  [1., 0., 1.],
                  [1., 1., 0.]])
    return A


@pytest.fixture
def tiny_M():
    """10×4 synthetic matrix with columns 0 and 1 strongly correlated."""
    rng = np.random.default_rng(42)
    M = rng.standard_normal((10, 4))
    M[:, 1] = M[:, 0] + 0.05 * rng.standard_normal(10)
    return M


# ---------------------------------------------------------------------------
# betaMix
# ---------------------------------------------------------------------------

class TestBetaMix:

    def test_returns_dict(self, dry_res):
        assert isinstance(dry_res, dict)

    def test_required_keys(self, dry_res):
        required = {"angleMat", "z_j", "m0", "p0", "ahat", "bhat", "etahat",
                    "ppthr", "nodes", "edges", "bmax", "N", "cnt", "method"}
        assert required.issubset(set(dry_res.keys()))

    def test_reference_p0(self, dry_res):
        assert abs(dry_res["p0"] - 0.4751) < 0.01, f"p0={dry_res['p0']:.4f}"

    def test_reference_ppthr(self, dry_res):
        assert abs(dry_res["ppthr"] - 0.7767) < 0.01, f"ppthr={dry_res['ppthr']:.4f}"

    def test_reference_edges(self, dry_res):
        assert dry_res["edges"] == 203, f"edges={dry_res['edges']}"

    def test_shape(self, dry_res, dry_seeds):
        P = dry_seeds.shape[1]
        assert dry_res["angleMat"].shape == (P, P)
        assert dry_res["nodes"] == P
        assert dry_res["N"] == dry_seeds.shape[0]

    def test_p0_in_unit_interval(self, dry_res):
        assert 0 < dry_res["p0"] < 1

    def test_m0_in_unit_interval(self, dry_res):
        m0 = dry_res["m0"]
        assert np.all(m0 >= 0) and np.all(m0 <= 1)

    def test_spearman_method(self, dry_seeds):
        res = bx.betaMix(dry_seeds, method="spearman", ppr=0.01, ind=True, msg=False)
        assert res["method"] == "spearman"
        assert res["edges"] >= 0

    def test_pearson_method_label(self, dry_res):
        assert dry_res["method"] == "pearson"

    def test_ind_false(self, dry_seeds):
        res = bx.betaMix(dry_seeds, ind=False, ppr=0.01, msg=False)
        assert res["etahat"] <= (dry_seeds.shape[0] - 1) / 2.0

    def test_reproducibility(self, dry_seeds):
        r1 = bx.betaMix(dry_seeds, ppr=0.01, ind=True, msg=False, seed=7777)
        r2 = bx.betaMix(dry_seeds, ppr=0.01, ind=True, msg=False, seed=7777)
        assert r1["edges"] == r2["edges"]
        assert abs(r1["p0"] - r2["p0"]) < 1e-12

    def test_tiny_smoke(self, tiny_M):
        res = bx.betaMix(tiny_M, msg=False)
        assert isinstance(res, dict)
        assert res["nodes"] == tiny_M.shape[1]

    def test_invalid_bmax(self, tiny_M):
        with pytest.raises(ValueError, match="bmax"):
            bx.betaMix(tiny_M, bmax=1.5, msg=False)

    def test_1d_raises(self):
        with pytest.raises((ValueError, Exception)):
            bx.betaMix(np.ones(10), msg=False)

    def test_single_column_raises(self):
        with pytest.raises((ValueError, Exception)):
            bx.betaMix(np.ones((10, 1)), msg=False)


# ---------------------------------------------------------------------------
# getAdjMat
# ---------------------------------------------------------------------------

class TestGetAdjMat:

    def test_shape(self, dry_res, dry_seeds):
        A = bx.getAdjMat(dry_res)
        assert A.shape == (dry_seeds.shape[1], dry_seeds.shape[1])

    def test_diagonal_zero(self, adj_mat):
        assert np.all(np.diag(adj_mat) == 0)

    def test_symmetric(self, adj_mat):
        assert np.allclose(adj_mat, adj_mat.T)

    def test_binary_values(self, adj_mat):
        assert set(np.unique(adj_mat)).issubset({0.0, 1.0})

    def test_signed_values(self, dry_res):
        A = bx.getAdjMat(dry_res, signed=True)
        assert set(np.unique(A)).issubset({-1.0, 0.0, 1.0})

    def test_signed_diagonal_zero(self, dry_res):
        A = bx.getAdjMat(dry_res, signed=True)
        assert np.all(np.diag(A) == 0)

    def test_edge_count_matches_res(self, dry_res):
        A = bx.getAdjMat(dry_res)
        assert int(A.sum() / 2) == dry_res["edges"]

    def test_node_submatrix(self, dry_res):
        A = bx.getAdjMat(dry_res, nodes=[0, 1, 2])
        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]

    def test_invalid_nodes(self, dry_res):
        with pytest.raises(ValueError, match="nodes"):
            bx.getAdjMat(dry_res, nodes=[9999])


# ---------------------------------------------------------------------------
# clusteringCoef
# ---------------------------------------------------------------------------

class TestClusteringCoef:

    def test_complete_graph(self, triangle_adj):
        cc = bx.clusteringCoef(triangle_adj)
        assert np.allclose(cc, 1.0)

    def test_path_graph(self, path_adj):
        cc = bx.clusteringCoef(path_adj)
        assert np.all(cc == 0.0)

    def test_isolated_nodes(self):
        cc = bx.clusteringCoef(np.zeros((5, 5)))
        assert np.all(cc == 0.0)

    def test_range_on_real_data(self, adj_mat):
        cc = bx.clusteringCoef(adj_mat)
        assert np.all(cc >= 0) and np.all(cc <= 1)

    def test_symmetry_invariant(self, adj_mat):
        cc1 = bx.clusteringCoef(adj_mat)
        cc2 = bx.clusteringCoef(adj_mat.T)
        assert np.allclose(cc1, cc2)


# ---------------------------------------------------------------------------
# graphComponents
# ---------------------------------------------------------------------------

class TestGraphComponents:

    def test_required_columns(self, adj_mat):
        df = bx.graphComponents(adj_mat, minCtr=2)
        required = {"labels", "degree", "cc", "ctr", "clustNo",
                    "iscenter", "intEdges", "extEdges", "distCenter"}
        assert required.issubset(set(df.columns))

    def test_row_count(self, adj_mat):
        df = bx.graphComponents(adj_mat)
        assert len(df) == adj_mat.shape[0]

    def test_clustNo_nonnegative(self, adj_mat):
        df = bx.graphComponents(adj_mat)
        assert (df["clustNo"] >= 0).all()

    def test_degree_matches_adj(self, adj_mat):
        df = bx.graphComponents(adj_mat)
        expected_deg = adj_mat.sum(axis=1)
        assert np.allclose(df["degree"].values, expected_deg)

    def test_no_edges(self):
        df = bx.graphComponents(np.zeros((5, 5)))
        assert (df["clustNo"] == 0).all()


# ---------------------------------------------------------------------------
# sphericalCaps
# ---------------------------------------------------------------------------

class TestSphericalCaps:

    def test_columns(self, adj_mat):
        df = bx.sphericalCaps(adj_mat)
        assert df is not None
        assert set(df.columns) == {"node", "capNum", "isCtr", "deg", "cc"}

    def test_no_edges_returns_none(self):
        assert bx.sphericalCaps(np.zeros((5, 5))) is None

    def test_cap_numbers_positive(self, adj_mat):
        df = bx.sphericalCaps(adj_mat)
        assert (df["capNum"] >= 1).all()

    def test_each_cap_has_one_centre(self, adj_mat):
        df = bx.sphericalCaps(adj_mat)
        for cap in df["capNum"].unique():
            assert df.loc[df["capNum"] == cap, "isCtr"].sum() == 1


# ---------------------------------------------------------------------------
# greedyBasis
# ---------------------------------------------------------------------------

class TestGreedyBasis:

    @pytest.fixture(scope="class")
    def basis_min(self, adj_mat):
        return bx.greedyBasis(adj_mat, strategy="min_degree")

    @pytest.fixture(scope="class")
    def basis_max(self, adj_mat):
        return bx.greedyBasis(adj_mat, strategy="max_degree")

    def test_required_columns(self, basis_min):
        required = {"node", "isBasis", "basisOrder", "degree", "cc",
                    "nCovered", "nCommunities", "communities"}
        assert required.issubset(set(basis_min.columns))

    def test_row_count(self, basis_min, adj_mat):
        assert len(basis_min) == adj_mat.shape[0]

    def _check_independence(self, A, basis_df):
        adj_bool = A > 0
        basis_idx = np.where(basis_df["isBasis"].values)[0]
        for i in basis_idx:
            for j in basis_idx:
                if i != j:
                    assert not adj_bool[i, j], \
                        f"Independence violated: basis nodes {i} and {j} are adjacent"

    def _check_maximality(self, basis_df):
        non_basis = basis_df[~basis_df["isBasis"]]
        for _, row in non_basis.iterrows():
            assert len(row["communities"]) >= 1, \
                f"Maximality violated: non-basis node {row['node']} has no community"

    def test_independence_min(self, adj_mat, basis_min):
        self._check_independence(adj_mat, basis_min)

    def test_independence_max(self, adj_mat, basis_max):
        self._check_independence(adj_mat, basis_max)

    def test_maximality_min(self, basis_min):
        self._check_maximality(basis_min)

    def test_maximality_max(self, basis_max):
        self._check_maximality(basis_max)

    def test_basis_order_positive(self, basis_min):
        basis_orders = basis_min.loc[basis_min["isBasis"], "basisOrder"]
        assert (basis_orders > 0).all()

    def test_non_basis_order_zero(self, basis_min):
        non_basis_orders = basis_min.loc[~basis_min["isBasis"], "basisOrder"]
        assert (non_basis_orders == 0).all()

    def test_n_communities_communities_consistent(self, basis_min):
        for _, row in basis_min.iterrows():
            assert row["nCommunities"] == len(row["communities"])

    def test_labels(self, adj_mat):
        labels = [f"var{i}" for i in range(adj_mat.shape[0])]
        bf = bx.greedyBasis(adj_mat, labels=labels)
        assert list(bf.index) == labels

    def test_invalid_strategy(self, adj_mat):
        with pytest.raises(ValueError, match="strategy"):
            bx.greedyBasis(adj_mat, strategy="unknown")

    def test_isolated_nodes_in_basis(self):
        """Isolated nodes have degree 0 — they must be selected as basis nodes."""
        A = np.zeros((5, 5))
        A[0, 1] = A[1, 0] = 1.0   # nodes 2, 3, 4 are isolated
        bf = bx.greedyBasis(A)
        isolated = bf.loc[bf["node"].isin([2, 3, 4]), "isBasis"]
        assert isolated.all(), "Isolated nodes must be in the basis"

    def test_complete_graph_basis_size_1(self):
        """In a complete graph every node is adjacent to all others; MIS size = 1."""
        n = 6
        A = np.ones((n, n)) - np.eye(n)
        bf = bx.greedyBasis(A, strategy="min_degree")
        assert bf["isBasis"].sum() == 1

    def test_n_covered_basis_only(self, basis_min):
        """nCovered must be > 0 for basis nodes and 0 for non-basis."""
        assert (basis_min.loc[basis_min["isBasis"], "nCovered"] > 0).all()
        assert (basis_min.loc[~basis_min["isBasis"], "nCovered"] == 0).all()

    def test_no_edges_all_basis(self):
        """With no edges, every node is isolated, so all nodes are in the basis."""
        A = np.zeros((5, 5))
        bf = bx.greedyBasis(A)
        assert bf["isBasis"].all()


# ---------------------------------------------------------------------------
# summarizeBasis
# ---------------------------------------------------------------------------

class TestSummarizeBasis:

    def test_returns_dict(self, adj_mat, capsys):
        bf = bx.greedyBasis(adj_mat)
        result = bx.summarizeBasis(bf)
        assert isinstance(result, dict)

    def test_required_keys(self, adj_mat, capsys):
        bf = bx.greedyBasis(adj_mat)
        result = bx.summarizeBasis(bf)
        required = {"n_basis", "basis_fraction", "mean_community_size",
                    "max_community_size", "n_shared_nodes",
                    "mean_n_communities", "max_n_communities", "n_isolated"}
        assert required.issubset(set(result.keys()))

    def test_basis_fraction_range(self, adj_mat, capsys):
        bf = bx.greedyBasis(adj_mat)
        result = bx.summarizeBasis(bf)
        assert 0 < result["basis_fraction"] <= 1


# ---------------------------------------------------------------------------
# shortestPathDistance
# ---------------------------------------------------------------------------

class TestShortestPathDistance:

    def test_zero_steps_is_identity(self, adj_mat):
        spd = bx.shortestPathDistance(adj_mat, numSteps=0)
        assert np.allclose(spd, adj_mat)

    def test_two_hop_in_path(self, path_adj):
        """Path 0–1–2–3–4, numSteps=1: 0↔2 reachable in 2 hops."""
        spd = bx.shortestPathDistance(path_adj, numSteps=1)
        assert spd[0, 2] == 2.0
        assert spd[0, 3] == 0.0   # 3-hop path, unreachable in 1 extra step

    def test_direct_edges_unchanged(self, path_adj):
        spd = bx.shortestPathDistance(path_adj, numSteps=1)
        direct = np.where(path_adj > 0)
        assert np.all(spd[direct] == 1.0)

    def test_invalid_steps_negative(self, adj_mat):
        with pytest.raises(ValueError, match="numSteps"):
            bx.shortestPathDistance(adj_mat, numSteps=-1)

    def test_invalid_steps_float(self, adj_mat):
        with pytest.raises(ValueError):
            bx.shortestPathDistance(adj_mat, numSteps=1.5)


# ---------------------------------------------------------------------------
# collapsedGraph
# ---------------------------------------------------------------------------

class TestCollapsedGraph:

    def test_symmetric(self, adj_mat):
        ci = bx.graphComponents(adj_mat, minCtr=2)
        coll = bx.collapsedGraph(adj_mat, ci)
        assert np.allclose(coll, coll.T)

    def test_dimension(self, adj_mat):
        ci = bx.graphComponents(adj_mat, minCtr=2)
        n_clust = int(ci["clustNo"].max())
        n_unclust = int((ci["clustNo"] == 0).sum())
        coll = bx.collapsedGraph(adj_mat, ci)
        assert coll.shape == (n_unclust + n_clust, n_unclust + n_clust)

    def test_diagonal_zero(self, adj_mat):
        ci = bx.graphComponents(adj_mat, minCtr=2)
        coll = bx.collapsedGraph(adj_mat, ci)
        assert np.all(np.diag(coll) == 0)


# ---------------------------------------------------------------------------
# summarizeClusters
# ---------------------------------------------------------------------------

class TestSummarizeClusters:

    def test_returns_dataframe_or_none(self, adj_mat, capsys):
        ci = bx.graphComponents(adj_mat, minCtr=2)
        result = bx.summarizeClusters(ci)
        assert result is None or isinstance(result, pd.DataFrame)

    def test_columns(self, adj_mat, capsys):
        ci = bx.graphComponents(adj_mat, minCtr=2)
        result = bx.summarizeClusters(ci)
        if result is not None:
            assert "Cluster" in result.columns
            assert "Nodes" in result.columns

    def test_no_clusters(self, capsys):
        A = np.zeros((5, 5))
        ci = bx.graphComponents(A)
        result = bx.summarizeClusters(ci)
        assert result is None


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------

def test_module_imports_cleanly():
    import importlib
    mod = importlib.import_module("betaMixPy")
    public = [name for name in dir(mod) if not name.startswith("_")]
    expected = ["betaMix", "getAdjMat", "clusteringCoef", "graphComponents",
                "sphericalCaps", "greedyBasis", "summarizeBasis",
                "summarizeClusters", "collapsedGraph", "shortestPathDistance",
                "shortSummary", "assessFit", "plotFittedBetaMix", "plotROC",
                "plotNonNullDensity", "plotBitmapCC", "plotDegCC", "plotCluster"]
    for name in expected:
        assert name in public, f"Expected public function '{name}' not found in module"
