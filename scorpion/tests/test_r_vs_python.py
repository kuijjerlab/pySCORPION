"""
Tests comparing Python SCORPION against R SCORPION reference outputs.

These tests load the real scorpionTest data from the R package,
run the Python implementation, and compare against pre-generated
R reference outputs (created by scorpionR/generate_reference.R).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import mmread

from scorpion import scorpion, run_scorpion


DATA_DIR = Path(__file__).parent / "data"

# Skip everything if the reference data hasn't been generated
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "ref_scorpion_regNet.csv").exists(),
    reason="R reference data not generated (run scorpionR/generate_reference.R)",
)


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_data():
    """Load the real scorpionTest data exported from R."""
    # Expression matrix (Matrix Market → genes × cells)
    expr_sparse = mmread(DATA_DIR / "expression.mtx").tocsr()
    genes = np.loadtxt(DATA_DIR / "expression_genes.txt", dtype=str)
    cells = np.loadtxt(DATA_DIR / "expression_cells.txt", dtype=str)

    # Verify dims
    assert expr_sparse.shape == (len(genes), len(cells)), (
        f"Expression shape {expr_sparse.shape} vs "
        f"genes {len(genes)} × cells {len(cells)}"
    )

    gex = pd.DataFrame(
        expr_sparse.toarray(),
        index=genes,
        columns=cells,
    )

    # Motif priors
    motifs = pd.read_csv(DATA_DIR / "motif.csv")

    # PPI
    ppi = pd.read_csv(DATA_DIR / "ppi.csv")

    # Metadata
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")

    return {
        "gex": gex,
        "motifs": motifs,
        "ppi": ppi,
        "metadata": metadata,
    }


# ── scorpion() tests ──────────────────────────────────────────────────────────

class TestScorpionVsR:
    """Compare scorpion() output against R reference."""

    def test_scorpion_metadata_matches_r(self, real_data):
        """Network dimensions should match R exactly."""
        ref_meta = pd.read_csv(DATA_DIR / "ref_scorpion_meta.csv")

        # Run on normal-region cells (same as R script)
        normal_mask = real_data["metadata"]["region"] == "N"
        normal_cells = real_data["metadata"].loc[normal_mask, "cell_id"].values
        gex_normal = real_data["gex"][normal_cells]

        result = scorpion(
            gex_matrix=gex_normal,
            tf_motifs=real_data["motifs"],
            ppi_net=real_data["ppi"],
            alpha_value=0.1,
            gamma_value=10,
            n_pc=25,
            assoc_method="pearson",
            hamming_value=0.001,
            n_iter=np.inf,
            out_net="regNet",
            z_scaling=True,
            show_progress=False,
        )

        assert result["numGenes"] == int(ref_meta["numGenes"].iloc[0]), (
            f"numGenes: Python {result['numGenes']} vs R {ref_meta['numGenes'].iloc[0]}"
        )
        assert result["numTFs"] == int(ref_meta["numTFs"].iloc[0]), (
            f"numTFs: Python {result['numTFs']} vs R {ref_meta['numTFs'].iloc[0]}"
        )
        assert result["numEdges"] == int(ref_meta["numEdges"].iloc[0]), (
            f"numEdges: Python {result['numEdges']} vs R {ref_meta['numEdges'].iloc[0]}"
        )

    def test_scorpion_regnet_matches_r(self, real_data):
        """Regulatory network edge weights should match R within tolerance."""
        # Load R reference (edge list: tf, target, weight)
        ref_edges = pd.read_csv(DATA_DIR / "ref_scorpion_regNet.csv")

        # Run Python scorpion on normal-region cells
        normal_mask = real_data["metadata"]["region"] == "N"
        normal_cells = real_data["metadata"].loc[normal_mask, "cell_id"].values
        gex_normal = real_data["gex"][normal_cells]

        result = scorpion(
            gex_matrix=gex_normal,
            tf_motifs=real_data["motifs"],
            ppi_net=real_data["ppi"],
            alpha_value=0.1,
            gamma_value=10,
            n_pc=25,
            assoc_method="pearson",
            hamming_value=0.001,
            n_iter=np.inf,
            out_net="regNet",
            z_scaling=True,
            show_progress=False,
        )

        reg_net = result["regNet"]  # (numTFs × numGenes) matrix
        tf_names = result["tfNames"]
        gene_names = result["geneNames"]

        # Build Python edge list to match R format
        py_edges = []
        for i, tf in enumerate(tf_names):
            for j, gene in enumerate(gene_names):
                py_edges.append({
                    "tf": tf,
                    "target": gene,
                    "weight": reg_net[i, j],
                })
        py_df = pd.DataFrame(py_edges)

        # Merge on tf+target to align edges
        merged = ref_edges.merge(
            py_df, on=["tf", "target"], suffixes=("_r", "_py"),
        )

        # All R edges should be present in Python
        assert len(merged) == len(ref_edges), (
            f"Edge count mismatch: merged {len(merged)} vs R {len(ref_edges)}"
        )

        # Compare weights
        r_weights = merged["weight_r"].values
        py_weights = merged["weight_py"].values

        # Correlation should be very high
        corr = np.corrcoef(r_weights, py_weights)[0, 1]
        assert corr > 0.99, f"Correlation between R and Python weights: {corr:.6f}"

        # Max absolute difference
        max_diff = np.max(np.abs(r_weights - py_weights))
        mean_diff = np.mean(np.abs(r_weights - py_weights))
        print(f"\n  scorpion() regNet comparison:")
        print(f"    Correlation: {corr:.8f}")
        print(f"    Mean |diff|: {mean_diff:.6f}")
        print(f"    Max  |diff|: {max_diff:.6f}")

        # Allow small numerical differences from floating-point / kNN / PCA
        # Typical tolerance for these stochastic algorithms
        assert np.allclose(r_weights, py_weights, atol=0.05, rtol=0.05), (
            f"Weights differ too much: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        )


# ── runSCORPION() tests ───────────────────────────────────────────────────────

class TestRunScorpionVsR:
    """Compare run_scorpion() output against R reference."""

    def test_run_scorpion_shape_matches_r(self, real_data):
        """Output dimensions should match R."""
        ref_nets = pd.read_csv(DATA_DIR / "ref_runSCORPION_region.csv")

        py_nets = run_scorpion(
            gex_matrix=real_data["gex"],
            tf_motifs=real_data["motifs"],
            ppi_net=real_data["ppi"],
            cells_metadata=real_data["metadata"],
            group_by="region",
            normalize_data=True,
            min_cells=30,
            alpha_value=0.1,
            gamma_value=10,
            n_pc=25,
            assoc_method="pearson",
            hamming_value=0.001,
            n_iter=np.inf,
            out_net="regNet",
            z_scaling=True,
            show_progress=True,
        )

        # Same number of edges
        assert py_nets.shape[0] == ref_nets.shape[0], (
            f"Row count: Python {py_nets.shape[0]} vs R {ref_nets.shape[0]}"
        )

        # Same columns (tf, target, + region groups)
        assert set(py_nets.columns) == set(ref_nets.columns), (
            f"Columns differ:\n  Python: {sorted(py_nets.columns)}\n"
            f"  R: {sorted(ref_nets.columns)}"
        )

    def test_run_scorpion_weights_match_r(self, real_data):
        """Edge weights from run_scorpion() should match R within tolerance."""
        ref_nets = pd.read_csv(DATA_DIR / "ref_runSCORPION_region.csv")

        py_nets = run_scorpion(
            gex_matrix=real_data["gex"],
            tf_motifs=real_data["motifs"],
            ppi_net=real_data["ppi"],
            cells_metadata=real_data["metadata"],
            group_by="region",
            normalize_data=True,
            min_cells=30,
            alpha_value=0.1,
            gamma_value=10,
            n_pc=25,
            assoc_method="pearson",
            hamming_value=0.001,
            n_iter=np.inf,
            out_net="regNet",
            z_scaling=True,
            show_progress=False,
        )

        # Merge on tf + target
        merged = ref_nets.merge(
            py_nets, on=["tf", "target"], suffixes=("_r", "_py"),
        )

        assert len(merged) == len(ref_nets), (
            f"Merged edge count {len(merged)} vs R {len(ref_nets)}"
        )

        # Compare each region column
        region_cols = [c for c in ref_nets.columns if c not in ("tf", "target")]
        print(f"\n  run_scorpion() comparison by region:")

        for col in region_cols:
            r_vals = merged[f"{col}_r"].values
            py_vals = merged[f"{col}_py"].values

            corr = np.corrcoef(r_vals, py_vals)[0, 1]
            max_diff = np.max(np.abs(r_vals - py_vals))
            mean_diff = np.mean(np.abs(r_vals - py_vals))

            print(f"    {col}: corr={corr:.8f}  mean|diff|={mean_diff:.6f}  max|diff|={max_diff:.6f}")

            assert corr > 0.99, (
                f"Region {col}: correlation {corr:.6f} too low"
            )
            assert np.allclose(r_vals, py_vals, atol=0.05, rtol=0.05), (
                f"Region {col}: max_diff={max_diff:.6f}"
            )
