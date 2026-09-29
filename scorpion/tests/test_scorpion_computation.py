"""
Tests for SCORPION core algorithm computation.
"""

import pytest
import numpy as np
import pandas as pd
from scorpion import scorpion
from scorpion.preprocessing import log_normalize_data
from scipy.sparse import csr_matrix


class TestScorpionComputation:
    """Test scorpion() algorithm."""
    
    def test_scorpion_returns_dict(self, toy_expression, toy_motifs):
        """scorpion() should return a dictionary."""
        result = scorpion(
            toy_expression,
            toy_motifs,
            show_progress=False
        )
        assert isinstance(result, dict)
    
    def test_scorpion_output_networks(self, toy_expression, toy_motifs):
        """Default output should include regNet."""
        result = scorpion(
            toy_expression,
            toy_motifs,
            out_net="regNet",
            show_progress=False
        )
        assert "regNet" in result
        assert isinstance(result["regNet"], np.ndarray)
    
    def test_scorpion_multiple_output_networks(self, toy_expression, toy_motifs):
        """Should return multiple networks when requested."""
        result = scorpion(
            toy_expression,
            toy_motifs,
            out_net=["regNet", "coregNet", "coopNet"],
            show_progress=False
        )
        assert "regNet" in result
        assert "coregNet" in result
        assert "coopNet" in result
    
    def test_scorpion_metadata(self, toy_expression, toy_motifs):
        """Output should include metadata."""
        result = scorpion(
            toy_expression,
            toy_motifs,
            show_progress=False
        )
        
        # Metadata fields
        assert "numGenes" in result
        assert "numTFs" in result
        assert "numEdges" in result
        
        # Types
        assert isinstance(result["numGenes"], (int, np.integer))
        assert isinstance(result["numTFs"], (int, np.integer))
        assert isinstance(result["numEdges"], (int, np.integer))
    
    def test_scorpion_network_dimensions(self, toy_expression, toy_motifs):
        """Regulatory network should be (TFs × genes), matching R."""
        result = scorpion(
            toy_expression,
            toy_motifs,
            out_net=["regNet", "coregNet", "coopNet"],
            show_progress=False
        )
        
        n_tfs = result["numTFs"]
        n_genes = result["numGenes"]
        
        # regNet is (TFs x genes) matching R convention
        assert result["regNet"].shape[0] == n_tfs
        assert result["regNet"].shape[1] == n_genes
        assert result["coregNet"].shape[0] == n_genes
        assert result["coregNet"].shape[1] == n_genes
        assert result["coopNet"].shape[0] == n_tfs
        assert result["coopNet"].shape[1] == n_tfs
    
    def test_scorpion_with_ppi(self, toy_expression, toy_motifs, toy_ppi):
        """scorpion() should work with PPI prior."""
        result = scorpion(
            toy_expression,
            toy_motifs,
            ppi_net=toy_ppi,
            show_progress=False
        )
        
        assert "regNet" in result
        assert result["numTFs"] > 0
    
    def test_scorpion_sparse_input(self, toy_expression, toy_motifs):
        """scorpion() should handle sparse expression matrix."""
        sparse_expr = csr_matrix(toy_expression.values)
        
        result = scorpion(
            sparse_expr,
            toy_motifs,
            gene_names=toy_expression.index.values,
            show_progress=False
        )
        
        assert "regNet" in result
    
    def test_scorpion_dataframe_input(self, toy_expression, toy_motifs):
        """scorpion() should handle DataFrame input."""
        # toy_expression is already a DataFrame with gene names
        result = scorpion(
            toy_expression,
            toy_motifs,
            show_progress=False
        )
        
        assert "regNet" in result
    
    def test_scorpion_alpha_effect(self, toy_expression, toy_motifs):
        """Higher alpha should increase motif influence."""
        # alpha = 0 (correlation only)
        result_low_alpha = scorpion(
            toy_expression,
            toy_motifs,
            alpha_value=0.0,
            show_progress=False
        )
        
        # alpha = 1 (motif only)
        result_high_alpha = scorpion(
            toy_expression,
            toy_motifs,
            alpha_value=1.0,
            show_progress=False
        )
        
        # Networks should be different
        assert not np.allclose(
            result_low_alpha["regNet"],
            result_high_alpha["regNet"]
        )
    
    def test_scorpion_correlation_methods(self, toy_expression, toy_motifs):
        """Should support multiple correlation methods."""
        result_pearson = scorpion(
            toy_expression,
            toy_motifs,
            assoc_method="pearson",
            show_progress=False
        )
        
        result_spearman = scorpion(
            toy_expression,
            toy_motifs,
            assoc_method="spearman",
            show_progress=False
        )
        
        # Results should differ
        assert not np.allclose(
            result_pearson["regNet"],
            result_spearman["regNet"]
        )
    
    def test_scorpion_z_scaling_effects(self, toy_expression, toy_motifs):
        """z_scaling parameter should affect output scale."""
        result_z = scorpion(
            toy_expression,
            toy_motifs,
            z_scaling=True,
            show_progress=False
        )
        
        result_range = scorpion(
            toy_expression,
            toy_motifs,
            z_scaling=False,
            show_progress=False
        )
        
        # Both should have values, but distributions differ
        assert result_z["regNet"].std() > 0
        assert result_range["regNet"].std() > 0
    
    def test_scorpion_reproducibility(self, toy_expression, toy_motifs):
        """Same random seed should produce same results."""
        result1 = scorpion(
            toy_expression,
            toy_motifs,
            random_state=42,
            show_progress=False
        )
        
        result2 = scorpion(
            toy_expression,
            toy_motifs,
            random_state=42,
            show_progress=False
        )
        
        np.testing.assert_allclose(
            result1["regNet"],
            result2["regNet"],
            rtol=1e-5
        )
    
    def test_scorpion_different_seeds_different_results(self, toy_expression, toy_motifs):
        """Different random seeds should still produce valid results.
        
        Note: walktrap community detection is deterministic, so different
        seeds produce identical results when kNN is also deterministic.
        This test verifies that different seeds don't cause errors.
        """
        result1 = scorpion(
            toy_expression,
            toy_motifs,
            random_state=42,
            show_progress=False
        )
        
        result2 = scorpion(
            toy_expression,
            toy_motifs,
            random_state=123,
            show_progress=False
        )
        
        # Both should produce valid results
        assert "regNet" in result1
        assert "regNet" in result2
        assert np.all(np.isfinite(result1["regNet"]))
        assert np.all(np.isfinite(result2["regNet"]))
        assert result1["regNet"].shape == result2["regNet"].shape
    
    def test_scorpion_minimum_cells(self, toy_expression, toy_motifs):
        """scorpion() should work with minimum viable cell count."""
        # Create expression with just 10 cells
        small_expr = toy_expression.iloc[:, :10]
        
        result = scorpion(
            small_expr,
            toy_motifs,
            n_pc=5,  # Must be < number of cells
            show_progress=False
        )
        
        assert "regNet" in result
        assert result["numGenes"] > 0
    
    def test_scorpion_network_values(self, toy_expression, toy_motifs):
        """Network values should be finite."""
        result = scorpion(
            toy_expression,
            toy_motifs,
            out_net=["regNet", "coregNet", "coopNet"],
            show_progress=False
        )
        
        assert np.all(np.isfinite(result["regNet"]))
        assert np.all(np.isfinite(result["coregNet"]))
        assert np.all(np.isfinite(result["coopNet"]))


class TestScorpionRParity:
    """Behaviour added to match R's scorpion()."""

    def test_coexpression_only(self, toy_expression):
        """No priors: return the super-cell gene-gene correlation matrix."""
        result = scorpion(toy_expression, show_progress=False)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (100, 100)
        assert list(result.index) == list(toy_expression.index)
        np.testing.assert_allclose(result.values, result.values.T)
        np.testing.assert_allclose(np.diag(result.values), 1.0)

    def test_coexpression_spearman(self, toy_expression):
        pearson = scorpion(toy_expression, show_progress=False, random_state=0)
        spearman = scorpion(toy_expression, assoc_method="spearman",
                            show_progress=False, random_state=0)
        assert not np.allclose(pearson.values, spearman.values)

    def test_filter_expr(self, toy_expression, toy_motifs):
        """filter_expr drops all-zero genes before inference."""
        gex = toy_expression.copy()
        gene = toy_motifs["target"].iloc[0]
        gex.loc[gene] = 0.0
        result = scorpion(gex, toy_motifs, filter_expr=True, show_progress=False)
        assert gene not in list(result["geneNames"])
        kept = scorpion(toy_expression, toy_motifs, show_progress=False)
        assert gene in list(kept["geneNames"])

    def test_scale_by_present(self, toy_expression, toy_motifs):
        # Gene-specific dropout so co-presence differs between gene pairs
        rng = np.random.default_rng(0)
        gex = toy_expression.where(rng.random(toy_expression.shape) > 0.8, 0.0)
        base = scorpion(gex, toy_motifs, show_progress=False, random_state=0)
        scaled = scorpion(gex, toy_motifs, scale_by_present=True,
                          show_progress=False, random_state=0)
        assert not np.allclose(base["regNet"], scaled["regNet"])

    @pytest.mark.parametrize("method", ["within.gene", "by.gene", "within_gene", "by_gene"])
    def test_randomization(self, toy_expression, toy_motifs, method):
        base = scorpion(toy_expression, toy_motifs, show_progress=False, random_state=0)
        rand1 = scorpion(toy_expression, toy_motifs, randomization_method=method,
                         show_progress=False, random_state=0)
        rand2 = scorpion(toy_expression, toy_motifs, randomization_method=method,
                         show_progress=False, random_state=0)
        assert not np.allclose(base["regNet"], rand1["regNet"])
        np.testing.assert_array_equal(rand1["regNet"], rand2["regNet"])

    def test_randomization_none_string(self, toy_expression, toy_motifs):
        a = scorpion(toy_expression, toy_motifs, show_progress=False, random_state=0)
        b = scorpion(toy_expression, toy_motifs, randomization_method="None",
                     show_progress=False, random_state=0)
        np.testing.assert_array_equal(a["regNet"], b["regNet"])

    def test_invalid_randomization(self, toy_expression, toy_motifs):
        with pytest.raises(ValueError, match="randomization_method"):
            scorpion(toy_expression, toy_motifs, randomization_method="bogus",
                     show_progress=False)

    def test_gpu_falls_back_to_cpu(self, toy_expression, toy_motifs):
        with pytest.warns(UserWarning, match="Falling back to CPU"):
            gpu = scorpion(toy_expression, toy_motifs, computing_engine="gpu",
                           show_progress=False, random_state=0)
        cpu = scorpion(toy_expression, toy_motifs, show_progress=False, random_state=0)
        np.testing.assert_array_equal(gpu["regNet"], cpu["regNet"])

    def test_invalid_parameters(self, toy_expression, toy_motifs):
        with pytest.raises(ValueError, match="hamming_value"):
            scorpion(toy_expression, toy_motifs, hamming_value=-1, show_progress=False)
        with pytest.raises(ValueError, match="assoc_method"):
            scorpion(toy_expression, toy_motifs, assoc_method="kendall", show_progress=False)
        with pytest.raises(ValueError, match="computing_engine"):
            scorpion(toy_expression, toy_motifs, computing_engine="tpu", show_progress=False)
        with pytest.raises(ValueError, match="at least 10 genes"):
            scorpion(toy_expression.iloc[:9], toy_motifs, show_progress=False)

    def test_few_cells_warns(self, toy_expression, toy_motifs):
        with pytest.warns(UserWarning, match="fewer than 30 cells"):
            scorpion(toy_expression.iloc[:, :29], toy_motifs, gamma_value=3,
                     n_pc=5, show_progress=False)
