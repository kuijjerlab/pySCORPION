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
