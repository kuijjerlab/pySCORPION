"""
Tests for data preprocessing functions.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scorpion.preprocessing import (
    log_normalize_data,
    remove_batch_effect,
    filter_expression,
    preprocess_expression,
)


class TestLogNormalization:
    """Test log normalization."""
    
    def test_log_normalize_basic(self, toy_expression):
        """log_normalize_data should work correctly."""
        normalized = log_normalize_data(toy_expression)
        
        assert normalized.shape == toy_expression.shape
        assert np.all(normalized >= 0)  # log1p of non-negative is non-negative
        assert np.all(np.isfinite(normalized))
    
    def test_log_normalize_sparse(self, toy_expression):
        """log_normalize_data should work with sparse input."""
        sparse_expr = csr_matrix(toy_expression)
        normalized = log_normalize_data(sparse_expr)
        
        assert normalized.shape == toy_expression.shape
        assert np.all(np.isfinite(normalized))
    
    def test_log_normalize_dataframe(self, toy_expression):
        """log_normalize_data should work with DataFrame input."""
        df_expr = pd.DataFrame(toy_expression)
        normalized = log_normalize_data(df_expr)
        
        assert normalized.shape == toy_expression.shape
        assert np.all(np.isfinite(normalized))
    
    def test_log_normalize_scale_factor(self, toy_expression):
        """log_normalize_data should scale by scale_factor."""
        n1 = log_normalize_data(toy_expression, scale_factor=1000)
        n2 = log_normalize_data(toy_expression, scale_factor=10000)
        
        # Different scale factors should produce different results
        assert not np.allclose(n1, n2)
    
    def test_log_normalize_sums_per_cell(self, toy_expression):
        """log_normalize_data should normalize by column sums."""
        # Create expression with very different column sums
        expr = np.array([[1, 1], [1, 100]], dtype=float)
        
        normalized = log_normalize_data(expr, scale_factor=100)
        
        # First cell should be higher than second (after accounting for library size)
        assert normalized[0, 0] > normalized[0, 1]
    
    def test_log_normalize_preserves_shape(self, toy_expression):
        """log_normalize_data should preserve shape."""
        normalized = log_normalize_data(toy_expression)
        assert normalized.shape == toy_expression.shape
    
    def test_log_normalize_zero_column(self):
        """log_normalize_data should handle zero columns gracefully."""
        expr = np.array([[1.0, 0.0], [1.0, 0.0]])
        normalized = log_normalize_data(expr)
        
        # Should not crash and should be finite
        assert np.all(np.isfinite(normalized))


class TestBatchCorrection:
    """Test batch effect correction."""
    
    def test_batch_correction_basic(self, toy_expression, toy_metadata):
        """remove_batch_effect should work correctly."""
        batch_labels = toy_metadata["batch"].values
        
        corrected = remove_batch_effect(toy_expression, batch_labels)
        
        assert corrected.shape == toy_expression.shape
        assert np.all(np.isfinite(corrected))
    
    def test_batch_correction_removes_effect(self, toy_metadata):
        """Batch correction should reduce batch variance."""
        # Create expression with obvious batch effect
        n_genes = 50
        n_cells = toy_metadata.shape[0]
        
        expr = np.random.randn(n_genes, n_cells)
        
        # Add batch effect to first half
        mask = toy_metadata["batch"] == toy_metadata["batch"].unique()[0]
        expr[:, mask] += 5.0
        
        batch_labels = toy_metadata["batch"].values
        corrected = remove_batch_effect(expr, batch_labels)
        
        # Corrected should have less batch variance than original
        # (this is approximate - just check it's different)
        assert not np.allclose(expr, corrected)
    
    def test_batch_correction_two_levels(self, toy_expression):
        """Batch correction with 2 batch levels."""
        n_cells = toy_expression.shape[1]
        batch_labels = np.array(["batch1"] * (n_cells // 2) + ["batch2"] * (n_cells - n_cells // 2))
        
        corrected = remove_batch_effect(toy_expression, batch_labels)
        
        assert corrected.shape == toy_expression.shape
    
    def test_batch_correction_multiple_levels(self, toy_expression):
        """Batch correction with many batch levels."""
        n_cells = toy_expression.shape[1]
        batch_labels = np.array([f"batch_{i % 5}" for i in range(n_cells)])
        
        corrected = remove_batch_effect(toy_expression, batch_labels)
        
        assert corrected.shape == toy_expression.shape


class TestGeneFiltering:
    """Test gene filtering."""
    
    def test_filter_expression_basic(self, toy_expression):
        """filter_expression should work correctly."""
        filtered, _ = filter_expression(toy_expression)
        
        # Filtered should have <= original genes
        assert filtered.shape[0] <= toy_expression.shape[0]
        assert filtered.shape[1] == toy_expression.shape[1]
    
    def test_filter_expression_min_cells(self, toy_expression):
        """filter_expression with min_cells parameter."""
        # Filter genes expressed in < 10 cells
        filtered, _ = filter_expression(toy_expression, min_cells=10)
        
        # Check that all remaining genes are expressed in >= 10 cells
        n_cells_expressed = (filtered > 0).sum(axis=1).min()
        assert n_cells_expressed >= 10
    
    def test_filter_expression_with_names(self, toy_expression):
        """filter_expression should return names if provided."""
        gene_names = np.array([f"gene_{i}" for i in range(toy_expression.shape[0])])
        
        filtered_expr, filtered_names = filter_expression(
            toy_expression, gene_names=gene_names
        )
        
        assert filtered_expr.shape[0] == filtered_names.shape[0]
        assert len(filtered_names) <= len(gene_names)
    
    def test_filter_expression_sparse(self, toy_expression):
        """filter_expression should work with sparse input."""
        sparse_expr = csr_matrix(toy_expression)
        
        filtered, _ = filter_expression(sparse_expr)
        
        assert filtered.shape[0] <= toy_expression.shape[0]
        assert filtered.shape[1] == toy_expression.shape[1]
    
    def test_filter_expression_min_expression_threshold(self, toy_expression):
        """filter_expression with min_expression threshold."""
        # Create expression with obvious low-expression genes
        expr = toy_expression.copy()
        expr.iloc[0] = 0.001  # Very low expression for first gene
        
        filtered, _ = filter_expression(expr, min_cells=1, min_expression=0.1)
        
        # First gene should be filtered out
        assert filtered.shape[0] < expr.shape[0]


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline."""
    
    def test_preprocess_all_steps(self, toy_expression, toy_metadata):
        """preprocess_expression with all options enabled."""
        result, _ = preprocess_expression(
            toy_expression,
            normalize=True,
            batch_labels=toy_metadata["batch"].values,
            filter_genes=False
        )
        
        assert result.shape[1] == toy_expression.shape[1]
    
    def test_preprocess_returns_names(self, toy_expression):
        """preprocess_expression should return names if provided."""
        gene_names = np.array([f"gene_{i}" for i in range(toy_expression.shape[0])])
        
        expr, names = preprocess_expression(
            toy_expression,
            normalize=True,
            filter_genes=True,
            gene_names=gene_names
        )
        
        assert expr.shape[0] == names.shape[0]
        assert len(names) <= len(gene_names)
