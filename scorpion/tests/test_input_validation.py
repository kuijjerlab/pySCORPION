"""
Tests for input validation and error handling.
"""

import pytest
import numpy as np
import pandas as pd
from scorpion import scorpion, run_scorpion
from scorpion.preprocessing import validate_inputs


class TestScorpionInputValidation:
    """Test scorpion() input validation."""
    
    def test_alpha_value_negative(self, toy_expression, toy_motifs):
        """alpha_value < 0 should raise error."""
        with pytest.raises(ValueError):
            scorpion(
                toy_expression,
                toy_motifs,
                alpha_value=-0.1,
                show_progress=False
            )
    
    def test_alpha_value_too_high(self, toy_expression, toy_motifs):
        """alpha_value > 1 should raise error."""
        with pytest.raises(ValueError):
            scorpion(
                toy_expression,
                toy_motifs,
                alpha_value=1.5,
                show_progress=False
            )
    
    def test_gamma_value_negative(self, toy_expression, toy_motifs):
        """gamma_value <= 0 should raise error."""
        with pytest.raises(ValueError):
            scorpion(
                toy_expression,
                toy_motifs,
                gamma_value=-1,
                show_progress=False
            )
    
    def test_gamma_value_zero(self, toy_expression, toy_motifs):
        """gamma_value = 0 should raise error."""
        with pytest.raises(ValueError):
            scorpion(
                toy_expression,
                toy_motifs,
                gamma_value=0,
                show_progress=False
            )
    
    def test_n_pc_exceeds_cells(self, toy_expression, toy_motifs):
        """n_pc > num_cells should be silently clamped (matching R behavior)."""
        n_cells = toy_expression.shape[1]
        # R clamps n_pc to min(n_pc, n_cells-1), so this should succeed
        result = scorpion(
            toy_expression,
            toy_motifs,
            n_pc=n_cells + 10,
            show_progress=False
        )
        assert "regNet" in result


class TestRunScorpionInputValidation:
    """Test run_scorpion() input validation."""
    
    def test_metadata_length_mismatch(self, toy_expression, toy_motifs):
        """gexMatrix and metadata dimension mismatch should raise error."""
        bad_metadata = pd.DataFrame({
            "cell_type": ["A"] * (toy_expression.shape[1] - 10)
        })
        
        with pytest.raises(ValueError):
            run_scorpion(
                toy_expression,
                toy_motifs,
                cells_metadata=bad_metadata,
                group_by="cell_type",
                show_progress=False
            )
    
    def test_invalid_group_by_column(self, toy_expression, toy_motifs, toy_metadata):
        """Invalid group_by column should raise error."""
        with pytest.raises(ValueError):
            run_scorpion(
                toy_expression,
                toy_motifs,
                cells_metadata=toy_metadata,
                group_by="nonexistent_column",
                show_progress=False
            )
    
    def test_batch_correction_without_batch_column(self, toy_expression, toy_motifs, toy_metadata):
        """Batch correction enabled without batch column should raise error."""
        bad_metadata = pd.DataFrame({
            "cell_type": toy_metadata["cell_type"]
        })
        
        with pytest.raises(ValueError):
            run_scorpion(
                toy_expression,
                toy_motifs,
                cells_metadata=bad_metadata,
                group_by="cell_type",
                remove_batch_effect=True,
                batch="batch",
                show_progress=False
            )
    
    def test_alpha_parameter_validation(self, toy_expression, toy_motifs, toy_metadata):
        """Invalid alpha_value in run_scorpion should raise error."""
        with pytest.raises(ValueError):
            run_scorpion(
                toy_expression,
                toy_motifs,
                cells_metadata=toy_metadata,
                group_by="cell_type",
                alpha_value=1.5,
                show_progress=False
            )


class TestValidateInputsFunction:
    """Test preprocessing.validate_inputs() directly."""
    
    def test_alpha_range_validation(self, toy_expression, toy_motifs):
        """alpha_value outside [0, 1] range."""
        valid, msg = validate_inputs(
            toy_expression, toy_motifs, alpha_value=-0.5
        )
        assert not valid
        assert "alpha_value" in msg
    
    def test_gamma_positive_validation(self, toy_expression, toy_motifs):
        """gamma_value must be positive."""
        valid, msg = validate_inputs(
            toy_expression, toy_motifs, gamma_value=0
        )
        assert not valid
        assert "gamma_value" in msg
    
    def test_n_pc_validation(self, toy_expression, toy_motifs):
        """n_pc must be positive."""
        # Test n_pc <= 0
        valid, msg = validate_inputs(
            toy_expression, toy_motifs, n_pc=-1
        )
        assert not valid
        assert "n_pc" in msg
        
        # n_pc > n_cells is now allowed (silently clamped, matching R)
        n_cells = toy_expression.shape[1]
        valid, msg = validate_inputs(
            toy_expression, toy_motifs, n_pc=n_cells + 1
        )
        assert valid
    
    def test_metadata_alignment(self, toy_expression, toy_motifs):
        """Metadata must have same number of cells as expression."""
        bad_metadata = pd.DataFrame({"col": [1, 2, 3]})
        
        valid, msg = validate_inputs(
            toy_expression, toy_motifs, cells_metadata=bad_metadata
        )
        assert not valid
        assert "cells_metadata" in msg
    
    def test_group_by_without_metadata(self, toy_expression, toy_motifs):
        """group_by requires cells_metadata."""
        valid, msg = validate_inputs(
            toy_expression, toy_motifs, group_by="cell_type"
        )
        assert not valid
        assert "cells_metadata" in msg
    
    def test_group_by_invalid_column(self, toy_expression, toy_motifs, toy_metadata):
        """group_by column must exist in metadata."""
        valid, msg = validate_inputs(
            toy_expression, toy_motifs,
            cells_metadata=toy_metadata,
            group_by="nonexistent"
        )
        assert not valid
        assert "group_by" in msg
    
    def test_batch_without_metadata(self, toy_expression, toy_motifs):
        """batch parameter requires cells_metadata."""
        valid, msg = validate_inputs(
            toy_expression, toy_motifs, batch="batch"
        )
        assert not valid
        assert "cells_metadata" in msg
    
    def test_batch_invalid_column(self, toy_expression, toy_motifs, toy_metadata):
        """batch column must exist in metadata."""
        valid, msg = validate_inputs(
            toy_expression, toy_motifs,
            cells_metadata=toy_metadata,
            batch="nonexistent"
        )
        assert not valid
        assert "batch" in msg
