"""
Tests for run_scorpion() - group-stratified network inference.
"""

import pytest
import numpy as np
import pandas as pd
from scorpion import run_scorpion


# R's runSCORPION() enforces at least 30 cells per group, so these tests use
# a larger, balanced dataset than the shared toy fixtures.

@pytest.fixture
def toy_expression():
    rng = np.random.default_rng(42)
    gex = rng.poisson(5, (100, 240)).astype(float)
    return pd.DataFrame(
        gex,
        index=[f"gene_{i}" for i in range(100)],
        columns=[f"cell_{i}" for i in range(240)],
    )


@pytest.fixture
def toy_metadata(toy_expression):
    n_cells = toy_expression.shape[1]
    return pd.DataFrame({
        "cell_id": toy_expression.columns,
        "cell_type": np.repeat(["TypeB", "TypeA", "TypeC"], n_cells // 3),
        "donor": np.tile(["donor_1", "donor_2"], n_cells // 2),
        "batch": np.tile(["batch_1", "batch_1", "batch_2", "batch_2"], n_cells // 4),
    })


class TestRunScorpionComputation:
    """Test run_scorpion() group-stratified network inference."""
    
    def test_run_scorpion_returns_dataframe(self, toy_expression, toy_motifs, toy_metadata):
        """run_scorpion() should return a DataFrame."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            show_progress=False
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_run_scorpion_has_tf_target_columns(self, toy_expression, toy_motifs, toy_metadata):
        """Output DataFrame must have 'tf' and 'target' columns."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            show_progress=False
        )
        
        assert "tf" in result.columns
        assert "target" in result.columns
    
    def test_run_scorpion_group_columns(self, toy_expression, toy_motifs, toy_metadata):
        """Output should have columns for each group."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            show_progress=False
        )
        
        # Should have tf, target, and cell_type groups
        n_groups = toy_metadata["cell_type"].nunique()
        assert result.shape[1] == 2 + n_groups
    
    def test_run_scorpion_single_group_column(self, toy_expression, toy_motifs, toy_metadata):
        """run_scorpion() with single group_by column."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="donor",
            min_cells=5,
            show_progress=False
        )
        
        assert "tf" in result.columns
        assert "target" in result.columns
        # Should have tf, target, plus one column per donor
        n_donors = toy_metadata["donor"].nunique()
        assert result.shape[1] == 2 + n_donors
    
    def test_run_scorpion_multiple_group_columns(self, toy_expression, toy_motifs, toy_metadata):
        """run_scorpion() with multiple group_by columns."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by=["cell_type", "donor"],
            min_cells=5,
            show_progress=False
        )
        
        assert "tf" in result.columns
        assert "target" in result.columns
        
        # Should have group columns for each unique combination
        # Some combos may have < 5 cells, so count those with >= 5
        # 3 cell types x 2 donors, 40 cells each, joined with "--" (as in R)
        group_cols = [c for c in result.columns if c not in ("tf", "target")]
        assert len(group_cols) == 6
        assert "TypeB--donor_1" in group_cols
    
    def test_run_scorpion_min_cells_filter(self, toy_expression, toy_motifs, toy_metadata):
        """No group left after filtering raises (as in R); each type has 80 cells."""
        with pytest.raises(ValueError, match="No groups have enough cells"):
            run_scorpion(
                toy_expression,
                toy_motifs,
                cells_metadata=toy_metadata,
                group_by="cell_type",
                min_cells=81,
                show_progress=False
            )

    def test_run_scorpion_min_cells_floor(self, toy_expression, toy_motifs, toy_metadata):
        """min_cells below 30 is raised to 30 (R: max(minCells, 30))."""
        small = toy_metadata.copy()
        small.loc[:19, "cell_type"] = "Tiny"  # 20 cells
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=small,
            group_by="cell_type",
            min_cells=5,
            show_progress=False
        )
        assert "Tiny" not in result.columns
    
    def test_run_scorpion_normalization(self, toy_expression, toy_motifs, toy_metadata):
        """run_scorpion() should normalize by default."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            normalize_data=True,
            show_progress=False
        )
        
        assert "tf" in result.columns
        assert result.shape[0] > 0
    
    def test_run_scorpion_no_normalization(self, toy_expression, toy_motifs, toy_metadata):
        """run_scorpion() with normalize_data=False."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            normalize_data=False,
            show_progress=False
        )
        
        assert "tf" in result.columns
        assert result.shape[0] > 0
    
    def test_run_scorpion_batch_correction(self, toy_expression, toy_motifs, toy_metadata):
        """run_scorpion() with batch correction enabled."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            remove_batch_effect=True,
            batch="batch",
            show_progress=False
        )
        
        assert "tf" in result.columns
        assert result.shape[0] > 0
    
    def test_run_scorpion_network_types(self, toy_expression, toy_motifs, toy_metadata):
        """run_scorpion() should support different output networks."""
        for net_type in ["regNet", "coregNet", "coopNet"]:
            result = run_scorpion(
                toy_expression,
                toy_motifs,
                cells_metadata=toy_metadata,
                group_by="cell_type",
                min_cells=5,
                out_net=net_type,
                show_progress=False
            )
            
            assert "tf" in result.columns
            assert "target" in result.columns
    
    def test_run_scorpion_full_grid_no_nan(self, toy_expression, toy_motifs, toy_metadata):
        """Output is the full TF x target grid; no NaN when all groups share genes."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            show_progress=False
        )

        n_tfs = result["tf"].nunique()
        n_targets = result["target"].nunique()
        assert result.shape[0] == n_tfs * n_targets
        assert not result.isna().any().any()
        # tf varies fastest, as in R's expand.grid
        assert result["target"].iloc[0] == result["target"].iloc[n_tfs - 1]

    def test_run_scorpion_missing_edges_are_nan(self, toy_expression, toy_motifs, toy_metadata):
        """Edges absent from a group's network are NaN (R: NA), not 0."""
        gex = toy_expression.copy()
        gene = toy_motifs["target"].iloc[0]
        gex.loc[gene, toy_metadata["cell_type"].eq("TypeA").to_numpy()] = 0.0
        result = run_scorpion(
            gex,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            filter_expr=True,
            show_progress=False
        )
        rows = result["target"] == gene
        assert rows.any()
        assert result.loc[rows, "TypeA"].isna().all()
        assert result.loc[rows, "TypeB"].notna().all()

    def test_run_scorpion_group_order(self, toy_expression, toy_motifs, toy_metadata):
        """Group columns follow order of first appearance (as in R)."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            show_progress=False
        )
        assert list(result.columns) == ["tf", "target", "TypeB", "TypeA", "TypeC"]

    def test_run_scorpion_multiple_out_nets(self, toy_expression, toy_motifs, toy_metadata):
        """Several out_net values are stacked with an edge_type column."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            out_net=["regNet", "coopNet"],
            show_progress=False
        )
        assert list(result.columns[:3]) == ["edge_type", "tf", "target"]
        assert list(pd.unique(result["edge_type"])) == ["tf-target", "tf-tf"]
        single = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            out_net="regNet",
            show_progress=False
        )
        reg = result[result["edge_type"] == "tf-target"].drop(columns="edge_type")
        pd.testing.assert_frame_equal(reg.reset_index(drop=True), single)

    def test_run_scorpion_invalid_out_net(self, toy_expression, toy_motifs, toy_metadata):
        with pytest.raises(ValueError, match="out_net"):
            run_scorpion(
                toy_expression,
                toy_motifs,
                cells_metadata=toy_metadata,
                group_by="cell_type",
                out_net="bogus",
                show_progress=False
            )

    def test_run_scorpion_batch_vector(self, toy_expression, toy_motifs, toy_metadata):
        """batch may be a per-cell vector (R) or a metadata column name."""
        by_vector = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            remove_batch_effect=True,
            batch=toy_metadata["batch"].to_numpy(),
            random_state=1,
            show_progress=False
        )
        by_column = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            remove_batch_effect=True,
            batch="batch",
            random_state=1,
            show_progress=False
        )
        pd.testing.assert_frame_equal(by_vector, by_column)

    def test_run_scorpion_single_batch_skipped(self, toy_expression, toy_motifs, toy_metadata):
        """A single-level batch skips correction entirely (no median shift)."""
        uncorrected = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            random_state=1,
            show_progress=False
        )
        with pytest.warns(UserWarning, match="fewer than two levels"):
            single = run_scorpion(
                toy_expression,
                toy_motifs,
                cells_metadata=toy_metadata,
                group_by="cell_type",
                remove_batch_effect=True,
                batch=["b1"] * toy_expression.shape[1],
                random_state=1,
                show_progress=False
            )
        pd.testing.assert_frame_equal(single, uncorrected)

    def test_run_scorpion_batch_required(self, toy_expression, toy_motifs, toy_metadata):
        with pytest.raises(ValueError, match="batch must be provided"):
            run_scorpion(
                toy_expression,
                toy_motifs,
                cells_metadata=toy_metadata,
                group_by="cell_type",
                remove_batch_effect=True,
                show_progress=False
            )
    
    def test_run_scorpion_alpha_parameter(self, toy_expression, toy_motifs, toy_metadata):
        """run_scorpion() should accept alpha_value parameter."""
        result_alpha_0 = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            alpha_value=0.0,
            show_progress=False
        )
        
        result_alpha_1 = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            alpha_value=1.0,
            show_progress=False
        )
        
        # Results should differ
        group_cols = [c for c in result_alpha_0.columns if c not in ["tf", "target"]]
        if group_cols:
            assert not np.allclose(
                result_alpha_0[group_cols[0]].values,
                result_alpha_1[group_cols[0]].values
            )
    
    def test_run_scorpion_reproducibility(self, toy_expression, toy_motifs, toy_metadata):
        """Same random seed should produce reproducible results."""
        result1 = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            random_state=42,
            show_progress=False
        )
        
        result2 = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            random_state=42,
            show_progress=False
        )
        
        # Results should match exactly
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_run_scorpion_with_ppi(self, toy_expression, toy_motifs, toy_ppi, toy_metadata):
        """run_scorpion() should work with PPI prior."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            ppi_net=toy_ppi,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            show_progress=False
        )
        
        assert "tf" in result.columns
        assert result.shape[0] > 0
    
    def test_run_scorpion_edge_list_format(self, toy_expression, toy_motifs, toy_metadata):
        """Output edge list should be proper wide format."""
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=toy_metadata,
            group_by="cell_type",
            min_cells=5,
            show_progress=False
        )
        
        # Check shape: rows are edges, columns are tf, target, groups
        assert result.shape[0] >= 0  # May be empty in edge cases
        assert result.shape[1] >= 2  # At least tf and target
        
        # Each edge should be unique
        if result.shape[0] > 0:
            n_edges = result.shape[0]
            n_unique = result[["tf", "target"]].drop_duplicates().shape[0]
            assert n_unique == n_edges


class TestRunScorpionGrouping:
    """Test run_scorpion() grouping logic."""
    
    def test_single_group(self, toy_expression, toy_motifs):
        """run_scorpion() with no group_by should work."""
        metadata = pd.DataFrame({"dummy": ["a"] * toy_expression.shape[1]})
        
        result = run_scorpion(
            toy_expression,
            toy_motifs,
            cells_metadata=metadata,
            group_by=None,
            min_cells=5,
            show_progress=False
        )
        
        assert "tf" in result.columns
        assert result.shape[0] > 0
