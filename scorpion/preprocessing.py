"""
Data preprocessing module: validation, normalization, batch correction.
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, csc_matrix
from sklearn.decomposition import PCA
from typing import Union, Tuple, Optional


def validate_inputs(
    gex_matrix: Union[np.ndarray, csr_matrix, csc_matrix, pd.DataFrame],
    tf_motifs: Union[np.ndarray, pd.DataFrame],
    cells_metadata: Optional[pd.DataFrame] = None,
    ppi_net: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    group_by: Optional[str] = None,
    alpha_value: float = 0.1,
    gamma_value: int = 10,
    n_pc: int = 25,
    batch: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate input parameters and data dimensions.
    
    Parameters
    ----------
    gex_matrix : {ndarray, csr_matrix, csc_matrix, DataFrame}
        Gene expression matrix (genes × cells).
    tf_motifs : {ndarray, DataFrame}
        TF-target motif pairs (n_motifs × 3): [TF, target, score].
    cells_metadata : DataFrame, optional
        Cell metadata (n_cells × n_features).
    ppi_net : {ndarray, DataFrame}, optional
        Protein-protein interactions (n_ppis × 3): [protein1, protein2, score].
    group_by : str, optional
        Column name in cells_metadata for grouping.
    alpha_value : float, default=0.1
        PANDA alpha parameter (0-1).
    gamma_value : int, default=10
        Super-cell aggregation level.
    n_pc : int, default=25
        Number of PCA components for kNN.
    batch : str, optional
        Batch column name in cells_metadata.
    
    Returns
    -------
    valid : bool
        Whether all validations passed.
    message : str
        Validation message (empty if valid, error description otherwise).
    """
    # Cell metadata alignment
    if cells_metadata is not None:
        n_cells_expr = gex_matrix.shape[1]
        n_cells_meta = cells_metadata.shape[0]
        if n_cells_expr != n_cells_meta:
            return False, (
                f"gex_matrix has {n_cells_expr} cells but cells_metadata has "
                f"{n_cells_meta} rows"
            )
    
    # Group column existence
    if group_by is not None:
        if cells_metadata is None:
            return False, "cells_metadata required when group_by is specified"
        # Handle both string and list of columns
        gb_list = [group_by] if isinstance(group_by, str) else group_by
        for col in gb_list:
            if col not in cells_metadata.columns:
                return False, f"group_by column '{col}' not found in cells_metadata"
    
    # Batch column existence
    if batch is not None:
        if cells_metadata is None:
            return False, "cells_metadata required when batch is specified"
        if batch not in cells_metadata.columns:
            return False, f"batch column '{batch}' not found in cells_metadata"
    
    # Alpha parameter range
    if not (0 <= alpha_value <= 1):
        return False, f"alpha_value must be in [0, 1], got {alpha_value}"
    
    # Gamma parameter (should be positive)
    if gamma_value <= 0:
        return False, f"gamma_value must be positive, got {gamma_value}"
    
    # n_pc parameter (should be positive; will be silently clamped if > n_cells)
    if n_pc <= 0:
        return False, f"n_pc must be positive, got {n_pc}"
    
    # TF motifs format
    if isinstance(tf_motifs, pd.DataFrame):
        if tf_motifs.shape[1] < 3:
            return False, "tf_motifs must have at least 3 columns"
    else:
        if tf_motifs.ndim != 2 or tf_motifs.shape[1] < 3:
            return False, "tf_motifs must be 2D array with at least 3 columns"
    
    # PPI network format (if provided)
    if ppi_net is not None:
        if isinstance(ppi_net, pd.DataFrame):
            if ppi_net.shape[1] < 3:
                return False, "ppi_net must have at least 3 columns"
        else:
            if ppi_net.ndim != 2 or ppi_net.shape[1] < 3:
                return False, "ppi_net must be 2D array with at least 3 columns"
    
    return True, ""


def log_normalize_data(
    gex_matrix: Union[np.ndarray, csr_matrix, csc_matrix, pd.DataFrame],
    scale_factor: float = 10000.0,
) -> np.ndarray:
    """
    Log-normalize expression matrix: CPM normalization followed by log1p.
    
    Formula: log1p(CPM × scale_factor) = log1p(matrix / colsums * scale_factor)
    
    Parameters
    ----------
    gex_matrix : {ndarray, csr_matrix, csc_matrix, DataFrame}
        Expression matrix (genes × cells), typically raw counts.
    scale_factor : float, default=10000.0
        Scaling factor (default: 10000 for 10K CPM).
    
    Returns
    -------
    normalized : ndarray
        Log-normalized expression matrix (genes × cells).
    """
    # Convert to dense if sparse
    if issparse(gex_matrix):
        gex_matrix = gex_matrix.toarray()
    elif isinstance(gex_matrix, pd.DataFrame):
        gex_matrix = gex_matrix.values
    else:
        gex_matrix = np.asarray(gex_matrix, dtype=float)
    
    # Compute column sums (library sizes)
    col_sums = gex_matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    
    # CPM normalization: normalize by library size and scale
    cpm = (gex_matrix / col_sums) * scale_factor
    
    # Log transform
    normalized = np.log1p(cpm)
    
    return normalized


def remove_batch_effect(
    gex_matrix: np.ndarray,
    batch_labels: Union[np.ndarray, pd.Series],
    design_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Remove batch effects using design matrix regression (limma-style).
    
    For each gene, fits a linear model:
        expression ~ biological_vars + batch_vars
    Returns residuals.
    
    Parameters
    ----------
    gex_matrix : ndarray
        Gene expression matrix (genes × cells).
    batch_labels : {ndarray, Series}
        Batch labels (n_cells,).
    design_matrix : ndarray, optional
        Design matrix (n_cells × n_vars). If None, uses one-hot encoded batch labels.
    
    Returns
    -------
    batch_corrected : ndarray
        Batch-corrected expression matrix (genes × cells).
    """
    gex_matrix = np.asarray(gex_matrix, dtype=float)
    batch_labels = np.asarray(batch_labels)
    
    n_genes, n_cells = gex_matrix.shape
    
    if design_matrix is None:
        # Create one-hot encoded batch design matrix
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
        batch_indices = np.array([batch_to_idx[b] for b in batch_labels])

        design_matrix = np.zeros((n_cells, n_batches))
        design_matrix[np.arange(n_cells), batch_indices] = 1
    
    design_matrix = np.asarray(design_matrix, dtype=float)

    # Solve all genes at once: expression.T = design_matrix @ beta
    beta, _, _, _ = np.linalg.lstsq(design_matrix, gex_matrix.T, rcond=None)
    batch_corrected = gex_matrix - (design_matrix @ beta).T
    
    return batch_corrected


def filter_expression(
    gex_matrix: Union[np.ndarray, csr_matrix, csc_matrix, pd.DataFrame],
    gene_names: Optional[np.ndarray] = None,
    min_cells: int = 1,
    min_expression: float = 0,
) -> Tuple[Union[np.ndarray, csr_matrix], Optional[np.ndarray]]:
    """
    Filter genes based on expression criteria.
    
    Removes genes that:
    - Have zero expression across all cells
    - Are expressed in fewer than min_cells cells
    - Have minimum expression below threshold
    
    Parameters
    ----------
    gex_matrix : {ndarray, csr_matrix, csc_matrix, DataFrame}
        Expression matrix (genes × cells).
    gene_names : ndarray, optional
        Gene names (n_genes,). If provided, returns filtered names.
    min_cells : int, default=1
        Minimum number of cells expressing a gene.
    min_expression : float, default=0
        Minimum expression level to count as "expressed".
    
    Returns
    -------
    filtered_matrix : {ndarray, csr_matrix}
        Filtered expression matrix.
    filtered_gene_names : ndarray or None
        Filtered gene names (None if gene_names was not provided).
    """
    if issparse(gex_matrix):
        matrix = gex_matrix.tocsr()
        # Count cells with expression > threshold per gene
        expressed_per_gene = np.asarray((matrix > min_expression).sum(axis=1)).flatten()
    else:
        matrix = np.asarray(gex_matrix, dtype=float)
        expressed_per_gene = (matrix > min_expression).sum(axis=1)
    
    # Genes to keep: expressed in >= min_cells cells
    keep_genes = expressed_per_gene >= min_cells
    
    if np.sum(keep_genes) == 0:
        raise ValueError("No genes pass filtering criteria")
    
    filtered_matrix = matrix[keep_genes, :]
    filtered_gene_names = gene_names[keep_genes] if gene_names is not None else None
    
    return filtered_matrix, filtered_gene_names


def preprocess_expression(
    gex_matrix: Union[np.ndarray, csr_matrix, csc_matrix, pd.DataFrame],
    normalize: bool = True,
    batch_labels: Optional[Union[np.ndarray, pd.Series]] = None,
    filter_genes: bool = False,
    gene_names: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """
    Complete preprocessing pipeline: normalize, batch-correct, filter.
    
    Parameters
    ----------
    gex_matrix : {ndarray, csr_matrix, csc_matrix, DataFrame}
        Expression matrix (genes × cells).
    normalize : bool, default=True
        Whether to apply log normalization.
    batch_labels : {ndarray, Series}, optional
        Batch labels for batch correction.
    filter_genes : bool, default=False
        Whether to filter genes.
    gene_names : ndarray, optional
        Gene names for filtering and output.
    
    Returns
    -------
    preprocessed : ndarray
        Preprocessed expression matrix.
    filtered_gene_names : ndarray or None
        Filtered gene names (None if gene_names was not provided or
        filter_genes is False).
    """
    result = np.asarray(gex_matrix, dtype=float)
    filtered_gene_names = None
    
    if normalize:
        result = log_normalize_data(result)
    
    if batch_labels is not None:
        result = remove_batch_effect(result, batch_labels)
    
    if filter_genes:
        result, filtered_gene_names = filter_expression(result, gene_names)
    
    return result, filtered_gene_names
