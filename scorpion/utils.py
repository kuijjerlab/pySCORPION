"""
Utility functions for SCORPION: correlation, normalization, tanimoto similarity,
and other building blocks matching the R implementation exactly.
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, csc_matrix
from typing import Tuple, Union, List, Dict


def compute_correlation(
    data: Union[np.ndarray, csr_matrix, csc_matrix],
    method: str = "pearson",
) -> np.ndarray:
    """
    Compute correlation matrix from expression data.

    Matches R's fastCorrelation(t(expr), t(expr), method).
    For (genes x cells) input, computes gene-gene correlation.

    Parameters
    ----------
    data : {ndarray, csr_matrix, csc_matrix}
        Expression matrix (genes x cells).
    method : str, default="pearson"
        Correlation method: "pearson" or "spearman".

    Returns
    -------
    correlation_matrix : ndarray
        Correlation matrix (genes x genes).
    """
    if issparse(data):
        data = data.toarray()

    data = np.asarray(data, dtype=float)

    if method == "spearman":
        from scipy.stats import rankdata
        data = np.apply_along_axis(rankdata, 1, data)

    # Center each row (gene) across cells
    row_means = data.mean(axis=1, keepdims=True)
    centered = data - row_means

    # Row norms
    norms = np.sqrt(np.sum(centered ** 2, axis=1))
    norms[norms == 0] = 1.0  # avoid division by zero

    # Correlation: centered @ centered.T / outer(norms, norms)
    corr = (centered @ centered.T) / np.outer(norms, norms)

    return corr


def normalize_network_panda(X: np.ndarray) -> np.ndarray:
    """
    Normalize network using R's normalizeNetwork: combined row + column z-score.

    Formula:
        Z1 = row-wise z-score (population std)
        Z2 = column-wise z-score (population std)
        normMat = Z1/sqrt(2) + Z2/sqrt(2)

    With NaN fallback:
        - Where Z1 is NaN (zero-variance row): use Z2/sqrt(2) + Z0/sqrt(2)
        - Where Z2 is NaN (zero-variance col): use Z1/sqrt(2) + Z0/sqrt(2)
        - Where both NaN: use 2*Z0/sqrt(2)

    Z0 is the global z-score with sample std (ddof=1).

    Parameters
    ----------
    X : ndarray
        Network matrix to normalize.

    Returns
    -------
    normMat : ndarray
        Normalized network.
    """
    X = np.asarray(X, dtype=float)
    nr, nc = X.shape

    # Global z-score (R uses sd() which is ddof=1)
    mu0 = np.mean(X)
    std0 = np.std(X, ddof=1)
    if std0 == 0:
        std0 = 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        Z0 = (X - mu0) / std0

    # Row z-score with population std
    # R: rowSds(X) * sqrt((nc-1)/nc) = population std
    mu1 = np.mean(X, axis=1, keepdims=True)  # (nr, 1)
    std1 = np.std(X, axis=1, ddof=0, keepdims=True)  # population std

    with np.errstate(divide="ignore", invalid="ignore"):
        Z1 = (X - mu1) / std1

    # Column z-score with population std
    # R: colSds(X) * sqrt((nr-1)/nr) = population std
    mu2 = np.mean(X, axis=0, keepdims=True)  # (1, nc)
    std2 = np.std(X, axis=0, ddof=0, keepdims=True)  # population std

    with np.errstate(divide="ignore", invalid="ignore"):
        Z2 = (X - mu2) / std2

    # Combine
    sqrt2 = np.sqrt(2.0)
    norm_mat = Z1 / sqrt2 + Z2 / sqrt2

    # Fix NaN values (from zero-variance rows/columns)
    f1 = np.isnan(Z1)
    f2 = np.isnan(Z2)

    norm_mat[f1] = Z2[f1] / sqrt2 + Z0[f1] / sqrt2
    norm_mat[f2] = Z1[f2] / sqrt2 + Z0[f2] / sqrt2
    norm_mat[f1 & f2] = 2.0 * Z0[f1 & f2] / sqrt2

    # Replace any remaining NaN with 0
    norm_mat = np.nan_to_num(norm_mat, nan=0.0)

    return norm_mat


def tanimoto(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute tanimoto similarity between two matrices.

    Matches R's tanimoto(X, Y) exactly:
        A = X %*% Y
        B = colSums(Y * Y)  (repeated per row)
        C = rowSums(X * X)  (repeated per col)
        den = B + C - abs(A)
        result = A / sqrt(den)

    Parameters
    ----------
    X : ndarray
        First matrix (nr x m).
    Y : ndarray
        Second matrix (m x nc).

    Returns
    -------
    result : ndarray
        Tanimoto similarity matrix (nr x nc).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    nr = X.shape[0]
    nc = Y.shape[1]

    A = X @ Y  # (nr, nc)

    # B = colSums(Y * Y), shape (nc,)
    B = np.sum(Y * Y, axis=0)
    # C = rowSums(X * X), shape (nr,)
    C = np.sum(X * X, axis=1)

    # Broadcast to (nr, nc)
    B_mat = np.broadcast_to(B[np.newaxis, :], (nr, nc)).copy()
    C_mat = np.broadcast_to(C[:, np.newaxis], (nr, nc)).copy()

    den = B_mat + C_mat - np.abs(A)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = A / np.sqrt(den)

    return result


def update_diagonal(diag_mat: np.ndarray, num: int, alpha: float, step: int) -> np.ndarray:
    """
    Update diagonal of a square matrix.

    Matches R's update.diagonal(diagMat, num, alpha, step):
        diag(diagMat) = NaN
        diagstd = rowSds(diagMat, na.rm=TRUE) * sqrt((num-2)/(num-1))
                = population std of non-diagonal elements
        diag(diagMat) = diagstd * num * exp(2 * alpha * step)

    Parameters
    ----------
    diag_mat : ndarray
        Square matrix.
    num : int
        Dimension of the matrix (num x num).
    alpha : float
        PANDA alpha parameter.
    step : int
        Current iteration step.

    Returns
    -------
    updated_mat : ndarray
        Matrix with updated diagonal.
    """
    mat = np.array(diag_mat, dtype=float)
    np.fill_diagonal(mat, np.nan)

    # Population std of each row's non-NaN elements
    # R: rowSds(na.rm=TRUE) * sqrt((num-2)/(num-1)) = population std
    diag_std = np.nanstd(mat, axis=1, ddof=0)

    np.fill_diagonal(mat, diag_std * num * np.exp(2 * alpha * step))

    return mat


def hamming_distance(X: np.ndarray, Y: np.ndarray, n_rows: int, n_cols: int) -> float:
    """
    Compute Hamming distance between two matrices.

    Matches R's: sum(abs(regulatoryNetwork - RA)) / (num.TFs * num.genes)

    Parameters
    ----------
    X : ndarray
        Matrix 1.
    Y : ndarray
        Matrix 2.
    n_rows : int
        Number of rows (e.g., num TFs).
    n_cols : int
        Number of columns (e.g., num genes).

    Returns
    -------
    distance : float
        Hamming distance.
    """
    return np.sum(np.abs(X - Y)) / (n_rows * n_cols)


def normalize_network(
    network: np.ndarray,
    method: str = "zscore",
    axis: int = None,
) -> np.ndarray:
    """
    Simple normalization (legacy, used by tests).

    Parameters
    ----------
    network : ndarray
        Network weight matrix.
    method : str, default="zscore"
        "zscore" or "range".
    axis : int, optional
        Axis to normalize along.

    Returns
    -------
    normalized_network : ndarray
    """
    from sklearn.preprocessing import StandardScaler

    if method == "zscore":
        if axis is None:
            scaler = StandardScaler()
            shape = network.shape
            normalized = scaler.fit_transform(network.reshape(-1, 1)).reshape(shape)
        else:
            if axis == 0:
                mean = network.mean(axis=0, keepdims=True)
                std = network.std(axis=0, keepdims=True)
            else:
                mean = network.mean(axis=1, keepdims=True)
                std = network.std(axis=1, keepdims=True)
            normalized = (network - mean) / (std + 1e-10)
    elif method == "range":
        min_val = network.min()
        max_val = network.max()
        normalized = (network - min_val) / (max_val - min_val + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return np.nan_to_num(normalized, nan=0.0)


def normalize_edges(
    edges: Union[np.ndarray, pd.DataFrame],
    method: str = "zscore",
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Normalize edge weights in a network edge list or matrix.

    Parameters
    ----------
    edges : {ndarray, DataFrame}
        Edge matrix or DataFrame.
    method : str, default="zscore"
        Normalization method.

    Returns
    -------
    normalized_edges : same type as input
    """
    is_df = isinstance(edges, pd.DataFrame)

    if is_df:
        edges_copy = edges.copy()
        weights = edges_copy.iloc[:, -1].values
    else:
        edges_copy = edges.copy()
        weights = edges_copy[:, -1]

    if method == "zscore":
        normalized_weights = (weights - weights.mean()) / (weights.std() + 1e-10)
    elif method == "range":
        min_val = weights.min()
        max_val = weights.max()
        normalized_weights = (weights - min_val) / (max_val - min_val + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if is_df:
        edges_copy.iloc[:, -1] = normalized_weights
    else:
        edges_copy[:, -1] = normalized_weights

    return edges_copy


def edge_list_to_matrix(
    edge_list: pd.DataFrame,
    gene_names: List[str] = None,
    tf_names: List[str] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Convert edge list to matrix representation.

    Parameters
    ----------
    edge_list : DataFrame
        Edge list with columns [source, target, weight].
    gene_names : list, optional
        Gene names (rows).
    tf_names : list, optional
        TF names (columns).

    Returns
    -------
    matrix : ndarray
        Edge matrix (genes x sources).
    gene_names : list
    tf_names : list
    """
    if gene_names is None:
        gene_names = sorted(set(edge_list.iloc[:, 1].values))
    if tf_names is None:
        tf_names = sorted(set(edge_list.iloc[:, 0].values))

    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    tf_to_idx = {tf: i for i, tf in enumerate(tf_names)}

    matrix = np.zeros((len(gene_names), len(tf_names)))

    tfs = edge_list.iloc[:, 0].values
    genes = edge_list.iloc[:, 1].values
    weights = edge_list.iloc[:, 2].to_numpy(dtype=float)

    tf_idx = np.array([tf_to_idx.get(t, -1) for t in tfs])
    gene_idx = np.array([gene_to_idx.get(g, -1) for g in genes])
    valid = (tf_idx >= 0) & (gene_idx >= 0)
    matrix[gene_idx[valid], tf_idx[valid]] = weights[valid]

    return matrix, gene_names, tf_names


def matrix_to_edge_list(
    matrix: np.ndarray,
    row_names: List[str],
    col_names: List[str],
    include_zeros: bool = False,
) -> pd.DataFrame:
    """
    Convert matrix to edge list representation.

    Parameters
    ----------
    matrix : ndarray
        Network matrix.
    row_names : list
        Row labels.
    col_names : list
        Column labels.
    include_zeros : bool, default=False
        Include zero-weight edges.

    Returns
    -------
    edge_list : DataFrame
        Edge list [source, target, weight].
    """
    if include_zeros:
        row_idx, col_idx = np.mgrid[:len(row_names), :len(col_names)]
        row_idx = row_idx.ravel()
        col_idx = col_idx.ravel()
    else:
        row_idx, col_idx = np.nonzero(matrix)

    return pd.DataFrame({
        "source": [col_names[j] for j in col_idx],
        "target": [row_names[i] for i in row_idx],
        "weight": matrix[row_idx, col_idx],
    })


def restrict_networks(
    edge_list: Union[np.ndarray, pd.DataFrame],
    genes_in_scope: set,
    tfs_in_scope: set = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Restrict edge list to genes and TFs in scope.

    Parameters
    ----------
    edge_list : {ndarray, DataFrame}
        Edge list (cols: [source, target, weight, ...]).
    genes_in_scope : set
        Valid gene names.
    tfs_in_scope : set, optional
        Valid TF names.

    Returns
    -------
    restricted_edges : same type as input
    """
    if tfs_in_scope is None:
        tfs_in_scope = genes_in_scope

    is_df = isinstance(edge_list, pd.DataFrame)

    if is_df:
        mask = (edge_list.iloc[:, 0].isin(tfs_in_scope)) & \
               (edge_list.iloc[:, 1].isin(genes_in_scope))
        return edge_list[mask].reset_index(drop=True)
    else:
        mask = np.array([
            (row[0] in tfs_in_scope) and (row[1] in genes_in_scope)
            for row in edge_list[:, :2]
        ])
        return edge_list[mask]


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    # Also set legacy seed for libraries that still use it
    np.random.seed(seed)
