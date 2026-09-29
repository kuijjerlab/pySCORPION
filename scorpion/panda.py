"""
PANDA: Passing Attributes between Networks for Data Assimilation algorithm.

This implementation matches the R SCORPION package's runPANDA() exactly,
using tanimoto similarity for message passing and R's normalizeNetwork
for pre-normalization.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
from typing import Tuple, Optional, Dict, List, Union
from .utils import (
    compute_correlation,
    normalize_network_panda,
    tanimoto,
    update_diagonal,
    hamming_distance,
)


def run_panda(
    expression_matrix: np.ndarray,
    tf_motifs: Union[np.ndarray, pd.DataFrame],
    ppi_network: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    alpha: float = 0.1,
    hamming_threshold: float = 0.001,
    max_iterations: int = 10000,
    correlation_method: str = "pearson",
    verbose: bool = True,
    gene_names: Optional[np.ndarray] = None,
    tf_names: Optional[np.ndarray] = None,
    randomize: Optional[str] = None,
    scale_by_present: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run PANDA algorithm for network inference.

    Matches R's runPANDA() exactly:
    1. Intersection-filter genes/TFs
    2. Build motif (TFs x genes), PPI (TFs x TFs), correlation (genes x genes)
    3. Normalize all three with normalizeNetwork (dual row+col z-score)
    4. Iterate: tanimoto message passing until convergence

    Parameters
    ----------
    expression_matrix : ndarray
        Gene expression matrix (genes x cells/super-cells).
    tf_motifs : {ndarray, DataFrame}
        TF-target motif prior. Edge list: 3 columns [TF, target, weight].
    ppi_network : {ndarray, DataFrame}, optional
        PPI prior. Edge list: 3 columns [protein1, protein2, weight].
    alpha : float, default=0.1
        Update parameter (0-1).
    hamming_threshold : float, default=0.001
        Convergence threshold.
    max_iterations : int, default=10000
        Maximum iterations.
    correlation_method : str, default="pearson"
        Correlation method: "pearson" or "spearman".
    verbose : bool, default=True
        Print iteration progress.
    gene_names : ndarray, optional
        Gene names (n_genes,). Used to match motif targets.
    tf_names : ndarray, optional
        TF names. If None, derived from motifs.
    randomize : str, optional
        Randomization of the expression matrix for null models, as in R's
        runPANDA(randomize=): None/"None" (default), "within.gene"
        (permute each gene's values across samples) or "by.gene"
        (permute gene labels). Underscore spellings are also accepted.
    scale_by_present : bool, default=False
        Scale correlations by the fraction of samples in which both genes
        are expressed (> 0).

    Returns
    -------
    regulatory_network : ndarray
        TF -> gene regulatory network (TFs x genes).
    gene_coregulation : ndarray
        Gene co-regulation network (genes x genes).
    tf_cooperation : ndarray
        TF cooperation network (TFs x TFs).
    """
    n_expr_genes = expression_matrix.shape[0]

    # Build gene name mapping
    if gene_names is not None:
        gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    else:
        gene_to_idx = {i: i for i in range(n_expr_genes)}

    # Extract unique TF and gene names from motifs
    if isinstance(tf_motifs, pd.DataFrame) and tf_motifs.shape[1] >= 3:
        motif_tfs = set(tf_motifs.iloc[:, 0])
        motif_targets = set(tf_motifs.iloc[:, 1])
    elif isinstance(tf_motifs, np.ndarray) and tf_motifs.ndim == 2 and tf_motifs.shape[1] >= 3:
        motif_tfs = set(tf_motifs[:, 0])
        motif_targets = set(tf_motifs[:, 1])
    else:
        motif_tfs = set()
        motif_targets = set()

    # Extract unique TF names from PPI
    ppi_tfs = set()
    if ppi_network is not None:
        if isinstance(ppi_network, pd.DataFrame):
            if ppi_network.shape[1] >= 2:
                ppi_tfs.update(ppi_network.iloc[:, 0].values)
                ppi_tfs.update(ppi_network.iloc[:, 1].values)
        elif isinstance(ppi_network, np.ndarray):
            if ppi_network.ndim == 2 and ppi_network.shape[1] >= 2:
                ppi_tfs.update(ppi_network[:, 0])
                ppi_tfs.update(ppi_network[:, 1])

    # Intersection filtering: keep genes/TFs present in both expression and priors
    expr_genes = set(gene_to_idx.keys())

    valid_genes = sorted(expr_genes & motif_targets) if motif_targets else sorted(expr_genes)
    if motif_tfs and ppi_tfs:
        valid_tfs = sorted(motif_tfs & ppi_tfs)
    elif motif_tfs:
        valid_tfs = sorted(motif_tfs)
    else:
        valid_tfs = []

    # Fallback if intersection is empty
    if not valid_genes:
        valid_genes = sorted(expr_genes)
    if not valid_tfs:
        valid_tfs = sorted(motif_tfs) if motif_tfs else []

    valid_gene_to_idx = {name: i for i, name in enumerate(valid_genes)}
    valid_tf_to_idx = {name: i for i, name in enumerate(valid_tfs)}

    num_genes = len(valid_genes)
    num_tfs = len(valid_tfs)

    if num_genes == 0 or num_tfs == 0:
        raise ValueError(
            f"No matched genes ({num_genes}) or TFs ({num_tfs}). "
            "Ensure gene names in expression match motif targets."
        )

    # Build motif matrix (TFs x genes) using vectorized indexing
    motif_matrix = np.zeros((num_tfs, num_genes))

    if isinstance(tf_motifs, pd.DataFrame) and tf_motifs.shape[1] >= 3:
        tfs_col = tf_motifs.iloc[:, 0].values
        targets_col = tf_motifs.iloc[:, 1].values
        weights_col = tf_motifs.iloc[:, 2].to_numpy(dtype=float)
    elif isinstance(tf_motifs, np.ndarray) and tf_motifs.ndim == 2 and tf_motifs.shape[1] >= 3:
        tfs_col = tf_motifs[:, 0]
        targets_col = tf_motifs[:, 1]
        weights_col = tf_motifs[:, 2].astype(float)
    else:
        tfs_col = targets_col = weights_col = np.array([])

    if len(tfs_col) > 0:
        tf_idx = np.array([valid_tf_to_idx.get(t, -1) for t in tfs_col])
        gene_idx = np.array([valid_gene_to_idx.get(g, -1) for g in targets_col])
        valid_mask = (tf_idx >= 0) & (gene_idx >= 0)
        motif_matrix[tf_idx[valid_mask], gene_idx[valid_mask]] = weights_col[valid_mask]

    # Filter expression to valid genes
    if gene_names is not None:
        valid_gene_indices = [gene_to_idx[name] for name in valid_genes if name in gene_to_idx]
        expr_filtered = expression_matrix[valid_gene_indices, :]
    else:
        expr_filtered = expression_matrix[:num_genes, :]

    # Optional randomization of the expression data (null models)
    randomize = "None" if randomize is None else str(randomize).replace("_", ".")
    if randomize not in ("None", "within.gene", "by.gene"):
        raise ValueError(
            "randomization_method must be one of None, 'within.gene', 'by.gene'"
        )
    if randomize == "within.gene":
        expr_filtered = np.array([np.random.permutation(row) for row in expr_filtered])
        if verbose:
            print("Randomizing by reordering each gene's expression")
    elif randomize == "by.gene":
        expr_filtered = expr_filtered[np.random.permutation(expr_filtered.shape[0]), :]
        if verbose:
            print("Randomizing by reordering each gene labels")

    # Compute gene co-regulation (genes x genes)
    num_conditions = expr_filtered.shape[1]

    if num_conditions < 3:
        if verbose:
            print("Not enough conditions for correlation. Using identity matrix.")
        gene_coreg = np.eye(num_genes)
    else:
        gene_coreg = compute_correlation(expr_filtered, method=correlation_method)
        if scale_by_present:
            present = (np.asarray(expr_filtered) > 0).astype(float)
            gene_coreg = gene_coreg * ((present @ present.T) / num_conditions)
            if np.isnan(gene_coreg).any():
                np.fill_diagonal(gene_coreg, 1.0)
                gene_coreg = np.nan_to_num(gene_coreg, nan=0.0)
        else:
            np.fill_diagonal(gene_coreg, 1.0)
            gene_coreg = np.nan_to_num(gene_coreg, nan=0.0)

    # Build PPI matrix (TFs x TFs) using vectorized indexing
    if ppi_network is None:
        tf_coop = np.eye(num_tfs)
    else:
        tf_coop = np.zeros((num_tfs, num_tfs))

        if isinstance(ppi_network, pd.DataFrame) and ppi_network.shape[1] >= 3:
            p1_col = ppi_network.iloc[:, 0].values
            p2_col = ppi_network.iloc[:, 1].values
            w_col = ppi_network.iloc[:, 2].to_numpy(dtype=float)
        elif isinstance(ppi_network, np.ndarray) and ppi_network.ndim == 2 and ppi_network.shape[1] >= 3:
            p1_col = ppi_network[:, 0]
            p2_col = ppi_network[:, 1]
            w_col = ppi_network[:, 2].astype(float)
        else:
            p1_col = p2_col = w_col = np.array([])

        if len(p1_col) > 0:
            p1_idx = np.array([valid_tf_to_idx.get(p, -1) for p in p1_col])
            p2_idx = np.array([valid_tf_to_idx.get(p, -1) for p in p2_col])
            valid_mask = (p1_idx >= 0) & (p2_idx >= 0)
            tf_coop[p1_idx[valid_mask], p2_idx[valid_mask]] = w_col[valid_mask]

        # Symmetrize: average bidirectional edges, keep unidirectional
        both_present = (tf_coop != 0) & (tf_coop.T != 0)
        tf_coop = tf_coop + tf_coop.T
        tf_coop[both_present] = tf_coop[both_present] / 2.0
        np.fill_diagonal(tf_coop, 1.0)

    # Normalize all three networks (dual row+column z-score)
    if verbose:
        print("Normalizing networks")
    regulatory_network = normalize_network_panda(motif_matrix)
    tf_coop_network = normalize_network_panda(tf_coop)
    gene_coreg_network = normalize_network_panda(gene_coreg)

    # PANDA message-passing loop
    if verbose:
        print("Learning Network")
        print("Using tanimoto similarity")

    complement_alpha = 1.0 - alpha
    step = 0
    hamming_cur = 1.0

    while hamming_cur > hamming_threshold:
        if step >= max_iterations:
            if verbose:
                print(f"Reached maximum iterations ({max_iterations})")
            break

        # Responsibility: tanimoto(tfCoopNetwork, regulatoryNetwork)
        responsibility = tanimoto(tf_coop_network, regulatory_network)

        # Availability: tanimoto(regulatoryNetwork, geneCoreg)
        availability = tanimoto(regulatory_network, gene_coreg_network)

        # Combined message
        combined = 0.5 * (responsibility + availability)

        hamming_cur = hamming_distance(regulatory_network, combined, num_tfs, num_genes)

        # Update regulatory network
        regulatory_network = complement_alpha * regulatory_network + alpha * combined

        # Update TF cooperation network
        ppi_new = tanimoto(regulatory_network, regulatory_network.T)
        ppi_new = update_diagonal(ppi_new, num_tfs, alpha, step)
        tf_coop_network = complement_alpha * tf_coop_network + alpha * ppi_new

        # Update gene co-regulation network
        coreg_new = tanimoto(regulatory_network.T, regulatory_network)
        coreg_new = update_diagonal(coreg_new, num_genes, alpha, step)
        gene_coreg_network = complement_alpha * gene_coreg_network + alpha * coreg_new

        if verbose:
            print(f"Iteration {step}: hamming distance = {hamming_cur:.5f}")

        step += 1

    if verbose:
        print(f"Successfully ran PANDA on {num_genes} Genes and {num_tfs} TFs")

    # Set dimension names for tracking
    regulatory_network = np.asarray(regulatory_network, dtype=float)
    gene_coreg_network = np.asarray(gene_coreg_network, dtype=float)
    tf_coop_network = np.asarray(tf_coop_network, dtype=float)

    return (
        regulatory_network,     # (TFs x genes)
        gene_coreg_network,     # (genes x genes)
        tf_coop_network,        # (TFs x TFs)
        valid_tfs,              # TF names
        valid_genes,            # gene names
    )


def integrate_networks(
    regulatory_network: np.ndarray,
    gene_coexpression: np.ndarray,
    tf_cooperation: np.ndarray,
    z_scaling: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-process PANDA output networks.

    Matches R's prepResult():
    - If zScale=TRUE: return raw converged networks (no transformation)
    - If zScale=FALSE: apply pnorm() (normal CDF) to all networks

    Parameters
    ----------
    regulatory_network : ndarray
        TF -> gene regulatory weights (TFs x genes).
    gene_coexpression : ndarray
        Gene co-regulation weights (genes x genes).
    tf_cooperation : ndarray
        TF cooperation weights (TFs x TFs).
    z_scaling : bool, default=True
        If True, return raw networks. If False, apply pnorm().

    Returns
    -------
    reg_net : ndarray
    coexp_net : ndarray
    coop_net : ndarray
    """
    if z_scaling:
        # R: if zScale=TRUE, return raw networks
        return regulatory_network, gene_coexpression, tf_cooperation
    else:
        # R: if zScale=FALSE, apply pnorm() (CDF of standard normal)
        reg_net = scipy_norm.cdf(regulatory_network)
        coexp_net = scipy_norm.cdf(gene_coexpression)
        coop_net = scipy_norm.cdf(tf_cooperation)
        return reg_net, coexp_net, coop_net
