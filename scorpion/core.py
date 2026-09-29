"""
Core SCORPION functions: scorpion() and run_scorpion().
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, csc_matrix
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor
from typing import Union, Optional, List, Tuple, Dict
import warnings

from .preprocessing import (
    validate_inputs, log_normalize_data,
    remove_batch_effect as _remove_batch_effect,
)
from .graph_utils import make_super_cells
from .panda import run_panda, integrate_networks
from .utils import set_random_seed


def scorpion(
    gex_matrix: Union[np.ndarray, csr_matrix, csc_matrix, pd.DataFrame],
    tf_motifs: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ppi_net: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    computing_engine: str = "cpu",
    n_cores: int = 1,
    gamma_value: int = 10,
    n_pc: int = 25,
    assoc_method: str = "pearson",
    alpha_value: float = 0.1,
    hamming_value: float = 0.001,
    n_iter: float = np.inf,
    out_net: Union[str, List[str]] = ["regNet", "coregNet", "coopNet"],
    z_scaling: bool = True,
    show_progress: bool = True,
    randomization_method: Optional[str] = None,
    scale_by_present: bool = False,
    filter_expr: bool = False,
    random_state: Optional[int] = None,
    gene_names: Optional[np.ndarray] = None,
) -> Union[Dict, pd.DataFrame]:
    """
    Build gene regulatory networks from single-cell RNA-seq data using PANDA.
    
    Constructs gene regulatory networks from single-cell/nuclei RNA-seq data by
    first applying coarse-graining to reduce sparsity, then running the PANDA
    (Passing Attributes between Networks for Data Assimilation) message-passing
    algorithm to integrate transcription factor motifs, protein-protein
    interactions, and gene expression data into unified regulatory networks.
    
    Parameters
    ----------
    gex_matrix : {ndarray, csr_matrix, csc_matrix, DataFrame}
        An expression dataset, with genes in the rows and barcodes (cells)
        in the columns.
    tf_motifs : {ndarray, DataFrame}, optional
        A motif dataset (DataFrame or matrix) with 3 columns: TF, target
        gene, and motif score. Pass None (together with ``ppi_net=None``)
        for co-expression analysis only.
    ppi_net : {ndarray, DataFrame}, optional
        A Protein-Protein-Interaction dataset (DataFrame or matrix) with
        3 columns: protein 1, protein 2, and interaction score. Pass None
        to disable protein interaction integration.
    computing_engine : str, default="cpu"
        Character specifying computing device: 'cpu' or 'gpu'. GPU
        computing is not implemented in pySCORPION; 'gpu' falls back to
        CPU with a warning (as R does when gpuR is unavailable).
    n_cores : int, default=1
        Number of processors to be used if BLAS or MPI is active.
    gamma_value : int, default=10
        Graining level of data (proportion of number of single cells in
        the initial dataset to the number of super-cells in the final
        dataset).
    n_pc : int, default=25
        Number of principal components to use for construction of
        single-cell kNN network.
    assoc_method : str, default="pearson"
        Association method. Must be one of 'pearson', 'spearman'
        or 'pcNet'.
    alpha_value : float, default=0.1
        Numeric update parameter (0 to 1) controlling relative
        contribution of prior networks. Default 0.1.
    hamming_value : float, default=0.001
        Numeric convergence threshold based on Hamming distance.
        Algorithm stops when updates fall below this. Default 0.001.
    n_iter : float, default=inf
        Sets the maximum number of iterations PANDA can run before
        exiting.
    out_net : {str, list}, default=["regNet", "coregNet", "coopNet"]
        A list containing which networks to return. Options include
        "regNet", "coregNet", "coopNet".
    z_scaling : bool, default=True
        Boolean to indicate use of Z-Scores in output. False will use
        [0,1] scale.
    show_progress : bool, default=True
        Boolean to indicate printing of output for algorithm progress.
    randomization_method : str, optional
        Method by which to randomize gene expression matrix.
        Default None (no randomization; ``"None"`` is also accepted).
        Options: ``"within.gene"`` (scrambles each row of the gene
        expression matrix), ``"by.gene"`` (scrambles gene labels).
        Underscore spellings (``"within_gene"``, ``"by_gene"``) are also
        accepted.
    scale_by_present : bool, default=False
        Boolean to indicate scaling of correlations by percentage of
        positive samples.
    filter_expr : bool, default=False
        Boolean to remove genes with zero expression across all cells
        before network inference. Default False.
    random_state : int, optional
        Random seed for reproducibility.
    gene_names : ndarray, optional
        Gene name labels for the rows of gex_matrix.
    
    Returns
    -------
    result : dict or DataFrame
        If both *tf_motifs* and *ppi_net* are None, the gene-gene
        co-expression matrix of the super-cells (DataFrame indexed by
        gene). Otherwise a dictionary with up to 6+2 elements describing the inferred
        networks at convergence:
        
        - "regNet": Regulatory network matrix (TFs × genes)
        - "coregNet": Co-regulation network matrix (genes × genes)
        - "coopNet": Cooperation network matrix (TFs × TFs)
        - "numGenes": Number of genes in the network
        - "numTFs": Number of transcription factors
        - "numEdges": Total number of edges in regulatory network
        - "geneNames": Gene name labels
        - "tfNames": TF name labels
    
    See Also
    --------
    run_scorpion : For building networks across cell groups.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scorpion import scorpion
    >>>
    >>> # Create toy data
    >>> gex = np.random.poisson(5, (1000, 100))  # 1000 genes, 100 cells
    >>> motifs = pd.DataFrame({
    ...     "tf": ["TF1", "TF2", "TF1"],
    ...     "target": ["gene1", "gene2", "gene3"],
    ...     "weight": [0.8, 0.7, 0.6]
    ... })
    >>>
    >>> net = scorpion(gex, motifs, gamma_value=10, n_pc=10)
    >>> print(net["regNet"].shape)
    >>> print(f"Edges: {net['numEdges']}")
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _scorpion_impl(
            gex_matrix=gex_matrix, tf_motifs=tf_motifs, ppi_net=ppi_net,
            computing_engine=computing_engine, n_cores=n_cores,
            gamma_value=gamma_value, n_pc=n_pc, assoc_method=assoc_method,
            alpha_value=alpha_value, hamming_value=hamming_value, n_iter=n_iter,
            out_net=out_net, z_scaling=z_scaling, show_progress=show_progress,
            randomization_method=randomization_method,
            scale_by_present=scale_by_present, filter_expr=filter_expr,
            random_state=random_state, gene_names=gene_names,
        )


def _scorpion_impl(
    gex_matrix, tf_motifs, ppi_net, computing_engine, n_cores,
    gamma_value, n_pc, assoc_method, alpha_value, hamming_value, n_iter,
    out_net, z_scaling, show_progress, randomization_method,
    scale_by_present, filter_expr, random_state, gene_names,
):
    """Internal implementation of scorpion(), called inside a warnings context."""
    if random_state is not None:
        set_random_seed(random_state)

    # Validate parameters (matches R's scorpion())
    if not isinstance(alpha_value, (int, float)) or not (0 <= alpha_value <= 1):
        raise ValueError("alpha_value must be a numeric value between 0 and 1")
    if not isinstance(hamming_value, (int, float)) or hamming_value < 0:
        raise ValueError("hamming_value must be a non-negative numeric value")
    if assoc_method not in ("pearson", "spearman", "pcNet"):
        raise ValueError("assoc_method must be one of: 'pearson', 'spearman', 'pcNet'")
    if computing_engine not in ("cpu", "gpu"):
        raise ValueError("computing_engine must be either 'cpu' or 'gpu'")
    if gamma_value <= 0:
        raise ValueError(f"gamma_value must be positive, got {gamma_value}")
    if n_pc <= 0:
        raise ValueError(f"n_pc must be positive, got {n_pc}")
    if gex_matrix.shape[1] < 30:
        warnings.warn(
            "gex_matrix has fewer than 30 cells. Network inference may be unreliable.",
            UserWarning,
        )
    if gex_matrix.shape[0] < 10:
        raise ValueError("gex_matrix must have at least 10 genes (rows)")
    if computing_engine == "gpu":
        warnings.warn(
            "GPU computing is not available in pySCORPION. Falling back to CPU computing.",
            UserWarning,
        )
        computing_engine = "cpu"

    # Convert output network specification
    if isinstance(out_net, str):
        out_net = [out_net]

    # Convert expression matrix to numpy array and extract gene names
    _gene_names = gene_names
    if issparse(gex_matrix):
        gex_array = gex_matrix.toarray()
    elif isinstance(gex_matrix, pd.DataFrame):
        if _gene_names is None:
            _gene_names = gex_matrix.index.values
        gex_array = gex_matrix.to_numpy(dtype=float)
    else:
        gex_array = np.asarray(gex_matrix, dtype=float)
    gene_names = None if _gene_names is None else np.asarray(_gene_names)

    n_genes, n_cells = gex_array.shape

    if show_progress:
        print(f"SCORPION: {n_genes} genes x {n_cells} cells")

    # R: gexMatrix <- gexMatrix[rowSums(gexMatrix) > 0, ]
    if filter_expr:
        keep = gex_array.sum(axis=1) > 0
        gex_array = gex_array[keep, :]
        if gene_names is not None:
            gene_names = gene_names[keep]
        n_genes = gex_array.shape[0]
        if show_progress:
            print(f"  After filtering: {n_genes} genes")

    # R's scorpion() does NOT log-normalize before makeSuperCells.
    # Raw expression goes directly to makeSuperCells.

    # Create super-cells
    if show_progress:
        print(f"Creating super-cells with gamma={gamma_value}...")

    super_cell_expr, super_cell_assignment, _, super_cell_names = make_super_cells(
        gex_array,
        gamma=gamma_value,
        n_pcs=n_pc,
        k_nn=5,
        random_state=random_state,
        gene_names=gene_names,
    )

    n_super_cells = super_cell_expr.shape[1]
    if show_progress:
        print(f"  Created {n_super_cells} super-cells")

    # Co-expression only: no priors, return the gene-gene correlation matrix
    if tf_motifs is None and ppi_net is None:
        expr = np.asarray(super_cell_expr, dtype=float)
        if assoc_method == "spearman":
            expr = np.apply_along_axis(rankdata, 1, expr)
        centered = expr - expr.mean(axis=1, keepdims=True)
        centered = centered / np.sqrt((centered ** 2).sum(axis=1, keepdims=True))
        coexpr = centered @ centered.T
        labels = gene_names if gene_names is not None else np.arange(coexpr.shape[0])
        return pd.DataFrame(coexpr, index=labels, columns=labels)

    # Run PANDA on super-cells
    if show_progress:
        print(f"Running PANDA (alpha={alpha_value})...")

    panda_result = run_panda(
        super_cell_expr,
        tf_motifs,
        ppi_network=ppi_net,
        alpha=alpha_value,
        hamming_threshold=hamming_value,
        max_iterations=int(n_iter) if n_iter != np.inf else 10000,
        correlation_method=assoc_method,
        gene_names=gene_names,
        verbose=show_progress,
        randomize=randomization_method,
        scale_by_present=scale_by_present,
    )

    # run_panda returns (regNet, coregNet, coopNet, tf_names, gene_names)
    reg_net, coexp_net, coop_net, result_tf_names, result_gene_names = panda_result

    # Apply prepResult: zScale=TRUE returns raw, zScale=FALSE applies pnorm
    if show_progress:
        print(f"Preparing results (z_scaling={z_scaling})...")

    reg_net, coexp_net, coop_net = integrate_networks(
        reg_net, coexp_net, coop_net, z_scaling=z_scaling
    )

    # Count edges (R: sum(apply(regulatoryNetwork, 1, function(X) { X != 0 })))
    num_edges = int(np.sum(reg_net != 0))
    num_genes_out = reg_net.shape[1]  # R: regNet is (TFs x genes), ncol = genes
    num_tfs_out = reg_net.shape[0]    # R: regNet is (TFs x genes), nrow = TFs

    # Collect results
    result = {}

    if "regNet" in out_net:
        result["regNet"] = reg_net          # (TFs x genes), matching R
    if "coregNet" in out_net:
        result["coregNet"] = coexp_net      # (genes x genes)
    if "coopNet" in out_net:
        result["coopNet"] = coop_net        # (TFs x TFs)

    # Add metadata (matches R return: numGenes, numTFs, numEdges)
    result["numGenes"] = num_genes_out
    result["numTFs"] = num_tfs_out
    result["numEdges"] = num_edges

    # Store names for later use (Python-specific)
    result["geneNames"] = result_gene_names
    result["tfNames"] = result_tf_names

    if show_progress:
        print(f"Complete: {num_edges} edges ({num_tfs_out} TFs x {num_genes_out} Genes)")

    return result


def run_scorpion(
    gex_matrix: Union[np.ndarray, csr_matrix, csc_matrix, pd.DataFrame],
    tf_motifs: Union[np.ndarray, pd.DataFrame],
    ppi_net: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    cells_metadata: pd.DataFrame = None,
    group_by: Union[str, List[str]] = None,
    normalize_data: bool = True,
    remove_batch_effect: bool = False,
    batch: Optional[Union[str, np.ndarray, pd.Series, list]] = None,
    min_cells: int = 30,
    computing_engine: str = "cpu",
    n_cores: int = 1,
    gamma_value: int = 10,
    n_pc: int = 25,
    assoc_method: str = "pearson",
    alpha_value: float = 0.1,
    hamming_value: float = 0.001,
    n_iter: float = np.inf,
    out_net: Union[str, List[str]] = "regNet",
    z_scaling: bool = True,
    show_progress: bool = True,
    randomization_method: Optional[str] = None,
    scale_by_present: bool = False,
    filter_expr: bool = False,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run SCORPION across cell groups and return combined networks.
    
    Builds per-group regulatory networks by running :func:`scorpion` on
    subsets of cells defined by *cells_metadata* and combining the
    resulting networks into a wide-format DataFrame where each column
    corresponds to a network.
    
    This function is a wrapper around :func:`scorpion` that groups cells
    according to metadata columns, filters out groups with insufficient
    cells, runs network inference on each remaining group independently,
    and finally combines all resulting networks into a single wide-format
    DataFrame.
    
    Parameters
    ----------
    gex_matrix : {ndarray, csr_matrix, csc_matrix, DataFrame}
        An expression dataset with genes in the rows and barcodes (cells)
        in the columns.
    tf_motifs : {ndarray, DataFrame}
        A motif dataset, a DataFrame or a matrix containing 3 columns.
        Each row describes a motif associated with a transcription factor
        (column 1) a gene (column 2) and a score (column 3).
    ppi_net : {ndarray, DataFrame}, optional
        A Protein-Protein-Interaction dataset, a DataFrame or matrix
        containing 3 columns. Each row describes a protein-protein
        interaction between transcription factor 1 (column 1),
        transcription factor 2 (column 2) and a score (column 3).
    cells_metadata : DataFrame
        A DataFrame with cell-level metadata; must contain columns
        specified in *group_by*.
    group_by : {str, list}, optional
        Character or list of one or more column names in *cells_metadata*
        to use for grouping cells into networks.
    normalize_data : bool, default=True
        Boolean to indicate normalization of expression data.
        Default True performs log normalization.
    remove_batch_effect : bool, default=False
        Boolean to indicate batch effect correction. Default False.
    batch : {str, array-like}, optional
        Batch assignment for each cell, given either as a vector with one
        entry per cell (as in R) or as a column name in *cells_metadata*;
        required if ``remove_batch_effect=True``. After correction the
        per-gene median expression is added back, as in R. With fewer
        than two batch levels, correction is skipped with a warning.
    min_cells : int, default=30
        Minimum number of cells per group required to build a network.
        Values below 30 are raised to 30, as in R. Default is 30.
    computing_engine : str, default="cpu"
        Either 'cpu' or 'gpu'. Passed to :func:`scorpion`.
    n_cores : int, default=1
        Number of worker processes used to build the per-group networks
        in parallel. When greater than 1, call this from within an
        ``if __name__ == "__main__":`` block in scripts.
    gamma_value : int, default=10
        Graining level of data (proportion of number of single cells to
        super-cells). Default 10.
    n_pc : int, default=25
        Number of principal components to use for kNN network
        construction. Default 25.
    assoc_method : str, default="pearson"
        Association method. Must be one of 'pearson', 'spearman'
        or 'pcNet'. Default 'pearson'.
    alpha_value : float, default=0.1
        Value to be used for update variable in PANDA. Default 0.1.
    hamming_value : float, default=0.001
        Value at which to terminate the process based on Hamming
        distance. Default 0.001.
    n_iter : float, default=inf
        Sets the maximum number of iterations PANDA can run before
        exiting. Default inf.
    out_net : {str, list}, default="regNet"
        Which network(s) to extract. Options include "regNet",
        "coregNet", "coopNet". Default "regNet". When more than one
        network is requested, an ``edge_type`` column ("tf-target",
        "gene-gene", "tf-tf") is added and the networks are stacked in
        long format.
    z_scaling : bool, default=True
        Boolean to indicate use of Z-Scores in output. False will
        use [0,1] scale. Default True.
    show_progress : bool, default=True
        Boolean to indicate printing of output for algorithm progress.
        Default True.
    randomization_method : str, optional
        Method by which to randomize gene expression matrix.
        Default None (no randomization). Options: ``"within.gene"``
        or ``"by.gene"`` (underscore spellings also accepted).
    scale_by_present : bool, default=False
        Boolean to indicate scaling of correlations by percentage
        of positive samples. Default False.
    filter_expr : bool, default=False
        Boolean to indicate whether or not to remove genes with 0
        expression across all cells. Default False.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    result : DataFrame
        A DataFrame in wide format where rows represent TF-target pairs
        (union across all networks) and columns represent network
        identifiers. Cell values are edge weights from the corresponding
        network; pairs absent from a group's network are NaN.
        
        - Column "edge_type" (only when several networks are requested)
        - Column "tf": source transcription factor name
        - Column "target": target gene name
        - Remaining columns: one per cell group, named by group ID
          (groups joined with "--" when grouping by multiple columns),
          in order of first appearance in *cells_metadata*
    
    See Also
    --------
    scorpion : For building a single network.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from scorpion import run_scorpion
    >>>
    >>> # Simulate data
    >>> gex = np.random.poisson(5, (1000, 200))
    >>> metadata = pd.DataFrame({
    ...     "cell_id": [f"cell_{i}" for i in range(200)],
    ...     "cell_type": ["TypeA"] * 100 + ["TypeB"] * 100
    ... })
    >>> motifs = pd.DataFrame({
    ...     "tf": ["TF1"] * 50,
    ...     "target": [f"gene_{i}" for i in range(50)],
    ...     "weight": np.random.uniform(0.5, 1.0, 50)
    ... })
    >>>
    >>> networks = run_scorpion(
    ...     gex, motifs, cells_metadata=metadata, group_by="cell_type"
    ... )
    >>> print(networks.shape)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _run_scorpion_impl(
            gex_matrix=gex_matrix, tf_motifs=tf_motifs, ppi_net=ppi_net,
            cells_metadata=cells_metadata, group_by=group_by,
            normalize_data=normalize_data,
            remove_batch_effect=remove_batch_effect, batch=batch,
            min_cells=min_cells, computing_engine=computing_engine,
            n_cores=n_cores, gamma_value=gamma_value, n_pc=n_pc,
            assoc_method=assoc_method, alpha_value=alpha_value,
            hamming_value=hamming_value, n_iter=n_iter, out_net=out_net,
            z_scaling=z_scaling, show_progress=show_progress,
            randomization_method=randomization_method,
            scale_by_present=scale_by_present, filter_expr=filter_expr,
            random_state=random_state,
        )


_EDGE_TYPE_LABELS = {"regNet": "tf-target", "coregNet": "gene-gene", "coopNet": "tf-tf"}


def _network_labels(result: Dict, net_name: str) -> Tuple[list, list]:
    """Row and column labels of one network returned by scorpion()."""
    tfs = list(result["tfNames"])
    genes = list(result["geneNames"])
    if net_name == "coregNet":
        return genes, genes
    if net_name == "coopNet":
        return tfs, tfs
    return tfs, genes


def _combine_networks(results: List[Dict], network_ids: List[str], net_name: str) -> pd.DataFrame:
    """
    Combine per-group networks of one type into a wide DataFrame.

    Matches R's runSCORPION(): the tf/target grid is built from the union of
    all networks' dimnames (tf varies fastest, as in expand.grid), and edges
    absent from a group's network are NaN.
    """
    labels = [_network_labels(r, net_name) for r in results]
    all_rows = list(dict.fromkeys(x for rows, _ in labels for x in rows))
    all_cols = list(dict.fromkeys(x for _, cols in labels for x in cols))
    row_pos = {name: i for i, name in enumerate(all_rows)}
    col_pos = {name: j for j, name in enumerate(all_cols)}
    n_rows, n_cols = len(all_rows), len(all_cols)

    weights = np.full((n_rows * n_cols, len(results)), np.nan)
    for k, (r, (rows, cols)) in enumerate(zip(results, labels)):
        net = np.asarray(r[net_name], dtype=float)
        ri = np.array([row_pos[x] for x in rows])
        ci = np.array([col_pos[x] for x in cols])
        # Column-major (R) linear index: row + col * n_rows
        global_idx = ri[:, None] + ci[None, :] * n_rows
        weights[global_idx.ravel(), k] = net.ravel()

    grid_tf = np.tile(np.asarray(all_rows, dtype=object), n_cols)
    grid_target = np.repeat(np.asarray(all_cols, dtype=object), n_rows)
    out = pd.DataFrame(weights, columns=network_ids)
    out.insert(0, "target", grid_target)
    out.insert(0, "tf", grid_tf)
    return out


def _run_scorpion_impl(
    gex_matrix, tf_motifs, ppi_net, cells_metadata, group_by,
    normalize_data, remove_batch_effect, batch, min_cells,
    computing_engine, n_cores, gamma_value, n_pc, assoc_method,
    alpha_value, hamming_value, n_iter, out_net, z_scaling,
    show_progress, randomization_method, scale_by_present,
    filter_expr, random_state,
) -> pd.DataFrame:
    """Internal implementation of run_scorpion(), called inside a warnings context."""
    if random_state is not None:
        set_random_seed(random_state)

    if cells_metadata is None:
        raise ValueError("cells_metadata must be provided")
    if gex_matrix.shape[1] != cells_metadata.shape[0]:
        raise ValueError("gex_matrix must have the same number of columns as cells_metadata has rows")

    # batch may be a column name of cells_metadata or a per-cell vector (as in R)
    batch_is_column = isinstance(batch, str)
    valid, msg = validate_inputs(
        gex_matrix, tf_motifs, cells_metadata, ppi_net,
        group_by, alpha_value, gamma_value, n_pc,
        batch if batch_is_column else None,
    )
    if not valid:
        raise ValueError(f"Input validation failed: {msg}")
    if not isinstance(min_cells, (int, float, np.integer, np.floating)) or min_cells < 1:
        raise ValueError("min_cells must be a positive integer")
    if remove_batch_effect and batch is None:
        raise ValueError("batch must be provided when remove_batch_effect=True")
    if batch is not None:
        batch_labels = (
            cells_metadata[batch].to_numpy() if batch_is_column else np.asarray(batch)
        )
        if len(batch_labels) != gex_matrix.shape[1]:
            raise ValueError("batch must have the same length as gex_matrix columns (cells)")

    out_net_list = [out_net] if isinstance(out_net, str) else list(out_net)
    valid_nets = ("regNet", "coregNet", "coopNet")
    if len(out_net_list) < 1 or not all(n in valid_nets for n in out_net_list):
        raise ValueError(f"out_net must be one or more of: {', '.join(valid_nets)}")

    # Convert expression matrix
    gene_names = None
    if issparse(gex_matrix):
        gex_array = gex_matrix.toarray().astype(float)
    elif isinstance(gex_matrix, pd.DataFrame):
        gene_names = gex_matrix.index.values
        gex_array = gex_matrix.to_numpy(dtype=float)
    else:
        gex_array = np.asarray(gex_matrix, dtype=float)

    n_genes, n_cells = gex_array.shape

    if show_progress:
        print(f"runSCORPION: {n_genes} genes x {n_cells} cells")

    # Normalize if requested
    if normalize_data:
        if show_progress:
            print("Normalizing data (log scale)")
        gex_array = log_normalize_data(gex_array)

    # Remove batch effects if requested. As in R, the per-gene median is
    # added back after the batch-effect residualisation.
    if remove_batch_effect:
        if len(pd.unique(pd.Series(batch_labels).dropna())) < 2:
            # A single batch leaves nothing to correct; skip so the per-gene
            # median is not added to uncorrected data.
            warnings.warn(
                "batch has fewer than two levels; skipping batch effect correction",
                UserWarning,
            )
        else:
            if show_progress:
                print("Correcting for batch effects")
            mean_expr = np.median(gex_array, axis=1, keepdims=True)
            gex_array = _remove_batch_effect(gex_array, batch_labels)
            gex_array = gex_array + mean_expr

    # Setting min number of cells to construct network (R: max(minCells, 30))
    min_cells_eff = max(min_cells, 30)

    # Determine cell groups
    if group_by is None:
        group_ids = np.array(["all"] * n_cells, dtype=object)
    else:
        group_by_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        group_ids = (
            cells_metadata[group_by_cols].astype(str)
            .agg("--".join, axis=1).to_numpy()
        )

    # Groups in order of first appearance, filtered by size (as in R)
    counts = pd.Series(group_ids).value_counts(sort=False)
    all_groups = list(pd.unique(group_ids))
    network_ids = [g for g in all_groups if counts[g] >= min_cells_eff]

    if show_progress:
        print(f"{len(all_groups)} networks requested")
        print(f"{len(network_ids)} networks meet the minimum cell requirement ({min_cells_eff})")

    if not network_ids:
        raise ValueError(
            f"No groups have enough cells (>= {min_cells_eff}) to build a network"
        )

    scorpion_kwargs = dict(
        tf_motifs=tf_motifs,
        ppi_net=ppi_net,
        computing_engine=computing_engine,
        n_cores=1,
        gamma_value=gamma_value,
        n_pc=n_pc,
        assoc_method=assoc_method,
        alpha_value=alpha_value,
        hamming_value=hamming_value,
        n_iter=n_iter,
        out_net=out_net_list,
        z_scaling=z_scaling,
        show_progress=False,
        randomization_method=randomization_method,
        scale_by_present=scale_by_present,
        filter_expr=filter_expr,
        random_state=random_state,
        gene_names=gene_names,
    )
    gex_chunks = [gex_array[:, group_ids == nid] for nid in network_ids]
    del gex_array

    if show_progress:
        print(f"Computing {len(network_ids)} networks")

    if n_cores > 1:
        if show_progress:
            print(f"Using {n_cores} cores for parallel processing")
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            futures = [
                pool.submit(scorpion, chunk, **scorpion_kwargs) for chunk in gex_chunks
            ]
            results = [f.result() for f in futures]
    else:
        results = []
        for i, (nid, chunk) in enumerate(zip(network_ids, gex_chunks), start=1):
            if show_progress:
                print(f"Network {i}/{len(network_ids)}: {nid}")
            results.append(scorpion(chunk, **scorpion_kwargs))
    del gex_chunks

    if show_progress:
        print("Networks successfully constructed")

    if len(out_net_list) == 1:
        # Single network type: wide format with tf, target and one column per group
        networks = _combine_networks(results, network_ids, out_net_list[0])
    else:
        # Multiple network types: stack in long format with an edge_type column
        per_type = []
        for net_name in out_net_list:
            block = _combine_networks(results, network_ids, net_name)
            block.insert(0, "edge_type", _EDGE_TYPE_LABELS[net_name])
            per_type.append(block)
        networks = pd.concat(per_type, ignore_index=True)

    if show_progress:
        print("Networks successfully combined")

    return networks
