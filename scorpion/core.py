"""
Core SCORPION functions: scorpion() and run_scorpion().
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, csc_matrix
from typing import Union, Optional, List, Tuple, Dict
import warnings

from .preprocessing import (
    validate_inputs, log_normalize_data,
    remove_batch_effect as _remove_batch_effect, filter_expression
)
from .graph_utils import make_super_cells, aggregate_by_groups
from .panda import run_panda, integrate_networks
from .utils import (
    matrix_to_edge_list, restrict_networks, normalize_network, set_random_seed
)


def scorpion(
    gex_matrix: Union[np.ndarray, csr_matrix, csc_matrix, pd.DataFrame],
    tf_motifs: Union[np.ndarray, pd.DataFrame],
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
) -> Dict:
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
    tf_motifs : {ndarray, DataFrame}
        A motif dataset (DataFrame or matrix) with 3 columns: TF, target
        gene, and motif score. Pass None for co-expression analysis only.
    ppi_net : {ndarray, DataFrame}, optional
        A Protein-Protein-Interaction dataset (DataFrame or matrix) with
        3 columns: protein 1, protein 2, and interaction score. Pass None
        to disable protein interaction integration.
    computing_engine : str, default="cpu"
        Character specifying computing device: 'cpu' or 'gpu'
        (if available). Default 'cpu'.
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
        Default None (no randomization). Options: ``"within_gene"``
        (scrambles each row of the gene expression matrix),
        ``"by_gene"`` (scrambles gene labels).
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
    result : dict
        A dictionary with up to 6+2 elements describing the inferred
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
    if random_state is not None:
        set_random_seed(random_state)
    
    # Validate parameters
    if not (0 <= alpha_value <= 1):
        raise ValueError(f"alpha_value must be in [0, 1], got {alpha_value}")
    if gamma_value <= 0:
        raise ValueError(f"gamma_value must be positive, got {gamma_value}")
    if n_pc <= 0:
        raise ValueError(f"n_pc must be positive, got {n_pc}")
    
    # Convert output network specification
    if isinstance(out_net, str):
        out_net = [out_net]
    
    # Convert expression matrix to numpy array and extract gene names
    _gene_names = gene_names
    if issparse(gex_matrix):
        gex_array = gex_matrix.toarray()
        if _gene_names is None and hasattr(gex_matrix, 'index'):
            _gene_names = gex_matrix.index.values
    elif isinstance(gex_matrix, pd.DataFrame):
        if _gene_names is None:
            _gene_names = gex_matrix.index.values
        gex_array = gex_matrix.values
    else:
        gex_array = np.asarray(gex_matrix, dtype=float)
    
    gene_names = _gene_names
    
    n_genes, n_cells = gex_array.shape
    
    if show_progress:
        print(f"SCORPION: {n_genes} genes x {n_cells} cells")
    
    # Filter expression if requested
    if filter_expr:
        gex_array = filter_expression(gex_array, min_cells=1)
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
    batch: Optional[str] = None,
    min_cells: int = 30,
    computing_engine: str = "cpu",
    n_cores: int = 1,
    gamma_value: int = 10,
    n_pc: int = 25,
    assoc_method: str = "pearson",
    alpha_value: float = 0.1,
    hamming_value: float = 0.001,
    n_iter: float = np.inf,
    out_net: str = "regNet",
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
    batch : str, optional
        Column name in *cells_metadata* giving batch assignment for
        each cell; required if ``remove_batch_effect=True``.
    min_cells : int, default=30
        Minimum number of cells per group required to build a network.
        Default is 30.
    computing_engine : str, default="cpu"
        Either 'cpu' or 'gpu'. Passed to :func:`scorpion`.
    n_cores : int, default=1
        Number of processors to be used if BLAS or MPI is active.
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
    out_net : str, default="regNet"
        Character specifying which network to extract. Options include
        "regNet", "coregNet", "coopNet". Default "regNet".
    z_scaling : bool, default=True
        Boolean to indicate use of Z-Scores in output. False will
        use [0,1] scale. Default True.
    show_progress : bool, default=True
        Boolean to indicate printing of output for algorithm progress.
        Default True.
    randomization_method : str, optional
        Method by which to randomize gene expression matrix.
        Default None (no randomization). Options: ``"within_gene"``
        or ``"by_gene"``.
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
        network.
        
        - Column "tf": source transcription factor name
        - Column "target": target gene name
        - Remaining columns: one per cell group, named by group ID
          (groups joined with "--" when grouping by multiple columns)
    
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
    if random_state is not None:
        set_random_seed(random_state)
    
    # Validate inputs
    valid, msg = validate_inputs(
        gex_matrix, tf_motifs, cells_metadata, ppi_net,
        group_by, alpha_value, gamma_value, n_pc, batch
    )
    if not valid:
        raise ValueError(f"Input validation failed: {msg}")
    
    # Convert expression matrix
    gene_names = None
    if issparse(gex_matrix):
        gex_array = gex_matrix.toarray()
    elif isinstance(gex_matrix, pd.DataFrame):
        gene_names = gex_matrix.index.values
        gex_array = gex_matrix.values
    else:
        gex_array = np.asarray(gex_matrix, dtype=float)
    
    n_genes, n_cells = gex_array.shape
    
    if show_progress:
        print(f"runSCORPION: {n_genes} genes × {n_cells} cells")
    
    # Normalize if requested
    if normalize_data:
        if show_progress:
            print("Normalizing expression...")
        gex_array = log_normalize_data(gex_array)
    
    # Remove batch effects if requested
    if remove_batch_effect and batch is not None:
        if show_progress:
            print(f"Removing batch effects ({batch})...")
        batch_labels = cells_metadata[batch].values
        gex_array = _remove_batch_effect(gex_array, batch_labels)
    
    # Determine cell groups
    if group_by is None:
        # Single network for all cells
        groups_df = pd.DataFrame({"__group__": ["all"] * n_cells})
        group_by_cols = ["__group__"]
    else:
        if isinstance(group_by, str):
            group_by_cols = [group_by]
        else:
            group_by_cols = group_by
        groups_df = cells_metadata[group_by_cols].copy()
    
    # Create group IDs
    if len(group_by_cols) == 1:
        group_ids = groups_df[group_by_cols[0]].astype(str).values
        group_names = sorted(set(group_ids))
    else:
        # Combine columns
        group_ids = groups_df[group_by_cols].astype(str).apply(
            lambda row: "--".join(row.values), axis=1
        ).values
        group_names = sorted(set(group_ids))
    
    if show_progress:
        print(f"Found {len(group_names)} groups")
    
    # Infer networks per group
    networks_dict = {}
    
    for group_name in group_names:
        group_mask = group_ids == group_name
        n_group_cells = np.sum(group_mask)
        
        if n_group_cells < min_cells:
            if show_progress:
                print(f"  Skipping {group_name} ({n_group_cells} cells < {min_cells})")
            continue
        
        if show_progress:
            print(f"  Processing {group_name} ({n_group_cells} cells)...")
        
        gex_group = gex_array[:, group_mask]
        
        # Run scorpion on this group
        result = scorpion(
            gex_group,
            tf_motifs,
            ppi_net=ppi_net,
            gamma_value=gamma_value,
            n_pc=n_pc,
            assoc_method=assoc_method,
            alpha_value=alpha_value,
            hamming_value=hamming_value,
            n_iter=n_iter,
            out_net=[out_net],  # Single network type
            z_scaling=z_scaling,
            show_progress=False,  # Suppress sub-progress
            random_state=random_state,
            gene_names=gene_names,
        )
        
        networks_dict[group_name] = result
    
    if show_progress:
        print(f"Combining results from {len(networks_dict)} groups...")
    
    # Convert networks to edge DataFrames and merge them
    edge_dfs = []

    for group_name, group_result in networks_dict.items():
        net_matrix = group_result[out_net]
        tf_name_list = group_result.get("tfNames", [f"tf_{j}" for j in range(group_result["numTFs"])])
        gene_name_list = group_result.get("geneNames", [f"gene_{i}" for i in range(group_result["numGenes"])])

        # Determine row/column names based on network type
        if out_net == "coregNet":
            row_names = gene_name_list
            col_names = gene_name_list
        elif out_net == "coopNet":
            row_names = tf_name_list
            col_names = tf_name_list
        else:  # regNet
            row_names = tf_name_list
            col_names = gene_name_list

        # Vectorized extraction of non-zero edges
        i_idx, j_idx = np.nonzero(net_matrix)
        if len(i_idx) > 0:
            df = pd.DataFrame({
                "tf": [row_names[i] for i in i_idx],
                "target": [col_names[j] for j in j_idx],
                group_name: net_matrix[i_idx, j_idx],
            })
            edge_dfs.append(df)

    # Merge all group DataFrames
    if not edge_dfs:
        result_df = pd.DataFrame(columns=["tf", "target"])
    else:
        result_df = edge_dfs[0]
        for df in edge_dfs[1:]:
            result_df = result_df.merge(df, on=["tf", "target"], how="outer")

        # Reorder columns: tf, target, then sorted groups
        group_cols = sorted([c for c in result_df.columns if c not in ["tf", "target"]])
        result_df = result_df[["tf", "target"] + group_cols]
        result_df = result_df.fillna(0.0)
    
    if show_progress:
        print(f"Results: {result_df.shape[0]} edges × {result_df.shape[1]} columns")
    
    return result_df
