"""
Graph-based utilities: kNN graph construction, community detection, super-cell creation.

Matches R SCORPION's makeSuperCells() exactly:
- Variable gene selection
- Scaling (center + unit variance, R's scale())
- PCA (prcomp with center=FALSE, scale.=FALSE on pre-scaled data)
- kNN via RANN::nn2 equivalent (k neighbors including self)
- Walktrap clustering with cut_at(k) where k=round(N/gamma)
- Aggregate original expression by cluster means
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Optional


def build_pca(
    data: np.ndarray,
    n_components: int = 25,
    random_state: Optional[int] = None,
) -> Tuple[object, np.ndarray]:
    """
    Perform PCA dimensionality reduction.

    Parameters
    ----------
    data : ndarray
        Input data (n_samples x n_features).
    n_components : int, default=25
        Number of principal components.
    random_state : int, optional
        Random seed.

    Returns
    -------
    pca : PCA
        Fitted PCA model.
    transformed : ndarray
        Transformed data (n_samples x n_components).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(data)
    return pca, transformed


def build_knn_graph(
    data: np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-nearest neighbor graph.

    Matches R's RANN::nn2(data=X, k=k): returns k neighbors INCLUDING self.
    After removing self, returns k-1 non-self neighbors (matching R behavior).

    Parameters
    ----------
    data : ndarray
        Input data (n_cells x n_features).
    k : int, default=5
        Number of nearest neighbors (including self, matching R's nn2).
    metric : str, default="euclidean"
        Distance metric.
    random_state : int, optional
        Random seed.

    Returns
    -------
    neighbor_indices : ndarray
        Non-self neighbor indices (n_cells x (k-1)).
    neighbor_distances : ndarray
        Distances to non-self neighbors (n_cells x (k-1)).
    """
    # R's nn2(data=X, k=k) returns k neighbors including self
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto").fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Remove self-loops (first neighbor is always self with distance 0)
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    return neighbor_indices, neighbor_distances


def build_knn_sparse_matrix(
    neighbor_indices: np.ndarray,
    n_nodes: int = None,
    symmetric: bool = True,
) -> csr_matrix:
    """
    Convert kNN indices to sparse adjacency matrix.

    Parameters
    ----------
    neighbor_indices : ndarray
        Neighbor indices (n_nodes x k).
    n_nodes : int, optional
        Number of nodes.
    symmetric : bool, default=True
        Make adjacency matrix symmetric.

    Returns
    -------
    adjacency : csr_matrix
        Sparse adjacency matrix (n_nodes x n_nodes).
    """
    if n_nodes is None:
        n_nodes = neighbor_indices.shape[0]

    adj = lil_matrix((n_nodes, n_nodes))

    for i in range(neighbor_indices.shape[0]):
        for j in neighbor_indices[i]:
            adj[i, j] = 1
            if symmetric:
                adj[j, i] = 1

    return adj.tocsr()


def find_communities_louvain(
    adjacency_matrix: csr_matrix,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Detect communities using Louvain algorithm (via networkx).

    Parameters
    ----------
    adjacency_matrix : csr_matrix
        Symmetric adjacency matrix.
    random_state : int, optional
        Random seed.

    Returns
    -------
    communities : ndarray
        Community assignment for each node (0-indexed).
    """
    try:
        import networkx as nx
        from networkx.algorithms import community
    except ImportError:
        raise ImportError(
            "networkx required for community detection. "
            "Install via: pip install networkx"
        )

    G = nx.from_scipy_sparse_array(adjacency_matrix)

    try:
        communities_gen = community.greedy_modularity_communities(
            G, weight=None, seed=random_state
        )
    except TypeError:
        communities_gen = community.greedy_modularity_communities(G, weight=None)

    community_map = {}
    for comm_idx, comm_nodes in enumerate(communities_gen):
        for node in comm_nodes:
            community_map[node] = comm_idx

    communities = np.array([community_map.get(i, 0) for i in range(G.number_of_nodes())])

    return communities


def find_communities_walktrap(
    adjacency_matrix: csr_matrix,
    n_clusters: int = None,
    steps: int = 4,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Detect communities using walktrap algorithm (via igraph).

    Matches R's: cluster_walktrap(graph) then cut_at(g.s, k)

    Parameters
    ----------
    adjacency_matrix : csr_matrix
        Symmetric adjacency matrix.
    n_clusters : int, optional
        Target number of clusters (for cut_at). If None, uses default.
    steps : int, default=4
        Length of random walks.
    random_state : int, optional
        Random seed (used for networkx fallback).

    Returns
    -------
    communities : ndarray
        Community assignment for each node (0-indexed).
    """
    try:
        import igraph as ig

        n = adjacency_matrix.shape[0]

        # Convert sparse matrix to igraph graph
        rows, cols = adjacency_matrix.nonzero()
        # Only keep upper triangle to avoid duplicate edges
        mask = rows < cols
        edges = list(zip(rows[mask].tolist(), cols[mask].tolist()))

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.simplify(multiple=True, loops=True)
        g.es["weight"] = 1

        # Walktrap clustering
        walktrap = g.community_walktrap(steps=steps)

        if n_clusters is not None:
            # R: cut_at(g.s, k)
            clustering = walktrap.as_clustering(n=n_clusters)
        else:
            clustering = walktrap.as_clustering()

        communities = np.array(clustering.membership)
        return communities

    except ImportError:
        # Fallback to Louvain if igraph not available
        import warnings
        warnings.warn(
            "igraph not available, falling back to Louvain clustering. "
            "Install igraph for R-equivalent results: pip install igraph"
        )
        return find_communities_louvain(adjacency_matrix, random_state=random_state)


def make_super_cells(
    gex_matrix: np.ndarray,
    gamma: int = 10,
    n_pcs: int = 25,
    k_nn: int = 5,
    random_state: Optional[int] = None,
    gene_names: Optional[np.ndarray] = None,
    cell_names: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create super-cells matching R's makeSuperCells() exactly.

    Pipeline:
    1. Filter zero-expression and zero-variance genes
    2. Select top n_var_genes variable genes (up to 1000)
    3. Transpose and scale (center + unit variance with ddof=1)
    4. PCA (without additional centering)
    5. kNN graph (k neighbors including self, matching R's nn2)
    6. Walktrap clustering with cut_at(round(N/gamma))
    7. Aggregate ALL original genes by cluster means

    Parameters
    ----------
    gex_matrix : ndarray
        Expression matrix (genes x cells).
    gamma : int, default=10
        Target aggregation level (N_cells / gamma = number of super-cells).
    n_pcs : int, default=25
        Number of PCA components.
    k_nn : int, default=5
        Number of nearest neighbors (including self, matching R).
    random_state : int, optional
        Random seed.
    gene_names : ndarray, optional
        Gene names (n_genes,).
    cell_names : ndarray, optional
        Cell names (n_cells,).

    Returns
    -------
    super_cell_expr : ndarray
        Super-cell expression matrix (genes x super_cells).
    super_cell_assignment : ndarray
        Assignment of cells to super-cells (n_cells,).
    super_cell_gene_names : ndarray, optional
        Gene names (unchanged).
    super_cell_names : ndarray, optional
        Super-cell names.
    """
    n_genes, n_cells = gex_matrix.shape

    # Store original expression for final aggregation (R: GE <- X)
    original_expr = gex_matrix.copy()

    # Step 1: Filter zero-expression genes and zero-variance genes
    # R: X <- X[rowSums(X) > 0, ]
    # R: X <- X[apply(X,1,var) > 0, ]
    row_sums = gex_matrix.sum(axis=1)
    keep_nonzero = row_sums > 0
    gex_filtered = gex_matrix[keep_nonzero, :]

    row_vars = np.var(gex_filtered, axis=1)
    keep_var = row_vars > 0
    gex_filtered = gex_filtered[keep_var, :]

    # Track which genes passed filtering (for variable gene selection)
    if gene_names is not None:
        filtered_gene_names = gene_names[keep_nonzero][keep_var]
    else:
        filtered_gene_names = np.arange(n_genes)[keep_nonzero][keep_var]

    # Step 2: Select top variable genes (R: n.var.genes = min(1000, nrow(X)))
    n_var_genes = min(1000, gex_filtered.shape[0])

    if n_cells > 50000:
        if random_state is not None:
            np.random.seed(random_state)
        idx_sample = np.random.choice(n_cells, 50000, replace=False)
        gene_var = np.var(gex_filtered[:, idx_sample], axis=1)
    else:
        gene_var = np.var(gex_filtered, axis=1)

    # Sort by variance descending, select top genes
    var_order = np.argsort(gene_var)[::-1]
    top_var_indices = var_order[:n_var_genes]

    gex_for_pca = gex_filtered[top_var_indices, :]

    # Step 3: Transpose and scale
    # R: X.for.pca <- Matrix::t(X[genes.use, presampled.cell.ids])
    # R: X.for.pca <- scale(X.for.pca)
    # R: X.for.pca[is.na(X.for.pca)] <- 0
    X_for_pca = gex_for_pca.T.astype(float)  # (cells x var_genes)

    # R's scale() uses ddof=1 for std
    col_means = X_for_pca.mean(axis=0)
    col_stds = X_for_pca.std(axis=0, ddof=1)
    col_stds[col_stds == 0] = 1.0  # avoid division by zero

    X_scaled = (X_for_pca - col_means) / col_stds
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # Step 4: PCA (R: prcomp(X.for.pca, rank.=max(n.pc), scale.=F, center=F))
    # Since data is already centered and scaled, don't center again
    n_pc_actual = min(n_pcs, X_scaled.shape[0] - 1, X_scaled.shape[1])
    if n_pc_actual < 1:
        n_pc_actual = 1

    # Use SVD directly to match R's prcomp(center=FALSE, scale.=FALSE)
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    pca_scores = U[:, :n_pc_actual] * s[:n_pc_actual]  # (cells x n_pc)

    # Step 5: Build kNN graph
    # R: buildNN2(X, k=k.knn) with nn2(data=X, k=k)
    # nn2 returns k neighbors INCLUDING self, then simplify removes self-loops
    # So effective non-self neighbors = k-1
    neighbor_indices, _ = build_knn_graph(
        pca_scores, k=k_nn, random_state=random_state
    )

    # Build igraph graph and run walktrap
    # R: igraph::graph_from_adj_list(adj.knn, duplicate=F, mode="all")
    # R: igraph::simplify(graph.knn, remove.multiple=T)
    adjacency = build_knn_sparse_matrix(neighbor_indices, n_nodes=n_cells, symmetric=True)

    # Step 6: Walktrap clustering with cut_at(k)
    # R: k <- round(N.c / gamma)
    k_clusters = round(n_cells / gamma)
    if k_clusters < 1:
        k_clusters = 1

    communities = find_communities_walktrap(
        adjacency, n_clusters=k_clusters, random_state=random_state
    )

    # Step 7: Aggregate ORIGINAL expression by cluster means
    # R: X <- GE (restore original)
    # R: GE <- X %*% M.AV; GE <- sweep(GE, 2, supercell_size, "/")
    unique_communities = np.unique(communities)
    n_super_cells = len(unique_communities)

    # Create mapping from community IDs to contiguous 0-based indices
    comm_to_idx = {comm: idx for idx, comm in enumerate(unique_communities)}

    super_cell_expr = np.zeros((n_genes, n_super_cells))
    super_cell_assignment = np.zeros(n_cells, dtype=int)

    for comm_id in unique_communities:
        sc_idx = comm_to_idx[comm_id]
        cell_mask = communities == comm_id
        super_cell_expr[:, sc_idx] = original_expr[:, cell_mask].mean(axis=1)
        super_cell_assignment[cell_mask] = sc_idx

    # Generate names
    result_gene_names = gene_names if gene_names is not None else None

    if cell_names is not None:
        super_cell_names = np.array([f"SC_{i}" for i in range(n_super_cells)])
    else:
        super_cell_names = None

    return super_cell_expr, super_cell_assignment, result_gene_names, super_cell_names


def aggregate_by_groups(
    gex_matrix: np.ndarray,
    group_labels: np.ndarray,
    gene_names: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Aggregate expression by group labels (e.g., cell types).

    Parameters
    ----------
    gex_matrix : ndarray
        Expression matrix (genes x cells).
    group_labels : ndarray
        Group assignment for each cell (n_cells,).
    gene_names : ndarray, optional
        Gene names (n_genes,).

    Returns
    -------
    grouped_expr : ndarray
        Aggregated expression (genes x n_groups).
    unique_groups : ndarray
        Unique group labels.
    gene_names_out : ndarray, optional
        Gene names (unchanged).
    group_names : ndarray
        Group names.
    """
    unique_groups = np.unique(group_labels)
    n_genes = gex_matrix.shape[0]

    grouped_expr = np.zeros((n_genes, len(unique_groups)))

    for idx, group in enumerate(unique_groups):
        mask = group_labels == group
        grouped_expr[:, idx] = gex_matrix[:, mask].mean(axis=1)

    result = [grouped_expr, unique_groups]
    if gene_names is not None:
        result.append(gene_names)
    else:
        result.append(None)
    result.append(unique_groups)

    return tuple(result)
