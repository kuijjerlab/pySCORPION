import numpy as np
import pandas as pd
from scipy.sparse import issparse


def compute_panda(correlation_matrix, ppi_matrix, motif_matrix, alpha=0.1):
    from .panda.calculations import compute_panda
    print("Using updated CPU implementation.")
    return compute_panda(correlation_matrix, ppi_matrix, motif_matrix, alpha=alpha)

from .make_super_cells import make_super_cells
from .preprocessing import log_normalize_data, remove_batch


def _normalize_edge_df(df):
    """Ensure an edge-list DataFrame uses integer column names 0, 1, 2.

    Panda expects columns accessible as ``df[0]``, ``df[1]``, ``df[2]``.
    If the DataFrame has string column names (e.g. *source*, *target*,
    *weight*) we rename them in-place to integers.
    """
    if df is None or isinstance(df, str):
        return df
    if isinstance(df, pd.DataFrame) and not all(
        isinstance(c, int) for c in df.columns
    ):
        df = df.copy()
        df.columns = list(range(len(df.columns)))
    return df



class PandaReplacement:
    """Panda replacement class."""

    def __init__(self, expression_file, motif_file, ppi_file, alpha=0.1):
        # Compute correlation matrix
        correlation_matrix = expression_file.corr(method="pearson")
        # Run PANDA
        # Convert motif and ppi edge lists to adjacency matrices
        def motif_edge_list_to_matrix(edge_df, tf_list, gene_list):
            matrix = np.zeros((len(tf_list), len(gene_list)))
            for _, row in edge_df.iterrows():
                tf, target, weight = row[0], row[1], row[2]
                if tf in tf_list and target in gene_list:
                    i = tf_list.index(tf)
                    j = gene_list.index(target)
                    matrix[i, j] = float(weight)
            return matrix

        def ppi_edge_list_to_matrix(edge_df, tf_list):
            matrix = np.zeros((len(tf_list), len(tf_list)))
            for _, row in edge_df.iterrows():
                tf1, tf2, weight = row[0], row[1], row[2]
                if tf1 in tf_list and tf2 in tf_list:
                    i = tf_list.index(tf1)
                    j = tf_list.index(tf2)
                    matrix[i, j] = float(weight)
            return matrix

        # Ensure consistent TF and gene lists for all matrices
        # Match R's intersection mode: union both PPI columns, intersect with motif TFs
        tf_set = set(ppi_file[0]).union(set(ppi_file[1])).intersection(set(motif_file[0]))
        gene_set = set(motif_file[1]).intersection(set(expression_file.index))
        tf_list = sorted(list(tf_set))
        gene_list = sorted(list(gene_set))

        # Robust checks for empty lists
        if not tf_list:
            raise ValueError("No TFs found after intersection. Check motif and PPI input files.")
        if not gene_list:
            raise ValueError("No genes found after intersection. Check motif and expression input files.")

        # Filter motif and ppi edge lists
        motif_df = motif_file[(motif_file[0].isin(tf_list)) & (motif_file[1].isin(gene_list))]
        ppi_df = ppi_file[(ppi_file[0].isin(tf_list)) & (ppi_file[1].isin(tf_list))]

        # Build expression matrix (filtered genes only, ordered)
        expr_matrix = expression_file.loc[gene_list]
        # Note: do NOT filter out all-zero genes - R includes all genes in the intersection
        # and handles NaN values in correlation naturally

        # Compute correlation between genes (rows)
        correlation_matrix_np = np.corrcoef(expr_matrix.values)
        # Handle NaN values in correlation matrix (from genes with zero variance)
        # Replace NaN with mean of non-NaN values in that row/column, or 0 if all NaN
        for i in range(correlation_matrix_np.shape[0]):
            row = correlation_matrix_np[i, :]
            nan_mask = np.isnan(row)
            if nan_mask.any():
                mean_val = np.nanmean(row)
                if np.isnan(mean_val):
                    mean_val = 0.0
                correlation_matrix_np[i, nan_mask] = mean_val
        # Set diagonal to 1.0 (perfect self correlation)
        np.fill_diagonal(correlation_matrix_np, 1.0)
        # Remove NaN rows/columns from correlation matrix
        if np.isnan(correlation_matrix_np).all():
            raise ValueError("Correlation matrix is all NaN. Check expression input for variance.")

        # Build motif matrix (TFs x Genes, ordered)
        motif_matrix = motif_edge_list_to_matrix(motif_df, tf_list, gene_list)
        # Build ppi matrix (TFs x TFs, ordered)
        ppi_matrix = ppi_edge_list_to_matrix(ppi_df, tf_list)

        # Ensure motif_matrix.shape[1] == correlation_matrix_np.shape[0]
        assert motif_matrix.shape[1] == correlation_matrix_np.shape[0], "Motif and correlation matrix gene dimensions must match"
        # Ensure motif_matrix.shape[0] == ppi_matrix.shape[0]
        assert motif_matrix.shape[0] == ppi_matrix.shape[0], "Motif and PPI matrix TF dimensions must match"

        # Initialize zero-row TFs with small values so they can learn during PANDA
        # This prevents TFs with no initial motif edges to filtered genes from staying zero
        row_sums = motif_matrix.sum(axis=1)
        zero_rows = row_sums == 0
        if zero_rows.any():
            avg_nonzero = motif_matrix[motif_matrix > 0].mean() if (motif_matrix > 0).any() else 0.01
            init_val = avg_nonzero * 0.01
            motif_matrix[zero_rows, :] = init_val

        panda_result = compute_panda(
            correlation_matrix_np,
            ppi_matrix,
            motif_matrix,
            alpha=alpha
        )
        regNet, coregNet, coopNet = panda_result
        # Z-score normalize regNet to match R output (zScaling=TRUE)
        from .panda.calculations import normalize_network
        regNet_z = normalize_network(regNet)
        # Replace NaN values with 0
        regNet_z = np.nan_to_num(regNet_z)
        self.regNet = pd.DataFrame(regNet_z, index=tf_list, columns=gene_list)
        self.coregNet = pd.DataFrame(coregNet, index=gene_list, columns=gene_list)
        self.coopNet = pd.DataFrame(coopNet, index=tf_list, columns=tf_list)
        # Output as edge-list: tf, target, score
        edge_list = []
        for i, tf in enumerate(tf_list):
            for j, gene in enumerate(gene_list):
                score = self.regNet.iloc[i, j]
                edge_list.append((tf, gene, score))
        self.edge_df = pd.DataFrame(edge_list, columns=["tf", "target", "score"])
        self.panda_network = self.edge_df

    def panda_loop(self, correlation_matrix=None, motif_matrix=None, ppi_matrix=None):
        return compute_panda(
            expression_file=self.expression_file,
            motif_file=self.motif_file,
            ppi_file=self.ppi_file,
            computing=self.computing,
            alpha=self.alpha,
        )


class Scorpion:
    """Infer gene regulatory networks from single-cell data with SCORPION.

    SCORPION (Single-Cell Oriented Regulatory Population Inference Of
    Networks) extends PANDA by first coarse-graining single-cell
    expression data into super-cells, then running the standard PANDA
    message-passing algorithm on the aggregated expression matrix.

    The workflow is:

    1. *(optional)* Log-normalize and/or batch-correct the expression
       matrix.
    2. *(optional)* Group cells by metadata columns and build one
       network per group.
    3. Coarse-grain single-cell expression into super-cells via PCA,
       kNN graph construction, and Walktrap/Louvain clustering.
    4. Load motif prior and PPI data (inherited from Panda).
    5. Compute coexpression network from super-cell expression.
    6. Normalize networks and run PANDA.

    ...existing docstring...
    """

    def __init__(
        self,
        expression_file,
        motif_file,
        ppi_file,
        cells_metadata=None,
        group_by=None,
        min_cells=30,
        normalize_data=True,
        remove_batch_effect=False,
        batch=None,
        gamma=10,
        n_pc=25,
        fast_pca=False,
        k_knn=5,
        igraph_clustering="walktrap",
        seed=12345,
        computing="cpu",
        precision="double",
        save_memory=True,
        save_tmp=False,
        remove_missing=False,
        keep_expression_matrix=False,
        modeProcess="intersection",
        alpha=0.1,
    ):
        # ...existing code from __init__...
        if isinstance(expression_file, pd.DataFrame):
            gene_names = list(expression_file.index)
            cell_names = list(expression_file.columns.astype(str))
            X = expression_file.values.astype(float)
        elif issparse(expression_file):
            gene_names = None
            cell_names = None
            X = expression_file
        elif isinstance(expression_file, np.ndarray):
            gene_names = None
            cell_names = None
            X = expression_file.astype(float)
        else:
            raise ValueError(
                "expression_file must be a pandas DataFrame, numpy ndarray, "
                "or scipy sparse matrix (genes x cells). "
                "File path strings are not supported for single-cell data; "
                "please load the data first."
            )

        motif_file = _normalize_edge_df(motif_file)
        ppi_file = _normalize_edge_df(ppi_file)

        panda_kwargs = dict(
            computing=computing,
            precision=precision,
            save_memory=save_memory,
            save_tmp=save_tmp,
            remove_missing=remove_missing,
            keep_expression_matrix=keep_expression_matrix,
            modeProcess=modeProcess,
            alpha=alpha,
        )
        sc_kwargs = dict(
            gamma=gamma,
            k_knn=k_knn,
            n_pc=n_pc,
            fast_pca=fast_pca,
            igraph_clustering=igraph_clustering,
            seed=seed,
        )

        if group_by is None:
            # Use original gene list for single-group mode
            self._run_single(
                X, gene_names, cell_names, motif_file, ppi_file,
                sc_kwargs, panda_kwargs,
            )
            if self.panda_network is not None and not self.panda_network.empty:
                self.panda_networks = {"single_group": self.panda_network}
                self.networks_df = self.panda_network.copy()
                print("SCORPION: Single-group network successfully combined")
            else:
                self.panda_networks = {}
                self.networks_df = pd.DataFrame()
                print("SCORPION: No valid single-group network produced.")
            return

        if cells_metadata is None:
            raise ValueError(
                "cells_metadata must be provided when group_by is set."
            )
        if isinstance(group_by, str):
            group_by = [group_by]

        n_cells = X.shape[1]
        if cells_metadata.shape[0] != n_cells:
            raise ValueError(
                f"cells_metadata has {cells_metadata.shape[0]} rows but "
                f"expression_file has {n_cells} columns (cells)."
            )
        missing = [c for c in group_by if c not in cells_metadata.columns]
        if missing:
            raise ValueError(
                f"group_by columns not found in cells_metadata: "
                f"{', '.join(missing)}"
            )
        if remove_batch_effect:
            if batch is None:
                raise ValueError(
                    "batch must be provided when remove_batch_effect is True."
                )
            batch = np.asarray(batch)
            if len(batch) != n_cells:
                raise ValueError(
                    f"batch has {len(batch)} elements but expression_file "
                    f"has {n_cells} columns (cells)."
                )

        if normalize_data:
            print("SCORPION: Normalizing data (log scale)")
            X = log_normalize_data(X)

        if remove_batch_effect:
            print("SCORPION: Correcting for batch effects")
            median_expr = np.median(X, axis=1, keepdims=True)
            X = remove_batch(X, batch)
            X = X + median_expr

        group_ids = (
            cells_metadata[group_by]
            .astype(str)
            .apply(lambda row: "--".join(row), axis=1)
            .values
        )
        unique_groups = list(dict.fromkeys(group_ids))

        min_cells = max(int(min_cells), 30)

        group_counts = {g: 0 for g in unique_groups}
        for g in group_ids:
            group_counts[g] += 1
        valid_groups = [g for g in unique_groups if group_counts[g] >= min_cells]

        total = len(unique_groups)
        kept = len(valid_groups)
        print(f"SCORPION: {total} groups requested, "
              f"{kept} meet the minimum cell requirement ({min_cells})")
        if kept == 0:
            raise ValueError(
                f"No groups have at least {min_cells} cells. "
                "Lower min_cells or provide more data."
            )

        if gene_names is None:
            gene_names = [f"gene_{i+1}" for i in range(X.shape[0])]

        # Compute intersection of gene and TF lists across all groups
        all_gene_sets = [set(expression_file.index)]
        all_tf_sets = [set(motif_file[0]), set(ppi_file[0]), set(ppi_file[1])]
        for group_id in valid_groups:
            mask = group_ids == group_id
            X_group = X[:, mask]
            aggregated = self._aggregate(X_group, gene_names, None, sc_kwargs)
            group_genes = set([g for g in gene_names if (aggregated != 0).any(axis=1)[gene_names.index(g)]])
            all_gene_sets.append(group_genes)
        common_genes = sorted(list(set.intersection(*all_gene_sets)))
        common_tfs = sorted(list(set.intersection(*all_tf_sets)))

        network_results = {}
        for idx, group_id in enumerate(valid_groups):
            print(f"SCORPION: [{idx+1}/{kept}] Computing network "
                  f"for group '{group_id}' "
                  f"({group_counts[group_id]} cells)")

            mask = group_ids == group_id
            X_group = X[:, mask]
            aggregated = self._aggregate(X_group, gene_names, None, sc_kwargs)
            # Build expression DataFrame with common genes only
            expression_df = self._to_dataframe(aggregated, gene_names).loc[common_genes]

            if idx == 0:
                PandaReplacement.__init__(
                    self,
                    expression_file=expression_df,
                    motif_file=motif_file,
                    ppi_file=ppi_file,
                    alpha=alpha,
                )
            else:
                tmp = object.__new__(PandaReplacement)
                PandaReplacement.__init__(
                    tmp,
                    expression_file=expression_df,
                    motif_file=motif_file,
                    ppi_file=ppi_file,
                    alpha=alpha,
                )
                network_results[group_id] = tmp.panda_network

        self.panda_networks = network_results
        # Combine edge-lists for all groups
        combined = None
        for group_id, net in network_results.items():
            df = net.copy()
            df = df.rename(columns={"score": group_id}) if "score" in df.columns else df.rename(columns={df.columns[-1]: group_id})
            if combined is None:
                combined = df[["tf", "target", group_id]]
            else:
                combined = combined.merge(df[["tf", "target", group_id]], on=["tf", "target"], how="outer")
        self.networks_df = combined
        print("SCORPION: Networks successfully combined")

    def _run_single(self, X, gene_names, cell_names,
                    motif_file, ppi_file, sc_kwargs, panda_kwargs):

        print("[SCORPION] Entered _run_single: exporting supercell matrix will be attempted.")
        aggregated = self._aggregate(X, gene_names, cell_names, sc_kwargs)

        if gene_names is None:
            gene_names = [f"gene_{i+1}" for i in range(aggregated.shape[0])]

        expression_df = self._to_dataframe(aggregated, gene_names)


        # Export the supercell-aggregated expression matrix for reproducibility
        import os
        # Use workspace root as base for absolute path
        supercell_export_path = os.path.abspath(os.path.join(os.getcwd(), "scorpion/tests/scorpion/python_supercells_single_group.csv"))
        export_dir = os.path.dirname(supercell_export_path)
        print(f"[SCORPION] Current working directory: {os.getcwd()}")
        print(f"[SCORPION] Attempting to export supercell matrix to: {supercell_export_path}")
        try:
            os.makedirs(export_dir, exist_ok=True)
            expression_df.to_csv(supercell_export_path, sep=",", index=True)
            print(f"[SCORPION] Exported supercell matrix to {supercell_export_path}")
        except Exception as e:
            import traceback
            print(f"[SCORPION] Failed to export supercell matrix: {e}\n{traceback.format_exc()}")

        panda_instance = PandaReplacement(
            expression_file=expression_df,
            motif_file=motif_file,
            ppi_file=ppi_file,
            alpha=panda_kwargs.get("alpha", 0.1),
        )
        self.panda_network = panda_instance.panda_network
        self.regNet = panda_instance.regNet
        self.coregNet = panda_instance.coregNet
        self.coopNet = panda_instance.coopNet
        if self.panda_network is None or self.panda_network.empty:
            raise ValueError("No valid single-group network produced. Check input data and aggregation.")

    @staticmethod
    def _aggregate(X, gene_names, cell_names, sc_kwargs):
        n_cells = X.shape[1] if not issparse(X) else X.shape[1]
        print("SCORPION: Aggregating single-cell data into super-cells ...")
        aggregated = make_super_cells(
            X=X,
            gene_names=gene_names,
            cell_names=cell_names,
            **sc_kwargs,
        )
        print(
            "SCORPION: Aggregated %d cells into %d super-cells (gamma=%.1f)"
            % (n_cells, aggregated.shape[1], sc_kwargs["gamma"])
        )
        return aggregated

    @staticmethod
    def _to_dataframe(aggregated, gene_names):
        supercell_names = [
            f"supercell_{i+1}" for i in range(aggregated.shape[1])
        ]
        return pd.DataFrame(
            aggregated, index=gene_names, columns=supercell_names,
        )

    @staticmethod
    def _combine_networks(network_results):
        if not network_results:
            return pd.DataFrame()
        first_id = next(iter(network_results))
        first_net = network_results[first_id]

        tf_target = pd.DataFrame(
            [
                (tf, gene)
                for tf in first_net.index
                for gene in first_net.columns
            ],
            columns=["tf", "target"],
        )

        combined = tf_target.copy()
        for group_id, net in network_results.items():
            combined[group_id] = net.values.ravel()

        return combined
