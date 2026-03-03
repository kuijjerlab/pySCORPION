"""
Pytest configuration and fixtures for SCORPION tests.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.io import mmread
from scipy.sparse import csr_matrix


@pytest.fixture(scope="module")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def load_r_test_data(test_data_dir):
    """Load R test data from files."""
    data = {}
    
    # Try loading expression matrix from MTX format (sparse)
    try:
        mtx_file = test_data_dir / "expression.mtx"
        genes_file = test_data_dir / "expression_genes.txt"
        cells_file = test_data_dir / "expression_cells.txt"
        
        if mtx_file.exists() and genes_file.exists() and cells_file.exists():
            # Load matrix
            expr = mmread(mtx_file).T.tocsr()  # Transpose to genes × cells
            
            # Load gene names
            genes = []
            with open(genes_file, 'r') as f:
                for line in f:
                    genes.append(line.strip())
            
            # Load cell names
            cells = []
            with open(cells_file, 'r') as f:
                for line in f:
                    cells.append(line.strip())
            
            # Subset to match (usually test uses subset of full data)
            expr = expr[:len(genes), :len(cells)].toarray()
            data['gex'] = expr
            data['gene_names'] = np.array(genes)
            data['cell_names'] = np.array(cells)
    except Exception as e:
        print(f"Could not load MTX format: {e}")
    
    # Try loading motif data
    try:
        motif_file = test_data_dir / "motif.txt"
        if motif_file.exists():
            motifs = pd.read_csv(motif_file, sep="\t", header=None, names=["tf", "target", "weight"])
            data['motifs'] = motifs
    except Exception as e:
        print(f"Could not load motif data: {e}")
    
    # Try loading PPI data
    try:
        ppi_file = test_data_dir / "ppi.txt"
        if ppi_file.exists():
            ppi = pd.read_csv(ppi_file, sep="\t", header=None, names=["protein1", "protein2", "weight"])
            data['ppi'] = ppi
    except Exception as e:
        print(f"Could not load PPI data: {e}")
    
    return data


@pytest.fixture
def toy_expression():
    """Create small toy expression matrix for fast testing.
    
    Returns a DataFrame with gene names as index so that gene names
    flow through to scorpion's motif-expression intersection.
    """
    np.random.seed(42)
    # 100 genes, 50 cells
    gex = np.random.poisson(5, (100, 50)).astype(float)
    gene_names = [f"gene_{i}" for i in range(100)]
    cell_names = [f"cell_{i}" for i in range(50)]
    return pd.DataFrame(gex, index=gene_names, columns=cell_names)


@pytest.fixture
def toy_motifs(toy_expression):
    """Create toy motif priors."""
    n_genes = toy_expression.shape[0]
    tf_names = [f"TF_{i}" for i in range(10)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    edges = []
    for tf_idx, tf in enumerate(tf_names):
        # Each TF regulates ~10 genes
        target_genes = np.random.choice(gene_names, size=10, replace=False)
        for gene in target_genes:
            weight = np.random.uniform(0.5, 1.0)
            edges.append({"tf": tf, "target": gene, "weight": weight})
    
    return pd.DataFrame(edges)


@pytest.fixture
def toy_ppi(toy_motifs):
    """Create toy PPI network."""
    tfs = toy_motifs['tf'].unique()
    
    edges = []
    for i, tf1 in enumerate(tfs):
        # Each TF interacts with ~3 other TFs
        other_tfs = np.random.choice([t for t in tfs if t != tf1], size=min(3, len(tfs)-1), replace=False)
        for tf2 in other_tfs:
            if (tf1, tf2) not in [(e['protein1'], e['protein2']) for e in edges]:
                weight = np.random.uniform(0.4, 0.9)
                edges.append({"protein1": tf1, "protein2": tf2, "weight": weight})
    
    return pd.DataFrame(edges)


@pytest.fixture
def toy_metadata(toy_expression):
    """Create toy cell metadata."""
    n_cells = toy_expression.shape[1]
    metadata = pd.DataFrame({
        "cell_id": [f"cell_{i}" for i in range(n_cells)],
        "cell_type": np.random.choice(["TypeA", "TypeB", "TypeC"], n_cells),
        "donor": np.random.choice(["donor_1", "donor_2"], n_cells),
        "batch": np.random.choice(["batch_1", "batch_2"], n_cells),
    })
    return metadata
