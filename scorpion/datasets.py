"""
Example datasets for SCORPION.

Provides a convenience function to load bundled example data derived from
colorectal cancer single-cell RNA-seq (300 genes, ~1 950 cells, 3 donors,
3 tissue regions).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import mmread
from typing import Dict


def _data_dir() -> Path:
    """Return the path to the bundled example-data directory."""
    return Path(__file__).resolve().parent / "data"


def load_example_data() -> Dict[str, object]:
    """Load the bundled example dataset.

    The dataset contains colorectal cancer single-cell RNA-seq data with
    300 genes and ~1 950 cells across 3 donors and 3 tissue regions
    (Tumor, Border, Normal).

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``gex_matrix`` : pd.DataFrame
            Gene expression matrix (genes × cells) loaded from Matrix
            Market format.
        - ``tf_motifs`` : pd.DataFrame
            Transcription-factor–target-gene motif prior network with
            columns ``[source_genesymbol, target_genesymbol, weight]``.
        - ``ppi_net`` : pd.DataFrame
            Protein–protein interaction network with columns
            ``[source_genesymbol, target_genesymbol, weight]``.
        - ``metadata`` : pd.DataFrame
            Cell-level metadata with columns
            ``[cell_id, donor, region, cell_type]``.

    Examples
    --------
    >>> from scorpion import load_example_data
    >>> data = load_example_data()
    >>> data["gex_matrix"].shape
    (300, 1954)
    >>> data["tf_motifs"].head()
    """
    d = _data_dir()

    # --- Expression matrix (MTX + row/col names) --------------------------
    expr = mmread(d / "expression.mtx").tocsr()  # genes × cells

    with open(d / "expression_genes.txt") as f:
        genes = [line.strip() for line in f]

    with open(d / "expression_cells.txt") as f:
        cells = [line.strip() for line in f]

    gex = pd.DataFrame(
        expr.toarray(),
        index=np.array(genes),
        columns=np.array(cells),
    )

    # --- Motif prior -------------------------------------------------------
    motifs = pd.read_csv(d / "motif.csv")

    # --- PPI network -------------------------------------------------------
    ppi = pd.read_csv(d / "ppi.csv")

    # --- Metadata -----------------------------------------------------------
    meta = pd.read_csv(d / "metadata.csv")

    return {
        "gex_matrix": gex,
        "tf_motifs": motifs,
        "ppi_net": ppi,
        "metadata": meta,
    }
