"""
SCORPION: Single-Cell Oriented Reconstruction of PANDA Individually Optimized Gene Regulatory Networks

Constructs cell-type-specific gene regulatory networks from single-cell
RNA-sequencing data. The method implements the SCORPION algorithm, which
first aggregates individual cells into super-cells and then applies PANDA
(Passing Attributes between Networks for Data Assimilation) to infer
transcription factor-target regulatory relationships.
"""

__version__ = "0.1.0"

from .core import scorpion, run_scorpion
from .edge_testing import test_edges, regress_edges

__all__ = ["scorpion", "run_scorpion", "test_edges", "regress_edges"]
