"""
Statistical testing of network edges: testEdges, regressEdges and maEdges.

Matches the R SCORPION package's testEdges(), regressEdges() and maEdges()
exactly, including SAM-style variance moderation, empirical null correction,
vectorised linear regression and fixed/random-effects meta-analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Union, Sequence
import warnings


P_ADJUST_METHODS = (
    "holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_global_s0(
    networks_df: pd.DataFrame,
    test_type: str,
    group1: List[str],
    group2: Optional[List[str]],
    paired: bool,
) -> float:
    """Pre-compute the global SAM fudge factor *s0* (median SE across all edges)."""

    if test_type == "single":
        mat = networks_df[group1].to_numpy(dtype=float)
        n_valid = np.sum(~np.isnan(mat), axis=1)
        mean_edge = np.nanmean(mat, axis=1)
        mean_sq = np.nanmean(mat ** 2, axis=1)
        sd_edge = np.sqrt(n_valid / (n_valid - 1) * (mean_sq - mean_edge ** 2))
        se = sd_edge / np.sqrt(n_valid)
    elif paired:
        mat1 = networks_df[group1].to_numpy(dtype=float)
        mat2 = networks_df[group2].to_numpy(dtype=float)
        diff = mat1 - mat2
        n_valid = np.sum(~np.isnan(mat1) & ~np.isnan(mat2), axis=1)
        diff_mean = np.nanmean(diff, axis=1)
        diff_mean_sq = np.nanmean(diff ** 2, axis=1)
        sd_diff = np.sqrt(n_valid / (n_valid - 1) * (diff_mean_sq - diff_mean ** 2))
        se = sd_diff / np.sqrt(n_valid)
    else:
        mat1 = networks_df[group1].to_numpy(dtype=float)
        mat2 = networks_df[group2].to_numpy(dtype=float)
        n1 = np.sum(~np.isnan(mat1), axis=1)
        n2 = np.sum(~np.isnan(mat2), axis=1)
        mean1 = np.nanmean(mat1, axis=1)
        mean2 = np.nanmean(mat2, axis=1)
        mean_sq1 = np.nanmean(mat1 ** 2, axis=1)
        mean_sq2 = np.nanmean(mat2 ** 2, axis=1)
        var1 = n1 / (n1 - 1) * (mean_sq1 - mean1 ** 2)
        var2 = n2 / (n2 - 1) * (mean_sq2 - mean2 ** 2)
        se = np.sqrt(var1 / n1 + var2 / n2)

    return float(np.nanmedian(se))


def _p_adjust(pvalues: np.ndarray, method: str = "BH") -> np.ndarray:
    """
    Multiple-testing p-value adjustment matching R's ``stats::p.adjust()``.

    NaN entries are left as NaN and excluded from the number of tests.
    """
    if method not in P_ADJUST_METHODS:
        raise ValueError(
            f"padjust_method must be one of {P_ADJUST_METHODS}, got '{method}'"
        )
    if method == "fdr":
        method = "BH"

    p0 = np.asarray(pvalues, dtype=float).copy()
    valid = ~np.isnan(p0)
    p = p0[valid]
    n = len(p)
    if n <= 1:
        return p0
    if n == 2 and method == "hommel":
        method = "hochberg"

    if method == "bonferroni":
        adj = np.minimum(1.0, n * p)
    elif method == "holm":
        o = np.argsort(p, kind="stable")
        i = np.arange(1, n + 1)
        adj = np.empty(n)
        adj[o] = np.minimum(1.0, np.maximum.accumulate((n + 1 - i) * p[o]))
    elif method in ("hochberg", "BH", "BY"):
        o = np.argsort(-p, kind="stable")
        i = np.arange(n, 0, -1)
        if method == "hochberg":
            scaled = (n + 1 - i) * p[o]
        elif method == "BH":
            scaled = n / i * p[o]
        else:
            q = np.sum(1.0 / np.arange(1, n + 1))
            scaled = q * n / i * p[o]
        adj = np.empty(n)
        adj[o] = np.minimum(1.0, np.minimum.accumulate(scaled))
    elif method == "hommel":
        o = np.argsort(p, kind="stable")
        ps = p[o]
        i = np.arange(1, n + 1)
        q = np.full(n, np.min(n * ps / i))
        pa = q.copy()
        for m in range(n - 1, 1, -1):
            i1 = np.arange(0, n - m + 1)
            i2 = np.arange(n - m + 1, n)
            q1 = np.min(m * ps[i2] / np.arange(2, m + 1))
            q[i1] = np.minimum(m * ps[i1], q1)
            q[i2] = q[n - m]
            pa = np.maximum(pa, q)
        adj = np.empty(n)
        adj[o] = np.maximum(pa, ps)
    else:  # none
        adj = p

    p0[valid] = adj
    return p0


def _p_adjust_bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg p-value adjustment (matches R's p.adjust(method='BH'))."""
    return _p_adjust(pvalues, "BH")


def _t_pvalues(t_stat: np.ndarray, df: np.ndarray, alternative: str) -> np.ndarray:
    if alternative == "two.sided":
        return 2 * scipy_stats.t.sf(np.abs(t_stat), df)
    if alternative == "greater":
        return scipy_stats.t.sf(t_stat, df)
    return scipy_stats.t.cdf(t_stat, df)


def _test_edges_single(
    networks_df: pd.DataFrame,
    group1: List[str],
    alternative: str,
    moderate_variance: bool,
    s0: Optional[float],
) -> pd.DataFrame:
    """Vectorised one-sample t-test against zero for every edge."""

    mat = networks_df[group1].to_numpy(dtype=float)
    mean_edge = np.nanmean(mat, axis=1)
    n_valid = np.sum(~np.isnan(mat), axis=1).astype(float)

    mean_sq = np.nanmean(mat ** 2, axis=1)
    sd_edge = np.sqrt(n_valid / (n_valid - 1) * (mean_sq - mean_edge ** 2))

    # Raw/unmoderated SE for downstream meta-analysis
    raw_se = sd_edge / np.sqrt(n_valid)

    # SE used for hypothesis testing
    se = raw_se
    if moderate_variance:
        if s0 is None:
            s0 = float(np.nanmedian(raw_se))
        se = raw_se + s0

    t_stat = mean_edge / se
    pval = _t_pvalues(t_stat, n_valid - 1, alternative)

    bad = (n_valid < 2) | np.isnan(sd_edge)
    if not moderate_variance:
        bad |= (sd_edge == 0)
    t_stat[bad] = np.nan
    pval[bad] = np.nan
    raw_se = raw_se.copy()
    raw_se[bad] = np.nan

    return pd.DataFrame({
        "tf": networks_df["tf"].values,
        "target": networks_df["target"].values,
        "meanEdge": mean_edge,
        "SE": raw_se,
        "tStatistic": t_stat,
        "pValue": pval,
    })


def _test_edges_two_sample(
    networks_df: pd.DataFrame,
    group1: List[str],
    group2: List[str],
    alternative: str,
    min_log2fc: float,
    moderate_variance: bool,
    s0: Optional[float],
) -> pd.DataFrame:
    """Vectorised Welch two-sample t-test for every edge."""

    mat1 = networks_df[group1].to_numpy(dtype=float)
    mat2 = networks_df[group2].to_numpy(dtype=float)

    mean1 = np.nanmean(mat1, axis=1)
    mean2 = np.nanmean(mat2, axis=1)
    mean_edge = (mean1 + mean2) / 2
    diff_mean = mean1 - mean2
    log2fc = mean1 - mean2

    # Filter by minLog2FC
    idx = np.where(np.abs(log2fc) >= min_log2fc)[0]
    mat1 = mat1[idx]
    mat2 = mat2[idx]
    mean1 = mean1[idx]
    mean2 = mean2[idx]
    mean_edge = mean_edge[idx]
    diff_mean = diff_mean[idx]
    log2fc = log2fc[idx]
    tf_vals = networks_df["tf"].values[idx]
    target_vals = networks_df["target"].values[idx]

    n1 = np.sum(~np.isnan(mat1), axis=1).astype(float)
    n2 = np.sum(~np.isnan(mat2), axis=1).astype(float)

    mean_sq1 = np.nanmean(mat1 ** 2, axis=1)
    mean_sq2 = np.nanmean(mat2 ** 2, axis=1)
    var1 = n1 / (n1 - 1) * (mean_sq1 - mean1 ** 2)
    var2 = n2 / (n2 - 1) * (mean_sq2 - mean2 ** 2)

    # Raw Welch SE
    raw_se = np.sqrt(var1 / n1 + var2 / n2)

    # SE used for hypothesis testing
    se = raw_se
    if moderate_variance:
        if s0 is None:
            s0 = float(np.nanmedian(raw_se))
        se = raw_se + s0

    t_stat = diff_mean / se

    # Welch-Satterthwaite degrees of freedom
    df = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    )
    pval = _t_pvalues(t_stat, df, alternative)

    no_mod = np.bool_(not moderate_variance)
    bad = (
        (n1 < 2) | (n2 < 2)
        | np.isnan(var1) | np.isnan(var2)
        | (no_mod & ((var1 == 0) | (var2 == 0) | (raw_se == 0)))
        | np.isnan(raw_se)
    )
    t_stat[bad] = np.nan
    pval[bad] = np.nan
    raw_se = raw_se.copy()
    raw_se[bad] = np.nan

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    cohens_d = diff_mean / pooled_sd
    cohens_d[(pooled_sd == 0) | np.isnan(pooled_sd)] = np.nan

    return pd.DataFrame({
        "tf": tf_vals,
        "target": target_vals,
        "meanGroup1": mean1,
        "meanGroup2": mean2,
        "cohensD": cohens_d,
        "log2FoldChange": log2fc,
        "meanEdge": mean_edge,
        "SE": raw_se,
        "tStatistic": t_stat,
        "pValue": pval,
    })


def _test_edges_paired(
    networks_df: pd.DataFrame,
    group1: List[str],
    group2: List[str],
    alternative: str,
    min_log2fc: float,
    moderate_variance: bool,
    s0: Optional[float],
) -> pd.DataFrame:
    """Vectorised paired t-test for every edge."""

    mat1 = networks_df[group1].to_numpy(dtype=float)
    mat2 = networks_df[group2].to_numpy(dtype=float)

    mean1 = np.nanmean(mat1, axis=1)
    mean2 = np.nanmean(mat2, axis=1)
    mean_edge = (mean1 + mean2) / 2
    diff_mat = mat1 - mat2
    diff_mean = np.nanmean(diff_mat, axis=1)
    log2fc = mean1 - mean2

    idx = np.where(np.abs(log2fc) >= min_log2fc)[0]
    diff_mat = diff_mat[idx]
    mat1 = mat1[idx]
    mat2 = mat2[idx]
    mean1 = mean1[idx]
    mean2 = mean2[idx]
    mean_edge = mean_edge[idx]
    diff_mean = diff_mean[idx]
    log2fc = log2fc[idx]
    tf_vals = networks_df["tf"].values[idx]
    target_vals = networks_df["target"].values[idx]

    valid_pairs = ~np.isnan(mat1) & ~np.isnan(mat2)
    n_valid = np.sum(valid_pairs, axis=1).astype(float)

    diff_mean_sq = np.nanmean(diff_mat ** 2, axis=1)
    sd_diff = np.sqrt(n_valid / (n_valid - 1) * (diff_mean_sq - diff_mean ** 2))

    # Raw paired SE before variance moderation
    raw_se = sd_diff / np.sqrt(n_valid)

    # SE used for hypothesis testing
    se = raw_se
    if moderate_variance:
        if s0 is None:
            s0 = float(np.nanmedian(raw_se))
        se = raw_se + s0

    t_stat = diff_mean / se
    pval = _t_pvalues(t_stat, n_valid - 1, alternative)

    bad = (n_valid < 2) | np.isnan(sd_diff)
    if not moderate_variance:
        bad |= (sd_diff == 0)
    t_stat[bad] = np.nan
    pval[bad] = np.nan
    raw_se = raw_se.copy()
    raw_se[bad] = np.nan

    cohens_d = diff_mean / sd_diff
    cohens_d[(sd_diff == 0) | np.isnan(sd_diff)] = np.nan

    return pd.DataFrame({
        "tf": tf_vals,
        "target": target_vals,
        "meanGroup1": mean1,
        "meanGroup2": mean2,
        "cohensD": cohens_d,
        "log2FoldChange": log2fc,
        "meanEdge": mean_edge,
        "SE": raw_se,
        "tStatistic": t_stat,
        "pValue": pval,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_edges(
    networks_df: pd.DataFrame,
    test_type: str = "single",
    group1: Optional[List[str]] = None,
    group2: Optional[List[str]] = None,
    paired: bool = False,
    alternative: str = "two.sided",
    padjust_method: str = "BH",
    min_log2fc: float = 0.0,
    moderate_variance: bool = True,
    empirical_null: bool = True,
    n_cores: int = 1,
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Test edges from SCORPION networks.

    Performs statistical testing of network edges from :func:`run_scorpion`
    output.  Supports single-sample tests (testing if edges differ from
    zero) and two-sample tests (comparing edges between two groups).

    Parameters
    ----------
    networks_df : DataFrame
        A DataFrame output from :func:`run_scorpion` containing TF-target
        pairs as rows and network identifiers as columns.
    test_type : str, default="single"
        Character specifying the test type.  Options are:

        - ``"single"``: Single-sample test (one-sample t-test against
          zero).
        - ``"two.sample"``: Two-sample comparison (t-test between two
          groups).
    group1 : list of str
        Character vector of column names in *networks_df* representing
        the first group (or the only group for single-sample tests).
    group2 : list of str, optional
        Character vector of column names in *networks_df* representing
        the second group.  Required for two-sample tests, ignored for
        single-sample tests.
    paired : bool, default=False
        Whether to perform a paired t-test.  Default False.  When True,
        *group1* and *group2* must have the same length and be in matched
        order (e.g., ``group1[0]`` is paired with ``group2[0]``).  Useful
        for comparing matched samples such as Tumor vs Normal from the
        same patient.
    alternative : str, default="two.sided"
        Character specifying the alternative hypothesis.  Options:
        ``"two.sided"`` (default), ``"greater"``, or ``"less"``.
    padjust_method : str, default="BH"
        Character specifying the p-value adjustment method for multiple
        testing correction.  One of ``"holm"``, ``"hochberg"``,
        ``"hommel"``, ``"bonferroni"``, ``"BH"``, ``"BY"``, ``"fdr"`` or
        ``"none"`` (as in R's ``p.adjust``).  Default ``"BH"``
        (Benjamini-Hochberg FDR).
    min_log2fc : float, default=0.0
        Numeric threshold for minimum absolute log2 fold change to
        include in testing.  For two-sample and paired tests, edges with
        ``|log2FoldChange|`` below this threshold are excluded.  Not
        applicable for single-sample tests.  Default 0.
    moderate_variance : bool, default=True
        Whether to apply SAM-style variance moderation.  When True, adds
        a fudge factor (*s0*, the median of all standard errors) to the
        denominator of the t-statistic.  This prevents edges with very
        small variance from producing extreme t-statistics, resulting in
        volcano plots more similar to limma output.  Default True.
    empirical_null : bool, default=True
        Whether to estimate the null distribution empirically from the
        observed t-statistics.  When True, uses the median and MAD
        (median absolute deviation) of all t-statistics to recenter and
        rescale them, then computes p-values from the standard normal.
        This is Efron's empirical null correction (as in locfdr) and is
        essential when testing millions of correlated edges.  Runs in
        O(n) time.  Default True.
    n_cores : int, default=1
        Number of parallel workers.  Default 1 (sequential processing).
        When greater than 1, edges are split into batches and processed
        in parallel, and the variance-moderation fudge factor *s0* is
        computed once over all edges.
    batch_size : int, optional
        Number of edges (rows) per batch for parallel processing.
        Default None, which uses ``ceil(len(networks_df) / n_cores)``.
        Only used when ``n_cores > 1``.

    Returns
    -------
    result : DataFrame
        A DataFrame containing:

        - ``tf``: Transcription factor
        - ``target``: Target gene
        - ``meanEdge``: Mean edge weight
        - ``SE``: Raw, unmoderated sampling standard error
        - ``tStatistic``: Test statistic
        - ``pValue``: Raw p-value
        - ``pAdj``: Adjusted p-value
        - For two-sample tests: ``meanGroup1``, ``meanGroup2``,
          ``cohensD``, ``log2FoldChange`` (Group1 - Group2)

    Notes
    -----
    For single-sample tests, the function tests whether the mean edge
    weight across replicates significantly differs from zero using a
    one-sample t-test.

    For two-sample tests, the function compares edge weights between two
    groups using Welch's t-test (unequal variances assumed).

    For paired tests, the function calculates the difference between
    matched pairs and performs a one-sample t-test on the differences
    (testing if mean difference differs from zero).  This is appropriate
    when samples are matched (e.g., Tumor and Normal from the same
    patient).

    The returned ``SE`` is the raw sampling standard error before
    optional SAM-style variance moderation.  It is intended for
    downstream effect-size meta-analysis with :func:`ma_edges`.  The
    moderated SE is used only internally for calculating the test
    statistic and p-value.

    Edges are tested independently, and p-values are adjusted for
    multiple testing using the specified method.

    See Also
    --------
    run_scorpion : Build per-group regulatory networks.
    regress_edges : Regression analysis across ordered conditions.
    ma_edges : Meta-analysis of edges across studies.

    Examples
    --------
    >>> # Single-sample test: Test if edges in Tumor region differ from zero
    >>> tumor_nets = [c for c in nets.columns if c.endswith("--T")]
    >>> results_single = test_edges(
    ...     networks_df=nets,
    ...     test_type="single",
    ...     group1=tumor_nets,
    ... )

    >>> # Two-sample test: Compare Tumor vs Normal regions
    >>> tumor_nets = [c for c in nets.columns if c.endswith("--T")]
    >>> normal_nets = [c for c in nets.columns if c.endswith("--N")]
    >>> results_tumor_vs_normal = test_edges(
    ...     networks_df=nets,
    ...     test_type="two.sample",
    ...     group1=tumor_nets,
    ...     group2=normal_nets,
    ... )

    >>> # Paired t-test: Compare matched Tumor vs Normal samples
    >>> tumor_ordered = ["P31--T", "P32--T", "P33--T"]
    >>> normal_ordered = ["P31--N", "P32--N", "P33--N"]
    >>> results_paired = test_edges(
    ...     networks_df=nets,
    ...     test_type="two.sample",
    ...     group1=tumor_ordered,
    ...     group2=normal_ordered,
    ...     paired=True,
    ... )
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _test_edges_impl(
            networks_df=networks_df, test_type=test_type,
            group1=group1, group2=group2, paired=paired,
            alternative=alternative, padjust_method=padjust_method,
            min_log2fc=min_log2fc, moderate_variance=moderate_variance,
            empirical_null=empirical_null, n_cores=n_cores,
            batch_size=batch_size,
        )


def _test_edges_impl(
    networks_df, test_type, group1, group2, paired,
    alternative, padjust_method, min_log2fc,
    moderate_variance, empirical_null, n_cores, batch_size,
) -> pd.DataFrame:
    """Internal implementation of test_edges(), called inside a warnings context."""
    # Normalize parameter values (accept both Python-style and R-style)
    test_type = test_type.replace("_", ".")
    alternative = alternative.replace("_", ".")

    # Validate
    if test_type not in ("single", "two.sample"):
        raise ValueError(f"test_type must be 'single' or 'two.sample', got '{test_type}'")
    if alternative not in ("two.sided", "greater", "less"):
        raise ValueError("alternative must be 'two.sided', 'greater', or 'less'")
    if padjust_method not in P_ADJUST_METHODS:
        raise ValueError(
            f"padjust_method must be one of {P_ADJUST_METHODS}, got '{padjust_method}'"
        )
    if group1 is None:
        raise ValueError("group1 must be specified")
    group1 = list(group1)
    missing1 = [c for c in group1 if c not in networks_df.columns]
    if missing1:
        raise ValueError(
            f"Some group1 columns not found in networks_df: {', '.join(map(str, missing1))}"
        )
    if test_type == "two.sample":
        if group2 is None:
            raise ValueError("group2 must be specified for two.sample test")
        group2 = list(group2)
        missing2 = [c for c in group2 if c not in networks_df.columns]
        if missing2:
            raise ValueError(
                f"Some group2 columns not found in networks_df: {', '.join(map(str, missing2))}"
            )
        if paired and len(group1) != len(group2):
            raise ValueError("For paired tests, group1 and group2 must have the same length")
    if paired and test_type == "single":
        raise ValueError("Paired tests require test_type='two.sample'")
    try:
        n_cores = int(n_cores)
    except (TypeError, ValueError):
        n_cores = 0
    if n_cores < 1:
        raise ValueError("n_cores must be a positive integer")
    if batch_size is not None:
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            batch_size = 0
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer or None")
    if "tf" not in networks_df.columns or "target" not in networks_df.columns:
        raise ValueError("networks_df must contain 'tf' and 'target' columns")

    n_edges = len(networks_df)

    # With parallel batches, s0 must be computed once over all edges
    s0 = None
    if moderate_variance and n_cores > 1:
        s0 = _compute_global_s0(networks_df, test_type, group1, group2, paired)

    if test_type == "single":
        def helper(df):
            return _test_edges_single(df, group1, alternative, moderate_variance, s0)
    elif paired:
        def helper(df):
            return _test_edges_paired(
                df, group1, group2, alternative, min_log2fc, moderate_variance, s0,
            )
    else:
        def helper(df):
            return _test_edges_two_sample(
                df, group1, group2, alternative, min_log2fc, moderate_variance, s0,
            )

    if n_cores == 1:
        results = helper(networks_df)
    else:
        if batch_size is None:
            batch_size = int(np.ceil(n_edges / n_cores))
        chunks = [
            networks_df.iloc[start:start + batch_size]
            for start in range(0, n_edges, batch_size)
        ]
        with ThreadPoolExecutor(max_workers=n_cores) as pool:
            parts = list(pool.map(helper, chunks))
        results = pd.concat(parts, ignore_index=True)

    # Empirical null correction (Efron's method)
    if empirical_null:
        t_vals = results["tStatistic"].to_numpy(dtype=float)
        valid_mask = np.isfinite(t_vals)
        if np.sum(valid_mask) > 100:
            null_center = np.median(t_vals[valid_mask])
            # R: mad(x, constant = 1.4826)
            null_scale = 1.4826 * np.median(np.abs(t_vals[valid_mask] - null_center))
            if null_scale > 0:
                z = (t_vals - null_center) / null_scale
                if alternative == "two.sided":
                    new_p = 2 * scipy_stats.norm.sf(np.abs(z))
                elif alternative == "greater":
                    new_p = scipy_stats.norm.sf(z)
                else:
                    new_p = scipy_stats.norm.cdf(z)
                results["pValue"] = new_p

    # Adjust p-values
    results["pAdj"] = _p_adjust(results["pValue"].to_numpy(), padjust_method)
    results.reset_index(drop=True, inplace=True)

    return results


def regress_edges(
    networks_df: pd.DataFrame,
    ordered_groups: Dict[str, List[str]],
    padjust_method: str = "BH",
    min_mean_edge: float = 0.0,
) -> pd.DataFrame:
    """
    Regression analysis of edges across ordered conditions.

    Performs linear regression on network edges from :func:`run_scorpion`
    output to identify edges that show significant trends across ordered
    conditions (e.g., disease progression: Normal -> Border -> Tumor).

    Parameters
    ----------
    networks_df : DataFrame
        A DataFrame output from :func:`run_scorpion` containing TF-target
        pairs as rows and network identifiers as columns.
    ordered_groups : dict of {str: list of str}
        A dictionary where each value is a list of column names in
        *networks_df*.  Keys represent ordered conditions (e.g.,
        ``{"Normal": ["P31--N", "P32--N"], "Border": ["P31--B",
        "P32--B"], "Tumor": ["P31--T", "P32--T"]}``).  The order of
        keys defines the progression (first to last).
    padjust_method : str, default="BH"
        Character specifying the p-value adjustment method for multiple
        testing correction (see :func:`test_edges`).  Default ``"BH"``
        (Benjamini-Hochberg FDR).
    min_mean_edge : float, default=0.0
        Numeric threshold for minimum mean absolute edge weight to
        include in testing.  Edges with mean absolute weight below this
        threshold are excluded.  Default 0 (no filtering).

    Returns
    -------
    result : DataFrame
        A DataFrame containing:

        - ``tf``: Transcription factor
        - ``target``: Target gene
        - ``slope``: Regression slope (change in edge weight per
          condition step)
        - ``intercept``: Regression intercept
        - ``rSquared``: R-squared value (proportion of variance
          explained)
        - ``fStatistic``: F-statistic for the regression
        - ``pValue``: Raw p-value for the slope
        - ``pAdj``: Adjusted p-value
        - ``meanEdge``: Overall mean edge weight across all conditions
        - One column per condition showing mean edge weight in that
          condition

    Notes
    -----
    This function performs simple linear regression for each edge,
    modeling edge weight as a function of an ordered categorical variable
    (coded as 0, 1, 2, ... for each condition level).

    The slope coefficient indicates the average change in edge weight per
    step along the ordered progression.  Positive slopes indicate
    increasing edge weights, negative slopes indicate decreasing edge
    weights.

    The function uses vectorized computations for efficiency with large
    datasets.

    See Also
    --------
    run_scorpion : Build per-group regulatory networks.
    test_edges : Pairwise or single-sample edge testing.

    Examples
    --------
    >>> # Define ordered progression: Normal -> Border -> Tumor
    >>> normal_nets = [c for c in nets.columns if c.endswith("--N")]
    >>> border_nets = [c for c in nets.columns if c.endswith("--B")]
    >>> tumor_nets = [c for c in nets.columns if c.endswith("--T")]
    >>> results_regression = regress_edges(
    ...     networks_df=nets,
    ...     ordered_groups={
    ...         "Normal": normal_nets,
    ...         "Border": border_nets,
    ...         "Tumor":  tumor_nets,
    ...     },
    ... )
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _regress_edges_impl(
            networks_df=networks_df,
            ordered_groups=ordered_groups,
            padjust_method=padjust_method,
            min_mean_edge=min_mean_edge,
        )


def _regress_edges_impl(
    networks_df, ordered_groups, padjust_method, min_mean_edge,
) -> pd.DataFrame:
    """Internal implementation of regress_edges(), called inside a warnings context."""
    # Validate
    if ordered_groups is None:
        raise ValueError("ordered_groups must be specified")
    if not isinstance(ordered_groups, dict):
        raise ValueError("ordered_groups must be a dict mapping condition names to columns")
    if len(ordered_groups) < 2:
        raise ValueError("ordered_groups must contain at least 2 conditions")
    if padjust_method not in P_ADJUST_METHODS:
        raise ValueError(
            f"padjust_method must be one of {P_ADJUST_METHODS}, got '{padjust_method}'"
        )
    all_cols = [c for cols in ordered_groups.values() for c in cols]
    missing = [c for c in all_cols if c not in networks_df.columns]
    if missing:
        raise ValueError(
            f"Some columns not found in networks_df: {', '.join(map(str, missing))}"
        )
    if "tf" not in networks_df.columns or "target" not in networks_df.columns:
        raise ValueError("networks_df must contain 'tf' and 'target' columns")

    condition_names = list(ordered_groups.keys())

    # Build predictor vector x (0, 1, 2, …) and combined edge matrix
    x_parts = []
    edge_parts = []
    for i, cols in enumerate(ordered_groups.values()):
        cols = list(cols)
        edge_parts.append(networks_df[cols].to_numpy(dtype=float))
        x_parts.append(np.full(len(cols), i, dtype=float))

    edge_matrix = np.hstack(edge_parts)   # (n_edges, n_samples)
    x = np.concatenate(x_parts)            # (n_samples,)

    mean_edge = np.nanmean(edge_matrix, axis=1)

    # Per-condition means
    cond_means = np.column_stack([
        np.nanmean(p, axis=1) for p in edge_parts
    ])

    # Filter by min_mean_edge
    idx = np.where(np.abs(mean_edge) >= min_mean_edge)[0]
    edge_matrix = edge_matrix[idx]
    mean_edge = mean_edge[idx]
    cond_means = cond_means[idx]
    tf_vals = networks_df["tf"].values[idx]
    target_vals = networks_df["target"].values[idx]

    # Vectorised linear regression
    mask = ~np.isnan(edge_matrix)
    edge_clean = np.where(mask, edge_matrix, 0.0)

    n_valid = mask.sum(axis=1).astype(float)

    sum_x = mask @ x
    sum_x2 = mask @ (x ** 2)
    sum_y = edge_clean.sum(axis=1)
    sum_y2 = (edge_clean ** 2).sum(axis=1)
    sum_xy = edge_clean @ x

    x_mean = sum_x / n_valid
    y_mean = sum_y / n_valid

    sxx = sum_x2 - n_valid * x_mean ** 2
    sxy = sum_xy - n_valid * x_mean * y_mean
    ss_tot = sum_y2 - n_valid * y_mean ** 2

    slopes = sxy / sxx
    intercepts = y_mean - slopes * x_mean

    ss_res = ss_tot - slopes ** 2 * sxx
    r_squared = 1 - ss_res / ss_tot
    df_res = n_valid - 2
    ms_res = ss_res / df_res
    f_stats = (ss_tot - ss_res) / ms_res
    pvalues = scipy_stats.f.sf(f_stats, dfn=1, dfd=df_res)

    bad = (n_valid < 3) | (sxx == 0) | (ms_res <= 0)
    slopes[bad] = np.nan
    intercepts[bad] = np.nan
    r_squared[bad] = np.nan
    f_stats[bad] = np.nan
    pvalues[bad] = np.nan

    p_adj = _p_adjust(pvalues, padjust_method)

    result = pd.DataFrame({
        "tf": tf_vals,
        "target": target_vals,
        "slope": slopes,
        "intercept": intercepts,
        "rSquared": r_squared,
        "fStatistic": f_stats,
        "pValue": pvalues,
        "pAdj": p_adj,
        "meanEdge": mean_edge,
    })
    for i, cname in enumerate(condition_names):
        result[f"mean{cname}"] = cond_means[:, i]

    result.reset_index(drop=True, inplace=True)
    return result


_MA_COLUMNS = [
    "tf", "target", "k", "log2FoldChange", "SE", "ciLow", "ciHigh",
    "zStatistic", "pValue", "pAdj", "Q", "iSquared", "tauSquared",
]


def ma_edges(
    edges_list: Union[Sequence[pd.DataFrame], Dict[str, pd.DataFrame]],
    method: str = "random",
    min_studies: int = 2,
    padjust_method: str = "BH",
    moderate_variance: bool = True,
    s0: Optional[float] = None,
) -> pd.DataFrame:
    """
    Meta-analysis of TF-target edges across studies.

    Performs a meta-analysis of TF-target edges across multiple studies
    using either a fixed-effect or DerSimonian-Laird random-effects model.
    Missing or non-finite effect sizes and standard errors are excluded
    from the corresponding study.  A TF-target pair is only counted as
    contributing to a study when both its effect size and SE are valid.

    Parameters
    ----------
    edges_list : list of DataFrame (or dict of DataFrame)
        One DataFrame per study, typically produced by :func:`test_edges`.
        Each must contain the columns ``tf``, ``target``,
        ``log2FoldChange`` and ``SE``.  At least two studies are required.
    method : str, default="random"
        Meta-analysis model.  Either ``"random"`` (DerSimonian-Laird
        random-effects) or ``"fixed"`` (inverse-variance fixed-effect).
    min_studies : int, default=2
        Minimum number of studies with valid numeric information required
        for a TF-target pair to be included.
    padjust_method : str, default="BH"
        P-value adjustment method for multiple testing correction (see
        :func:`test_edges`).  Default ``"BH"`` (Benjamini-Hochberg FDR).
    moderate_variance : bool, default=True
        Whether to apply SAM-style variance moderation to the
        meta-analysis SE.
    s0 : float, optional
        Variance-moderation fudge factor.  If None and
        ``moderate_variance=True``, the median of all valid
        meta-analysis SEs is used.

    Returns
    -------
    result : DataFrame
        A DataFrame containing:

        - ``tf``: Transcription factor
        - ``target``: Target gene
        - ``k``: Number of studies contributing to the meta-analysis
        - ``log2FoldChange``: Meta-analytic effect size
        - ``SE``: Meta-analysis standard error
        - ``ciLow``: Lower bound of the 95% confidence interval
        - ``ciHigh``: Upper bound of the 95% confidence interval
        - ``zStatistic``: Test statistic
        - ``pValue``: Raw p-value
        - ``pAdj``: Adjusted p-value
        - ``Q``: Cochran's Q heterogeneity statistic
        - ``iSquared``: I-squared heterogeneity (percentage)
        - ``tauSquared``: DerSimonian-Laird between-study variance

    See Also
    --------
    test_edges : Produces the per-study inputs.

    Examples
    --------
    >>> study1 = test_edges(nets1, test_type="two.sample",
    ...                     group1=tumor1, group2=normal1)
    >>> study2 = test_edges(nets2, test_type="two.sample",
    ...                     group1=tumor2, group2=normal2)
    >>> meta = ma_edges([study1, study2], method="random")
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _ma_edges_impl(
            edges_list=edges_list, method=method, min_studies=min_studies,
            padjust_method=padjust_method,
            moderate_variance=moderate_variance, s0=s0,
        )


def _ma_edges_impl(
    edges_list, method, min_studies, padjust_method, moderate_variance, s0,
) -> pd.DataFrame:
    """Internal implementation of ma_edges(), called inside a warnings context."""
    if method not in ("random", "fixed"):
        raise ValueError(f"method must be 'random' or 'fixed', got '{method}'")
    if padjust_method not in P_ADJUST_METHODS:
        raise ValueError(
            f"padjust_method must be one of {P_ADJUST_METHODS}, got '{padjust_method}'"
        )
    if isinstance(edges_list, dict):
        edges_list = list(edges_list.values())
    if not isinstance(edges_list, (list, tuple)) or len(edges_list) < 2:
        raise ValueError("edges_list must be a list of at least 2 DataFrames")
    required = ("tf", "target", "log2FoldChange", "SE")
    for s, x in enumerate(edges_list):
        missing = [c for c in required if c not in x.columns]
        if missing:
            raise ValueError(f"Study {s + 1} is missing columns: {', '.join(missing)}")

    # Stack all studies. A study contributes to a TF-target pair only when
    # both its effect and SE are valid (finite, SE > 0, finite weight).
    parts = []
    for x in edges_list:
        part = pd.DataFrame({
            "tf": x["tf"].values,
            "target": x["target"].values,
            "effect": pd.to_numeric(x["log2FoldChange"], errors="coerce").to_numpy(dtype=float),
            "SE": pd.to_numeric(x["SE"], errors="coerce").to_numpy(dtype=float),
        })
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long = long[long["tf"].notna() & long["target"].notna()]

    effect = long["effect"].to_numpy()
    se = long["SE"].to_numpy()
    w = 1.0 / se ** 2
    valid = np.isfinite(effect) & np.isfinite(se) & (se > 0) & np.isfinite(w) & (w > 0)
    long = long[valid].copy()
    long["w"] = w[valid]

    if len(long) == 0:
        return _empty_ma_result()

    # PASS 1: inverse-variance (fixed-effect) accumulators per TF-target
    long["wy"] = long["w"] * long["effect"]
    long["wy2"] = long["w"] * long["effect"] ** 2
    long["w2"] = long["w"] ** 2
    keys = ["tf", "target"]
    grouped = long.groupby(keys, sort=False)
    acc = grouped[["w", "wy", "wy2", "w2"]].sum()
    acc["k"] = grouped.size()

    acc = acc[(acc["k"] >= min_studies) & np.isfinite(acc["w"]) & (acc["w"] > 0)]
    if len(acc) == 0:
        return _empty_ma_result()

    k = acc["k"].to_numpy()
    sum_w = acc["w"].to_numpy()
    sum_wy = acc["wy"].to_numpy()
    sum_wy2 = acc["wy2"].to_numpy()
    sum_w2 = acc["w2"].to_numpy()

    fixed_log2fc = sum_wy / sum_w
    fixed_se = np.sqrt(1.0 / sum_w)

    # Cochran's Q
    Q = np.maximum(0.0, sum_wy2 - sum_wy ** 2 / sum_w)
    df = k - 1

    # DerSimonian-Laird tau^2
    C = sum_w - sum_w2 / sum_w
    tau2 = np.zeros(len(acc))
    tau_ok = (k > 1) & np.isfinite(C) & (C > 0)
    tau2[tau_ok] = np.maximum(0.0, (Q[tau_ok] - df[tau_ok]) / C[tau_ok])

    if method == "random":
        # PASS 2: random-effects weights using raw study-level SE
        tau2_s = pd.Series(tau2, index=acc.index, name="tau2")
        long_meta = long.join(tau2_s, on=keys, how="inner")
        wr = 1.0 / (long_meta["SE"] ** 2 + long_meta["tau2"])
        ok_w = np.isfinite(wr) & (wr > 0)
        long_meta = long_meta[ok_w].assign(
            wr=wr[ok_w], wry=wr[ok_w] * long_meta.loc[ok_w, "effect"],
        )
        racc = long_meta.groupby(keys, sort=False)[["wr", "wry"]].sum()
        racc = racc.reindex(acc.index, fill_value=0.0)
        sum_wr = racc["wr"].to_numpy()
        sum_wry = racc["wry"].to_numpy()

        meta_log2fc = np.zeros(len(acc))
        meta_se = np.zeros(len(acc))
        random_ok = np.isfinite(sum_wr) & (sum_wr > 0)
        meta_log2fc[random_ok] = sum_wry[random_ok] / sum_wr[random_ok]
        meta_se[random_ok] = np.sqrt(1.0 / sum_wr[random_ok])
    else:
        meta_log2fc = fixed_log2fc
        meta_se = fixed_se

    valid_effect = np.isfinite(meta_log2fc) & np.isfinite(meta_se) & (meta_se > 0)

    # I^2
    I2 = np.zeros(len(acc))
    i2_ok = np.isfinite(Q) & (Q > 0) & (df > 0)
    I2[i2_ok] = np.maximum(0.0, (Q[i2_ok] - df[i2_ok]) / Q[i2_ok]) * 100

    tf_idx = acc.index.get_level_values("tf")[valid_effect]
    target_idx = acc.index.get_level_values("target")[valid_effect]
    result = pd.DataFrame({
        "tf": np.asarray(tf_idx),
        "target": np.asarray(target_idx),
        "k": k[valid_effect].astype(int),
        "log2FoldChange": meta_log2fc[valid_effect],
        "SE": meta_se[valid_effect],
        "Q": Q[valid_effect],
        "iSquared": I2[valid_effect],
        "tauSquared": tau2[valid_effect],
    })
    if len(result) == 0:
        return _empty_ma_result()

    # Global SAM-style variance moderation: s0 is computed ONCE over all
    # meta-analysed edges (as in test_edges), not per TF.
    raw_se = result["SE"].to_numpy()
    valid_se = np.isfinite(raw_se) & (raw_se > 0)
    if moderate_variance and valid_se.any():
        if s0 is None:
            s0 = float(np.median(raw_se[valid_se]))
        moderated_se = raw_se + s0
    else:
        moderated_se = raw_se

    lfc = result["log2FoldChange"].to_numpy()
    ok = np.isfinite(lfc) & np.isfinite(moderated_se) & (moderated_se > 0)
    z = np.full(len(result), np.nan)
    z[ok] = lfc[ok] / moderated_se[ok]
    pval = np.full(len(result), np.nan)
    pval[ok] = 2 * scipy_stats.norm.cdf(-np.abs(z[ok]))
    zcrit = scipy_stats.norm.ppf(0.975)
    ci_low = np.full(len(result), np.nan)
    ci_high = np.full(len(result), np.nan)
    ci_low[ok] = lfc[ok] - zcrit * moderated_se[ok]
    ci_high[ok] = lfc[ok] + zcrit * moderated_se[ok]

    result["ciLow"] = ci_low
    result["ciHigh"] = ci_high
    result["zStatistic"] = z
    result["pValue"] = pval
    result["pAdj"] = _p_adjust(pval, padjust_method)

    result = result[_MA_COLUMNS]
    result = result.sort_values(["tf", "target"], kind="stable").reset_index(drop=True)
    return result


def _empty_ma_result() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=float) for c in _MA_COLUMNS})
    df["tf"] = df["tf"].astype(object)
    df["target"] = df["target"].astype(object)
    df["k"] = df["k"].astype(int)
    return df
