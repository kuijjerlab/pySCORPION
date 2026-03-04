"""
Statistical testing of network edges: testEdges and regressEdges.

Matches the R SCORPION package's testEdges() and regressEdges() exactly,
including SAM-style variance moderation, empirical null correction,
and vectorised linear regression.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import List, Optional, Dict
import warnings


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


def _p_adjust_bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg p-value adjustment (matches R's p.adjust(method='BH'))."""
    p = np.asarray(pvalues, dtype=float).copy()
    n = len(p)
    nan_mask = np.isnan(p)
    valid = ~nan_mask
    p_valid = p[valid]
    n_valid = len(p_valid)
    if n_valid == 0:
        return p

    order = np.argsort(p_valid)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n_valid + 1)

    adjusted = p_valid * n_valid / ranks
    # Enforce monotonicity (cumulative minimum in reverse rank order)
    adjusted_sorted = adjusted[np.argsort(-ranks)]
    cum_min = np.minimum.accumulate(adjusted_sorted)
    adjusted = cum_min[np.argsort(np.argsort(-ranks))]
    adjusted = np.clip(adjusted, 0, 1)

    out = np.full(n, np.nan)
    out[valid] = adjusted
    return out


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
    se = sd_edge / np.sqrt(n_valid)

    if moderate_variance:
        if s0 is None:
            s0 = float(np.nanmedian(se))
        se = se + s0

    t_stat = mean_edge / se
    df = n_valid - 1

    if alternative == "two.sided":
        pval = 2 * scipy_stats.t.sf(np.abs(t_stat), df)
    elif alternative == "greater":
        pval = scipy_stats.t.sf(t_stat, df)
    else:  # less
        pval = scipy_stats.t.cdf(t_stat, df)

    bad = (n_valid < 2) | np.isnan(sd_edge)
    if not moderate_variance:
        bad |= (sd_edge == 0)
    t_stat[bad] = np.nan
    pval[bad] = np.nan

    return pd.DataFrame({
        "tf": networks_df["tf"].values,
        "target": networks_df["target"].values,
        "meanEdge": mean_edge,
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

    # log2 fold change via pnorm (matches R: pnorm(x, log.p=TRUE))
    log_p1 = scipy_stats.norm.logcdf(mean1)
    log_p2 = scipy_stats.norm.logcdf(mean2)
    log2fc = (log_p1 - log_p2) / np.log(2)

    # Filter by minLog2FC
    keep = np.abs(log2fc) >= min_log2fc
    idx = np.where(keep)[0]
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

    se = np.sqrt(var1 / n1 + var2 / n2)

    if moderate_variance:
        if s0 is None:
            s0 = float(np.nanmedian(se))
        se = se + s0

    t_stat = diff_mean / se

    # Welch-Satterthwaite degrees of freedom
    df = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    )

    if alternative == "two.sided":
        pval = 2 * scipy_stats.t.sf(np.abs(t_stat), df)
    elif alternative == "greater":
        pval = scipy_stats.t.sf(t_stat, df)
    else:
        pval = scipy_stats.t.cdf(t_stat, df)

    se_before = np.sqrt(var1 / n1 + var2 / n2)
    no_mod = np.bool_(not moderate_variance)
    bad = (
        (n1 < 2) | (n2 < 2)
        | np.isnan(var1) | np.isnan(var2)
        | (no_mod & ((var1 == 0) | (var2 == 0) | (se_before == 0)))
        | np.isnan(se)
    )
    t_stat[bad] = np.nan
    pval[bad] = np.nan

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    cohens_d = diff_mean / pooled_sd
    cohens_d[(pooled_sd == 0) | np.isnan(pooled_sd)] = np.nan

    return pd.DataFrame({
        "tf": tf_vals,
        "target": target_vals,
        "meanGroup1": mean1,
        "meanGroup2": mean2,
        "diffMean": diff_mean,
        "cohensD": cohens_d,
        "log2FoldChange": log2fc,
        "meanEdge": mean_edge,
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

    log_p1 = scipy_stats.norm.logcdf(mean1)
    log_p2 = scipy_stats.norm.logcdf(mean2)
    log2fc = (log_p1 - log_p2) / np.log(2)

    keep = np.abs(log2fc) >= min_log2fc
    idx = np.where(keep)[0]
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
    se = sd_diff / np.sqrt(n_valid)

    if moderate_variance:
        if s0 is None:
            s0 = float(np.nanmedian(se))
        se = se + s0

    t_stat = diff_mean / se
    df = n_valid - 1

    if alternative == "two.sided":
        pval = 2 * scipy_stats.t.sf(np.abs(t_stat), df)
    elif alternative == "greater":
        pval = scipy_stats.t.sf(t_stat, df)
    else:
        pval = scipy_stats.t.cdf(t_stat, df)

    bad = (n_valid < 2) | np.isnan(sd_diff)
    if not moderate_variance:
        bad |= (sd_diff == 0)
    t_stat[bad] = np.nan
    pval[bad] = np.nan

    cohens_d = diff_mean / sd_diff
    cohens_d[(sd_diff == 0) | np.isnan(sd_diff)] = np.nan

    return pd.DataFrame({
        "tf": tf_vals,
        "target": target_vals,
        "meanGroup1": mean1,
        "meanGroup2": mean2,
        "diffMean": diff_mean,
        "cohensD": cohens_d,
        "log2FoldChange": log2fc,
        "meanEdge": mean_edge,
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
        testing correction.  Default ``"BH"`` (Benjamini-Hochberg FDR).
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

    Returns
    -------
    result : DataFrame
        A DataFrame containing:

        - ``tf``: Transcription factor
        - ``target``: Target gene
        - ``meanEdge``: Mean edge weight
        - ``tStatistic``: Test statistic
        - ``pValue``: Raw p-value
        - ``pAdj``: Adjusted p-value
        - For two-sample tests: ``meanGroup1``, ``meanGroup2``,
          ``diffMean`` (Group1 - Group2), ``cohensD``,
          ``log2FoldChange``

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

    Edges are tested independently, and p-values are adjusted for
    multiple testing using the specified method.

    The function uses fully vectorized computations for efficiency,
    making it suitable for large-scale analyses with millions of edges.
    T-statistics and p-values are calculated using matrix operations
    without iteration.

    See Also
    --------
    run_scorpion : Build per-group regulatory networks.
    regress_edges : Regression analysis across ordered conditions.

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
            empirical_null=empirical_null,
        )


def _test_edges_impl(
    networks_df, test_type, group1, group2, paired,
    alternative, padjust_method, min_log2fc,
    moderate_variance, empirical_null,
) -> pd.DataFrame:
    """Internal implementation of test_edges(), called inside a warnings context."""
    # Normalize parameter values (accept both Python-style and R-style)
    test_type = test_type.replace("_", ".")
    alternative = alternative.replace("_", ".")

    # Validate
    if test_type not in ("single", "two.sample"):
        raise ValueError(f"test_type must be 'single' or 'two.sample', got '{test_type}'")
    if alternative not in ("two.sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two.sided', 'greater', or 'less'")
    if group1 is None:
        raise ValueError("group1 must be specified")
    missing1 = set(group1) - set(networks_df.columns)
    if missing1:
        raise ValueError(f"group1 columns not found in networks_df: {missing1}")
    if test_type == "two.sample":
        if group2 is None:
            raise ValueError("group2 must be specified for two.sample test")
        missing2 = set(group2) - set(networks_df.columns)
        if missing2:
            raise ValueError(f"group2 columns not found in networks_df: {missing2}")
        if paired and len(group1) != len(group2):
            raise ValueError("For paired tests, group1 and group2 must have the same length")
    if paired and test_type == "single":
        raise ValueError("Paired tests require test_type='two.sample'")
    if "tf" not in networks_df.columns or "target" not in networks_df.columns:
        raise ValueError("networks_df must contain 'tf' and 'target' columns")

    # Dispatch
    if test_type == "single":
        results = _test_edges_single(
            networks_df, group1, alternative,
            moderate_variance, s0=None,
        )
    elif paired:
        results = _test_edges_paired(
            networks_df, group1, group2, alternative, min_log2fc,
            moderate_variance, s0=None,
        )
    else:
        results = _test_edges_two_sample(
            networks_df, group1, group2, alternative, min_log2fc,
            moderate_variance, s0=None,
        )

    # Empirical null correction (Efron's method)
    if empirical_null:
        t_vals = results["tStatistic"].to_numpy(dtype=float)
        valid_mask = np.isfinite(t_vals)
        if np.sum(valid_mask) > 100:
            null_center = np.median(t_vals[valid_mask])
            null_scale = scipy_stats.median_abs_deviation(
                t_vals[valid_mask], scale="normal"
            )
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
    results["pAdj"] = _p_adjust_bh(results["pValue"].to_numpy())
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
        testing correction.  Default ``"BH"`` (Benjamini-Hochberg FDR).
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
    if ordered_groups is None or len(ordered_groups) < 2:
        raise ValueError("ordered_groups must contain at least 2 conditions")
    all_cols = [c for cols in ordered_groups.values() for c in cols]
    missing = set(all_cols) - set(networks_df.columns)
    if missing:
        raise ValueError(f"Columns not found in networks_df: {missing}")
    if "tf" not in networks_df.columns or "target" not in networks_df.columns:
        raise ValueError("networks_df must contain 'tf' and 'target' columns")

    condition_names = list(ordered_groups.keys())
    n_conditions = len(condition_names)

    # Build predictor vector x (0, 1, 2, …) and combined edge matrix
    x_parts = []
    edge_parts = []
    for i, (cname, cols) in enumerate(ordered_groups.items()):
        part = networks_df[cols].to_numpy(dtype=float)
        edge_parts.append(part)
        x_parts.append(np.full(len(cols), i, dtype=float))

    edge_matrix = np.hstack(edge_parts)   # (n_edges, n_samples)
    x = np.concatenate(x_parts)            # (n_samples,)

    mean_edge = np.nanmean(edge_matrix, axis=1)

    # Per-condition means
    cond_means = np.column_stack([
        np.nanmean(p, axis=1) for p in edge_parts
    ])

    # Filter by min_mean_edge
    keep = np.abs(mean_edge) >= min_mean_edge
    idx = np.where(keep)[0]
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

    p_adj = _p_adjust_bh(pvalues)

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
