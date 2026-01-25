"""
Statistical analysis utilities for benchmark experiments.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import stats


def paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Perform paired t-test (parametric).
    
    Returns:
        (t_statistic, p_value)
    """
    return stats.ttest_rel(a, b)


def wilcoxon_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric).
    
    Returns:
        (statistic, p_value)
    """
    try:
        return stats.wilcoxon(a, b)
    except ValueError:
        # If all differences are zero
        return 0.0, 1.0


def cohens_d(a: List[float], b: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    
    Interpretation:
        |d| < 0.2: negligible
        |d| < 0.5: small
        |d| < 0.8: medium
        |d| >= 0.8: large
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    
    mean_diff = np.mean(a_arr) - np.mean(b_arr)
    pooled_std = np.sqrt((np.var(a_arr, ddof=1) + np.var(b_arr, ddof=1)) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return mean_diff / pooled_std


def compute_confidence_interval(
    data: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for mean.
    
    Returns:
        (lower_bound, upper_bound)
    """
    n = len(data)
    if n < 2:
        mean = np.mean(data)
        return mean, mean
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * se
    
    return mean - margin, mean + margin


def compute_statistics(data: List[float]) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a list of values.
    
    Returns:
        Dict with mean, std, median, min, max, ci_lower, ci_upper
    """
    data_arr = np.array(data)
    ci_lower, ci_upper = compute_confidence_interval(data)
    
    return {
        'mean': float(np.mean(data_arr)),
        'std': float(np.std(data_arr, ddof=1)) if len(data) > 1 else 0.0,
        'median': float(np.median(data_arr)),
        'min': float(np.min(data_arr)),
        'max': float(np.max(data_arr)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }


def compare_models(
    model_a_results: List[float],
    model_b_results: List[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    lower_is_better: bool = True
) -> Dict:
    """
    Perform comprehensive statistical comparison between two models.
    
    Args:
        model_a_results: List of metric values for model A
        model_b_results: List of metric values for model B
        model_a_name: Name of model A
        model_b_name: Name of model B
        lower_is_better: If True, lower values are better (e.g., MSE, MAE)
    
    Returns:
        Dict with comparison statistics
    """
    a_stats = compute_statistics(model_a_results)
    b_stats = compute_statistics(model_b_results)
    
    # Statistical tests
    t_stat, t_pval = paired_ttest(model_a_results, model_b_results)
    w_stat, w_pval = wilcoxon_test(model_a_results, model_b_results)
    effect_size = cohens_d(model_a_results, model_b_results)
    
    # Determine winner
    if lower_is_better:
        improvement = ((b_stats['mean'] - a_stats['mean']) / b_stats['mean']) * 100
        winner = model_a_name if a_stats['mean'] < b_stats['mean'] else model_b_name
    else:
        improvement = ((a_stats['mean'] - b_stats['mean']) / b_stats['mean']) * 100
        winner = model_a_name if a_stats['mean'] > b_stats['mean'] else model_b_name
    
    # Significance
    is_significant_t = t_pval < 0.05
    is_significant_w = w_pval < 0.05
    
    return {
        f'{model_a_name}_stats': a_stats,
        f'{model_b_name}_stats': b_stats,
        'mean_difference': a_stats['mean'] - b_stats['mean'],
        'improvement_pct': improvement,
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pval,
        'cohens_d': effect_size,
        'is_significant_ttest': is_significant_t,
        'is_significant_wilcoxon': is_significant_w,
        'winner': winner,
    }


def format_comparison_table(
    all_results: Dict[str, List[float]],
    metric_name: str = "Metric",
    baseline: str = 'GRU',
    lower_is_better: bool = True
) -> str:
    """
    Format statistical comparison results as a table.
    
    Args:
        all_results: Dict mapping model names to lists of metric values
        metric_name: Name of the metric being compared
        baseline: Name of baseline model for comparisons
        lower_is_better: If True, lower values are better
    
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Model':<10} {'Mean':>10} {'Std':>10} {'95% CI':>25} {'vs ' + baseline:>30}")
    lines.append("=" * 100)
    
    baseline_results = all_results.get(baseline, [])
    
    # Sort by mean (best first)
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: np.mean(x[1]),
        reverse=not lower_is_better
    )
    
    for name, results in sorted_models:
        stats_dict = compute_statistics(results)
        mean = stats_dict['mean']
        std = stats_dict['std']
        ci_low = stats_dict['ci_lower']
        ci_high = stats_dict['ci_upper']
        
        if name == baseline or not baseline_results:
            comparison_str = "-"
        else:
            comparison = compare_models(
                results, baseline_results, name, baseline, lower_is_better
            )
            
            improvement = comparison['improvement_pct']
            t_pval = comparison['t_pvalue']
            w_pval = comparison['wilcoxon_pvalue']
            cohens = comparison['cohens_d']
            
            # Significance markers
            sig_t = "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else ""
            sig_w = "‡" if w_pval < 0.05 else ""
            
            comparison_str = f"{improvement:+.2f}% (p={t_pval:.4f}{sig_t}{sig_w}, d={cohens:.2f})"
        
        lines.append(
            f"{name:<10} {mean:>10.4f} {std:>10.4f} "
            f"[{ci_low:>8.4f}, {ci_high:>8.4f}] {comparison_str:>30}"
        )
    
    lines.append("=" * 100)
    lines.append("* p < 0.05, ** p < 0.01 (t-test), ‡ p < 0.05 (Wilcoxon)")
    lines.append("d = Cohen's d effect size (|d| >= 0.8 is large)")
    
    return "\n".join(lines)


def format_results(
    all_results: Dict[str, List[float]],
    baseline: str = 'GRU'
) -> str:
    """
    Legacy function for backward compatibility.
    Format statistical results as a table (assumes higher is better, like accuracy).
    """
    return format_comparison_table(
        all_results,
        metric_name="Accuracy",
        baseline=baseline,
        lower_is_better=False
    )
