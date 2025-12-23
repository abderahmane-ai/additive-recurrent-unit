"""
Statistical analysis utilities.
"""

import numpy as np
from typing import List, Tuple, Dict


def paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Perform paired t-test. Returns (t_statistic, p_value)."""
    from scipy import stats
    return stats.ttest_rel(a, b)


def compute_confidence_interval(
    data: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence interval for mean."""
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * se
    
    return mean - margin, mean + margin


def format_results(
    all_results: Dict[str, List[float]],
    baseline: str = 'SPMN'
) -> str:
    """Format statistical results as a table."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'Model':<10} {'Mean':>10} {'Std':>10} {'95% CI':>20} {'vs ' + baseline:>15}")
    lines.append("=" * 70)
    
    baseline_accs = all_results.get(baseline, [])
    
    for name, accs in all_results.items():
        mean = np.mean(accs)
        std = np.std(accs, ddof=1)
        ci_low, ci_high = compute_confidence_interval(accs)
        
        if name == baseline:
            diff_str = "-"
        elif baseline_accs:
            diff = mean - np.mean(baseline_accs)
            _, p = paired_ttest(accs, baseline_accs)
            sig = "*" if p < 0.05 else ""
            diff_str = f"{diff:+.2f}% (p={p:.4f}){sig}"
        else:
            diff_str = "N/A"
        
        lines.append(f"{name:<10} {mean:>9.2f}% {std:>9.2f} [{ci_low:>6.1f}, {ci_high:>6.1f}] {diff_str:>15}")
    
    lines.append("=" * 70)
    lines.append("* indicates p < 0.05 (statistically significant)")
    
    return "\n".join(lines)
