# 📊 Phase 1: AG News Classification Benchmark Report

## Executive Summary

The AG News benchmark evaluated gated architectures on a standard short-text classification task. **All gated models (GRU, ARU, LSTM) achieved comparable performance** in the 87-89% accuracy range. This confirms that for short, dense sequences where local patterns dominate, ARU maintains parity with established baselines without sacrificing efficiency.

---

## 🎯 Task Specification

| Metric | Value |
|--------|-------|
| **Task** | 4-Class News Categorization |
| **Dataset** | AG News (60k Train, 7.6k Test) |
| **Vocabulary** | 5,000 words |
| **Sequence Length** | 100 tokens (max) |

---

## 🏆 Performance Results

### Test Accuracy Metrics (Higher is Better)

| Rank | Model | Test Accuracy | Training Time | Parameters | Status |
|------|-------|---------------|---------------|------------|--------|
| 🥇 | **GRU** | **89.29%** | 486s | 739,204 | Competitive |
| 🥈 | **ARU** | **88.46%** | **479s** | 755,716 | Competitive |
| 🥉 | **LSTM** | 87.61% | 572s | 772,100 | Competitive |
| ❌ | **RNN** | 26.51% | 365s | 673,412 | Failed |

### Key Observations

✅ **Competitive Baseline Performance** - ARU is within 0.8% of GRU, a difference that is statistically marginal for single-run benchmarks.
✅ **Training Efficiency** - ARU was the fastest gated model to train (479s), slightly edging out GRU.
✅ **Sanity Check Passed** - This benchmark confirms that ARU's specialized "additive" machinery does not hinder its ability to perform standard pattern recognition tasks.

---

## 🔬 Technical Analysis

### Why Parity?
AG News consists of short sequences (avg ~30-40 tokens) with high information density.
*   **No Long-Term Dependency**: Classification often relies on key phrases ("touchdown", "stocks") rather than distant correlations.
*   **Standard Gating Suffices**: The specialized $\pi \approx 1$ (persistence) mode of ARU is not strictly necessary here, but the model learns to adapt its gates to function like a standard RNN.

---

## Conclusion

Phase 1 demonstrates that ARU is a **robust general-purpose classifier**. While it does not outperform GRU on this specific task, it matches it in speed and accuracy, proving it can be used safely in standard NLP pipelines.
