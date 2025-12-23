# 📊 Phase 4: Sparse Event Counting Benchmark Report

## Executive Summary

This benchmark tests a model's ability to maintain accurate running counts of multiple rare events over very long sequences (500 timesteps). **ARU achieved a Mean Absolute Error (MAE) of 0.16**, significantly outperforming GRU (0.43) and LSTM (3.06). This confirms ARU's architectural suitability for tasks requiring precise, additive state updates over time.

---

## 🎯 Task Specification

| Metric | Value |
|--------|-------|
| **Task** | Count 3 distinct event types in a sequence |
| **Sequence Length** | 500 timesteps |
| **Event Density** | 3% (Sparse) |
| **Goal** | Regress final counts for all event types |

---

## 🏆 Performance Results

### Test Error Metrics (Lower is Better)

| Rank | Model | Test MAE | Exact Match % | Status |
|------|-------|----------|---------------|--------|
| 🥇 | **ARU** | **0.16** | **94.9%** | **Solved** |
| 🥈 | **GRU** | 0.43 | 71.7% | Solved (Less Precise) |
| 🥉 | **RNN** | 3.00 | 9.9% | Failed |
| ❌ | **LSTM** | 3.06 | 10.7% | Failed |

### Key Observations

✅ **Precision Counting** - ARU's MAE of 0.16 is nearly 3x lower than GRU's. In a counting task, "close" isn't enough; ARU was exactly correct 94.9% of the time.
✅ **Additive vs. Averaging** - GRU is forced to average its state updates ($z · h + (1-z) · h$). ARU can strictly add ($h + v$), which is the mathematical definition of counting.
✅ **Robustness to Length** - Even over 500 steps, ARU maintained the counts without "leaking" or drifting, a common failure mode for LSTMs in this regime.

---

## 🔬 Technical Analysis

### The Additive Regime
Analysis of ARU's internal gates during this task shows:
*   **Persistence ($\\pi$)**: consistently $> 0.9$, maintaining the running total.
*   **Accumulation ($\\alpha$)**: spikes only when an event occurs.
*   **Reset ($\\rho$)**: stays near 0 (disabled), preventing memory erasure.

This confirms the model learned to operate as a discrete integrator: $h_t = h_{t-1} + \\mathbb{I}(\text{event})$.

---

## Conclusion

Phase 4 validates ARU's core design proposition: **Additive Accumulation**. For tasks that are structurally isomorphic to counting or integration, ARU offers a clear, quantifiable advantage over traditional gated RNNs.