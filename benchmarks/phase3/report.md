# 📊 Phase 3: Counting Task Benchmark Report

## Executive Summary

The Counting Task evaluates a model's ability to maintain a precise running count of specific events within a noisy sequence. **ARU demonstrated superior performance**, achieving a Test Mean Absolute Error (MAE) of **0.02**, significantly outperforming GRU (0.25) and LSTM (0.41). This confirms ARU's ability to perform true additive accumulation.

---

## 🎯 Task Specification

| Metric | Value |
|--------|-------|
| **Task** | Count occurrences of '1's in binary sequence |
| **Sequence Length** | 100 timesteps |
| **Event Density** | 10% (approx. 10 events per sequence) |
| **Goal** | Regress final total count |

---

## 🏆 Performance Results

### Test Error Metrics (Lower is Better)

| Rank | Model | Test MSE | Test MAE | Parameters |
|------|-------|----------|----------|------------|
| 🥇 | **ARU** | **0.0014** | **0.02** | 12,865 |
| 🥈 | **GRU** | 0.1405 | 0.25 | 12,737 |
| 🥉 | **LSTM** | 0.4061 | 0.41 | 16,961 |
| ❌ | **RNN** | 1.6678 | 0.97 | 4,289 |

*Baseline MSE (Predicting Mean): 9.8259*

### Key Observations

✅ **ARU acts as a digital counter** - An MAE of 0.02 implies the model is correct almost 100% of the time, with negligible variance.
✅ **GRU limitation exposed** - While GRU "solved" the task (MAE 0.25), its error is 12.5x higher than ARU. This is due to the "weighted average" update rule, which makes precise integer addition difficult.
✅ **LSTM struggles** - Surprisingly, LSTM performed worse than GRU, likely due to the complexity of its gating mechanism interfering with simple accumulation.

---

## 🔬 Technical Analysis

### Why ARU Wins
The counting task requires the operation: `state = state + 1`.

*   **ARU**: Sets Persistence ($\pi$) $\approx 1$ and Accumulation ($\alpha$) $\approx 1$ (when input is 1). The equation becomes $h_t \approx h_{t-1} + 1$.
*   **GRU**: The update $h_t = (1-z)h_{t-1} + z \tilde{h}$ forces a trade-off. To add information ($z > 0$), it *must* decay the previous state. It cannot simply "add".

---

## Conclusion

Phase 3 provides empirical proof of ARU's "Additive" hypothesis. On tasks requiring discrete accumulation, ARU is not just better; it is structurally superior to standard gated RNNs.