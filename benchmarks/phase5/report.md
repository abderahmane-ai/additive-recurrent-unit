# 📊 Phase 5: Adding Problem Benchmark Report

## Executive Summary

The Adding Problem is a classic stress test for RNNs, requiring models to sum two numbers separated by an arbitrary time lag. Both **GRU and ARU successfully solved the task**, achieving near-zero Mean Squared Error. This result confirms that ARU retains the general-purpose sequence modeling capabilities of GRU while offering specialized advantages in other domains.

---

## 🎯 Task Specification

| Metric | Value |
|--------|-------|
| **Task** | Sum two marked numbers in a sequence |
| **Sequence Length** | 200 timesteps |
| **Input** | 2 channels (Value in [0,1], Marker in {0,1}) |
| **Constraint** | Exactly two markers per sequence |

---

## 🏆 Performance Results

### Test Error Metrics (Lower is Better)

| Rank | Model | Test MSE | Status | Parameters |
|------|-------|----------|--------|------------|
| 🥇 | **GRU** | **0.0009** | **Solved** | 50,433 |
| 🥈 | **ARU** | **0.0013** | **Solved** | 50,817 |
| 🥉 | **LSTM** | 0.1631 | Learning | 67,201 |
| ❌ | **RNN** | 0.1661 | Learning | 16,897 |

### Key Observations

✅ **Parity with State-of-the-Art** - ARU's performance (MSE 0.0013) is effectively tied with GRU (MSE 0.0009). The difference is negligible for this task.
✅ **Robust Long-Term Memory** - To solve this, the model must remember the first number (seen early in the sequence) until the end (T=200). ARU's persistence gate handles this effortlessly.
⚠️ **LSTM Training Speed** - In this configuration, LSTM was slower to converge, still in the "Learning" phase when ARU and GRU had solved the task.

---

## 🔬 Technical Analysis

### The "Standard Task" Baseline
Unlike the Copy Task (Phase 2) or Counting (Phase 3), the Adding Problem does not require *pure* accumulation or *perfect* symbolic retention. It requires **latching**: holding a value until it's needed.

Both GRU and ARU are excellent at latching:
*   **GRU**: Sets $z \approx 0$ to hold state.
*   **ARU**: Sets $\pi \approx 1, \alpha \approx 0$ to hold state.

This benchmark serves as a "sanity check" to ensure that ARU's specialized architecture hasn't sacrificed general competency.

---

## Conclusion

Phase 5 demonstrates that ARU is a **safe general-purpose replacement** for GRU. It matches GRU's performance on standard long-term dependency tasks while significantly outperforming it on counting and strict memorization tasks (as seen in Phases 2 and 3).