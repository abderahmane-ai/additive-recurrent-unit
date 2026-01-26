# Phase 7: ETT Time Series Forecasting

Tests ARU on real-world industrial time series data—the Electricity Transformer Temperature (ETT) dataset, a standard benchmark for long-term forecasting.

## Task

- **Dataset**: ETTh1 (Electricity Transformer Temperature - Hourly)
- **Input Features**: 7 (6 power loads + 1 target temperature)
- **Sequence Length**: 96 timesteps (lookback window)
- **Prediction Horizon**: 720 timesteps (30 hours ahead)
- **Challenge**: Long-horizon forecasting with non-stationarity

## Running the Benchmark

```bash
python -m benchmarks.phase7.ett_benchmark
```

## Expected Results

ARU achieves 52.7% improvement over GRU on this challenging 720-step prediction horizon, demonstrating its effectiveness on real-world industrial data with complex multi-scale temporal patterns.

## Results

See [detailed report](report.md) for full analysis and results.
