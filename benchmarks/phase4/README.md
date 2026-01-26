# Phase 4: Sparse Event Counting

Tests the ability to detect and count rare events in long sequences—combining pattern recognition with accumulation.

## Task

- **Challenge**: Count rare "signal" patterns in noisy sequences
- **Sequence Length**: 200 timesteps
- **Event Rarity**: 2-5% occurrence rate
- **Goal**: Accurate sparse event detection and summation

## Running the Benchmark

```bash
python -m benchmarks.phase4.sparse_event_counting_benchmark
```

## Expected Results

ARU excels at maintaining accurate counts even when events are rare and separated by long intervals. Baselines tend to lose precision as sequence length increases.

## Results

See [detailed report](report.md) for full analysis and results.
