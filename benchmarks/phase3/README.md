# Phase 3: Counting Task

Tests the ability to count occurrences of a signal in a sequence—the ideal task for ARU's additive accumulation design.

## Task

- **Challenge**: Count how many "1"s appear in a binary sequence
- **Sequence Length**: 100 timesteps
- **Signal Density**: ~10% (sparse counting)
- **Goal**: Precise integer accumulation

## Running the Benchmark

```bash
python -m benchmarks.phase3.counting_benchmark
```

## Expected Results

ARU achieves near-perfect counting accuracy (~99%) while baselines struggle with accumulation errors. This task directly validates ARU's core architectural advantage: true additive memory.

## Results

See [detailed report](report.md) for full analysis and results.
