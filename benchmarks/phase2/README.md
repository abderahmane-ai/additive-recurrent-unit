# Phase 2: Copy Task

Tests ARU's ability to memorize and reproduce a sequence after a long delay—the classic long-term memory challenge for RNNs.

## Task

- **Challenge**: Memorize 10 symbols, wait 50 steps, reproduce
- **Sequence Length**: 70 timesteps (10 input + 50 delay + 10 output)
- **Vocabulary**: 8 symbols + markers
- **Goal**: Perfect sequence reproduction

## Running the Benchmark

```bash
python -m benchmarks.phase2.copy_task_benchmark
```

## Expected Results

ARU solves this task with ~74% sequence accuracy while GRU, LSTM, and RNN completely fail (0% sequence accuracy). This demonstrates ARU's unique ability to retain information without decay over long intervals.

## Results

See [detailed report](report.md) for full analysis and results.
