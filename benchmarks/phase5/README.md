# Phase 5: Adding Problem

Tests the ability to latch onto specific values and sum them—a classic benchmark for evaluating memory retention.

## Task

- **Challenge**: Add two numbers indicated by special markers in a long sequence
- **Sequence Length**: 100+ timesteps
- **Target Values**: Randomly positioned with markers
- **Goal**: Memorize values and compute sum

## Running the Benchmark

```bash
python -m benchmarks.phase5.adding_problem_benchmark
```

## Expected Results

ARU ties with GRU on this task, both achieving strong performance. The task requires selective attention and memory, which both architectures handle effectively.

## Results

See [detailed report](report.md) for full analysis and results.
