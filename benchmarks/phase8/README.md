# Phase 8: Multi-Scale Pattern Recognition

## Overview

This benchmark tests the ability to simultaneously track patterns at multiple timescales—a task that naturally favors hierarchical architectures.

## Task Description

**Multi-Scale Pattern Counting**: Models must detect and count pattern completions at three different temporal frequencies:

- **Fast patterns**: Every 5 timesteps
- **Medium patterns**: Every 20 timesteps  
- **Slow patterns**: Every 80 timesteps

The model outputs a cumulative count of all completed patterns at each timestep.

## Why This Tests Hierarchical Processing

Single-scale RNNs face a fundamental tension:
- High persistence (π ≈ 1) → Good for slow patterns, but slow to react to fast patterns
- Low persistence (π ≈ 0.5) → Good for fast patterns, but forgets slow patterns

Hierarchical architectures resolve this by specializing layers:
- **Fast layer**: Low persistence, high reactivity
- **Medium layer**: Balanced dynamics
- **Slow layer**: High persistence, stable long-term memory

## Expected Results

**HARU should outperform ARU** because:
1. Each layer can optimize for its timescale independently
2. No single hidden state trying to balance competing temporal dynamics
3. Natural separation of fast transients from slow trends

## Running the Benchmark

```bash
python -m benchmarks.phase8.multi_scale_pattern_benchmark
```

## Metrics

- **MSE**: Mean squared error on count predictions
- **MAE**: Mean absolute error (primary metric)
- **Parameters**: Model size for fair comparison
