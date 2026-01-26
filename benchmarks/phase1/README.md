# Phase 1: AG News Text Classification

Tests ARU's ability to perform short-text classification on real-world news articles.

## Task

- **Dataset**: AG News (4 classes: World, Sports, Business, Sci/Tech)
- **Input**: Tokenized news headlines and snippets (max 100 tokens)
- **Output**: Category prediction
- **Samples**: 60,000 train / 7,600 test

## Running the Benchmark

```bash
python -m benchmarks.phase1.ag_news_benchmark
```

## Expected Results

ARU matches GRU performance on this standard text classification task, demonstrating competitiveness on mainstream NLP applications.

## Results

See [detailed report](report.md) for full analysis and results.
