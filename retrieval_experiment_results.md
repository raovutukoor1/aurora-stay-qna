# Retrieval experiment results

These are lexical coverage screening results; validate finalists with DeepEval.

| Rank | Run | Chunk | Overlap | Lambda | Fetch k | Proxy | >=.75 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | run_01 | 512 | 64 | 0.5 | 20 | 0.836 | 9/10 |
| 2 | run_03 | 512 | 64 | 0.55 | 30 | 0.821 | 8/10 |
| 3 | run_02 | 512 | 64 | 0.5 | 40 | 0.817 | 8/10 |
| 4 | run_04 | 512 | 64 | 0.6 | 40 | 0.816 | 7/10 |
| 5 | run_05 | 512 | 64 | 0.6 | 40 | 0.315 | 3/10 |

The detailed per-question diagnostics are in `retrieval_experiment_details.csv`.
