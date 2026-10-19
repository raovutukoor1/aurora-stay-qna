# 120/16 held-out evaluation repeatability findings

Evaluation repeated 3 times with the same configuration: K=5, temperature=0, top_p=1, max_tokens=512, 120-token chunks, 16-token overlap, and the existing FAISS index.

## Run comparison

| Run | Passed | Failed | Pass rate | Average composite |
|---:|---:|---:|---:|---:|
| 1 | 7 | 3 | 70.0% | 0.8650 |
| 2 | 7 | 3 | 70.0% | 0.8637 |
| 3 | 7 | 3 | 70.0% | 0.8630 |

Pass-rate consistency: **consistent**.
Stable passing cases across all runs: 1, 2, 3, 7, 8, 9, 10.
Stable failing cases across all runs: 4, 5, 6.
Cases whose pass/fail status varied: none.

## Maximum case-level metric ranges across runs

- faithfulness: maximum case-level range 0.0703
- contextual recall: maximum case-level range 0.0136
- contextual precision: maximum case-level range 0.0307
- answer relevancy: maximum case-level range 0.1502
- composite: maximum case-level range 0.0108

## Interpretation

The three runs use deterministic generation settings, but the evaluation uses model-judged metrics. Therefore, small score movement is expected even when the retrieved context and generated answer are unchanged. The pass/fail conclusion is considered operationally consistent when the pass rate and case-level pass sets remain unchanged.

The detailed case-level results are saved as `reports/notebook_120_16_repeat_1.csv`, `reports/notebook_120_16_repeat_2.csv`, and `reports/notebook_120_16_repeat_3.csv`.
