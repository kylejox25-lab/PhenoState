# Frozen image embedding retrieval

Query split: val; gallery: train; preprocessing: raw; metric: cosine.

| Model | Task | Metric | Score | 95% CI | Evaluable queries |
| --- | --- | --- | --- | --- | --- |
| mae | cross-guide | mrr | 1.000000 | NA | 8/8 |
| mae | same-guide | map | 1.000000 | NA | 8/8 |

Bootstrap intervals describe gene sampling for the supplied checkpoints. Training-seed variation is not estimated.
The reference gallery participated in self-supervised training. These scores evaluate new-plate queries.
See coverage.json for wells lost before scoring, queries.csv for excluded tasks and paired_differences.csv for model differences.
