# Well embedding evaluation

| Embedding | Task | Metric | Score |
| --- | --- | --- | --- |
| mae | cross-guide | mrr | 1.000000 |
| mae | same-guide | map | 1.000000 |
| diffmae | cross-guide | mrr | 0.500000 |
| diffmae | same-guide | map | 0.333333 |

Query wells come from held-out plates; the reference gallery is train.
Activity and replication additionally use matched reference wells; see protocol.json.
Coverage, excluded reasons and permutation resolution must accompany scores.
