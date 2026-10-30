# Probe pilot: colab-l4-run1

360 RAG prompts from 20 facts; target = greedy answer labelled true (144 of 360). 5 folds split by fact; every prompt is predicted by a probe that never saw its fact. Adam, 500 full-batch steps, learning rate 0.01, weight decay 0.01. Written by `python -m crag.probes`.

m-only baseline: accuracy 0.80, AUC 0.88, within-m AUC 0.50 (by construction).

Best layer by within-m AUC: layer 3 (accuracy 0.54, AUC 0.56, within-m AUC 0.63).

m = 0 control (parametric vs extractive, 54 clean prompts, chance about 0.56): best accuracy 0.89 at layer 12.

Diagnostic, random prompt split (facts shared between training and test, so NOT a result): best AUC 0.88 at layer 31.

How far the layer-16 vector moves (mean distance; mean vector length 9.7): editing all 5 passages 0.6, another wording of the same question 2.7, another fact 4.1.

| layer | accuracy | AUC | within-m AUC | control accuracy | leaky AUC (diagnostic) |
|---|---|---|---|---|---|
| 1 | 0.50 | 0.51 | 0.51 | 0.72 | 0.66 |
| 2 | 0.52 | 0.55 | 0.58 | 0.74 | 0.67 |
| 3 | 0.54 | 0.56 | 0.63 | 0.74 | 0.70 |
| 4 | 0.56 | 0.51 | 0.54 | 0.63 | 0.70 |
| 5 | 0.56 | 0.54 | 0.58 | 0.69 | 0.65 |
| 6 | 0.49 | 0.48 | 0.47 | 0.70 | 0.67 |
| 7 | 0.50 | 0.47 | 0.43 | 0.70 | 0.67 |
| 8 | 0.51 | 0.46 | 0.39 | 0.70 | 0.67 |
| 9 | 0.49 | 0.47 | 0.45 | 0.76 | 0.65 |
| 10 | 0.49 | 0.45 | 0.43 | 0.74 | 0.67 |
| 11 | 0.50 | 0.51 | 0.54 | 0.87 | 0.71 |
| 12 | 0.52 | 0.52 | 0.57 | 0.89 | 0.72 |
| 13 | 0.53 | 0.53 | 0.53 | 0.87 | 0.71 |
| 14 | 0.54 | 0.56 | 0.61 | 0.85 | 0.71 |
| 15 | 0.53 | 0.51 | 0.49 | 0.83 | 0.71 |
| 16 | 0.55 | 0.56 | 0.59 | 0.83 | 0.69 |
| 17 | 0.50 | 0.49 | 0.47 | 0.81 | 0.70 |
| 18 | 0.52 | 0.49 | 0.50 | 0.81 | 0.79 |
| 19 | 0.55 | 0.50 | 0.51 | 0.85 | 0.79 |
| 20 | 0.54 | 0.53 | 0.56 | 0.85 | 0.79 |
| 21 | 0.57 | 0.55 | 0.57 | 0.85 | 0.80 |
| 22 | 0.56 | 0.54 | 0.56 | 0.83 | 0.81 |
| 23 | 0.53 | 0.59 | 0.62 | 0.81 | 0.84 |
| 24 | 0.57 | 0.59 | 0.62 | 0.80 | 0.86 |
| 25 | 0.55 | 0.56 | 0.58 | 0.78 | 0.86 |
| 26 | 0.54 | 0.58 | 0.59 | 0.81 | 0.87 |
| 27 | 0.51 | 0.57 | 0.56 | 0.81 | 0.87 |
| 28 | 0.51 | 0.58 | 0.58 | 0.80 | 0.88 |
| 29 | 0.50 | 0.54 | 0.54 | 0.80 | 0.88 |
| 30 | 0.48 | 0.53 | 0.51 | 0.80 | 0.88 |
| 31 | 0.50 | 0.56 | 0.52 | 0.80 | 0.88 |
| 32 | 0.53 | 0.55 | 0.52 | 0.74 | 0.88 |
