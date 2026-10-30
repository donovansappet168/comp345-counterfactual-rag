# Results: colab-l4-run1

420 prompts. Commit 3507479, NVIDIA L4, unsloth/Llama-3.1-8B-Instruct, chat date 26 Oct 2026, 192 new tokens, 10 samples, seed 345. Written by `python -m crag.analyze`; do not edit by hand.

## Condition 1: closed-book

Greedy correct = wordings (of 3) answered correctly with no passages. Rates are over the 10 samples per wording, averaged over wordings.

| fact | picked as | greedy correct | true rate | cf rate | group (this run) |
|---|---|---|---|---|---|
| harvey_year | parametric | 3/3 | 1.00 | 0.00 | parametric |
| hemp_thc_limit | parametric | 3/3 | 1.00 | 0.00 | parametric |
| mexico_energy_reform | parametric | 3/3 | 1.00 | 0.00 | parametric |
| nafta_start | parametric | 3/3 | 1.00 | 0.00 | parametric |
| russia_ukraine_invasion | parametric | 3/3 | 0.97 | 0.00 | parametric |
| tcja_corporate_rate | parametric | 3/3 | 1.00 | 0.00 | parametric |
| tcja_salt_cap | parametric | 3/3 | 1.00 | 0.00 | parametric |
| uri_month | parametric | 3/3 | 0.97 | 0.00 | parametric |
| usmca_start | parametric | 3/3 | 1.00 | 0.00 | parametric |
| water_treaty_year | parametric | 3/3 | 0.83 | 0.03 | parametric |
| baker_institute_director | extractive | 0/3 | 0.00 | 0.13 | extractive |
| carney_prime_minister | extractive | 0/3 | 0.00 | 0.00 | extractive |
| iran_war_start | extractive | 0/3 | 0.00 | 0.00 | extractive |
| metals_tariff_2025 | extractive | 0/3 | 0.03 | 0.00 | extractive |
| mexico_president_2024 | extractive | 0/3 | 0.03 | 0.00 | extractive |
| mexico_top_partner_year | extractive | 0/3 | 0.07 | 0.00 | extractive |
| rubio_secretary_of_state | extractive | 0/3 | 0.00 | 0.00 | extractive |
| sheinbaum_took_office | extractive | 0/3 | 0.00 | 0.03 | extractive |
| hamas_attack_date | extractive | 1/3 | 0.10 | 0.00 | mixed |
| mexico_canada_tariff_2025 | extractive | 1/3 | 0.23 | 0.00 | mixed |

Groups: 10 parametric, 8 extractive, 2 mixed.

## Condition 2: clean passages (m = 0)

| group | facts | prompts | truth-support | true rate | cf rate |
|---|---|---|---|---|---|
| parametric | 10 | 30 | 1.00 | 0.99 | 0.00 |
| extractive | 8 | 24 | 1.00 | 1.00 | 0.00 |
| mixed | 2 | 6 | 1.00 | 0.98 | 0.00 |
| all | 20 | 60 | 1.00 | 0.99 | 0.00 |

Prompts with a greedy label other than true, or a sampled true rate below 1:
- mexico_energy_reform wording 0: greedy true, true rate 0.9
- mexico_energy_reform wording 2: greedy true, true rate 0.9
- hamas_attack_date wording 1: greedy true, true rate 0.9

## Condition 3: dose (m = 0 to 5)

Greedy label counts, truth-support (greedy), and the three sampled rates. m = 0 is Condition 2, repeated so each curve starts from the clean prompt.

### parametric (10 facts)

| m | prompts | true | cf | both | neither | truth-support | true rate | cf rate | both rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 30 | 30 | 0 | 0 | 0 | 1.00 | 0.99 | 0.00 | 0.00 |
| 1 | 30 | 17 | 1 | 12 | 0 | 0.57 | 0.68 | 0.04 | 0.28 |
| 2 | 30 | 14 | 5 | 11 | 0 | 0.47 | 0.39 | 0.14 | 0.46 |
| 3 | 30 | 8 | 13 | 9 | 0 | 0.27 | 0.21 | 0.34 | 0.45 |
| 4 | 30 | 2 | 18 | 10 | 0 | 0.07 | 0.08 | 0.64 | 0.27 |
| 5 | 30 | 0 | 30 | 0 | 0 | 0.00 | 0.00 | 0.98 | 0.02 |

Flip point: m = 2 by truth-support, m = 2 by true-answer rate.

### extractive (8 facts)

| m | prompts | true | cf | both | neither | truth-support | true rate | cf rate | both rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 24 | 24 | 0 | 0 | 0 | 1.00 | 1.00 | 0.00 | 0.00 |
| 1 | 24 | 14 | 2 | 8 | 0 | 0.58 | 0.59 | 0.06 | 0.35 |
| 2 | 24 | 9 | 3 | 12 | 0 | 0.38 | 0.36 | 0.12 | 0.52 |
| 3 | 24 | 4 | 11 | 9 | 0 | 0.17 | 0.20 | 0.27 | 0.53 |
| 4 | 24 | 1 | 17 | 6 | 0 | 0.04 | 0.05 | 0.61 | 0.33 |
| 5 | 24 | 0 | 24 | 0 | 0 | 0.00 | 0.00 | 1.00 | 0.00 |

Flip point: m = 2 by truth-support, m = 2 by true-answer rate.

### mixed (2 facts)

| m | prompts | true | cf | both | neither | truth-support | true rate | cf rate | both rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 6 | 0 | 0 | 0 | 1.00 | 0.98 | 0.00 | 0.00 |
| 1 | 6 | 5 | 1 | 0 | 0 | 0.83 | 0.67 | 0.10 | 0.22 |
| 2 | 6 | 5 | 0 | 1 | 0 | 0.83 | 0.60 | 0.13 | 0.23 |
| 3 | 6 | 2 | 2 | 2 | 0 | 0.33 | 0.30 | 0.32 | 0.38 |
| 4 | 6 | 3 | 2 | 1 | 0 | 0.50 | 0.18 | 0.45 | 0.37 |
| 5 | 6 | 0 | 6 | 0 | 0 | 0.00 | 0.00 | 0.97 | 0.02 |

Flip point: m = 3 by truth-support, m = 3 by true-answer rate.

### all (20 facts)

| m | prompts | true | cf | both | neither | truth-support | true rate | cf rate | both rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 60 | 60 | 0 | 0 | 0 | 1.00 | 0.99 | 0.00 | 0.00 |
| 1 | 60 | 36 | 4 | 20 | 0 | 0.60 | 0.64 | 0.06 | 0.30 |
| 2 | 60 | 28 | 8 | 24 | 0 | 0.47 | 0.40 | 0.13 | 0.46 |
| 3 | 60 | 14 | 26 | 20 | 0 | 0.23 | 0.21 | 0.31 | 0.47 |
| 4 | 60 | 6 | 37 | 17 | 0 | 0.10 | 0.08 | 0.61 | 0.31 |
| 5 | 60 | 0 | 60 | 0 | 0 | 0.00 | 0.00 | 0.98 | 0.01 |

Flip point: m = 2 by truth-support, m = 2 by true-answer rate.

## Per-fact sigmoid fits

r(m) = 1 / (1 + exp(s (m - m0))) fitted to the true-answer rate of all 18 RAG prompts of a fact. Width = 2 ln 9 / s = passages to fall from 90% to 10% (width 1 or less = flips within one passage). Flip point = smallest m where the fact's mean true-answer rate is below 50%.

| fact | group | flip point | s | m0 | width |
|---|---|---|---|---|---|
| harvey_year | parametric | 2 | 0.94 | 1.66 | 4.66 |
| hemp_thc_limit | parametric | 3 | 1.29 | 2.54 | 3.41 |
| mexico_energy_reform | parametric | 3 | 1.10 | 2.44 | 4.01 |
| nafta_start | parametric | 2 | 2.52 | 1.10 | 1.74 |
| russia_ukraine_invasion | parametric | 3 | 0.83 | 2.33 | 5.28 |
| tcja_corporate_rate | parametric | 2 | 4.78 | 1.67 | 0.92 |
| tcja_salt_cap | parametric | 2 | 1.68 | 1.43 | 2.61 |
| uri_month | parametric | 1 | 2.24 | 1.04 | 1.96 |
| usmca_start | parametric | 2 | 1.90 | 1.77 | 2.31 |
| water_treaty_year | parametric | 3 | 1.14 | 1.97 | 3.85 |
| baker_institute_director | extractive | 2 | 2.82 | 1.11 | 1.56 |
| carney_prime_minister | extractive | 2 | 1.47 | 1.82 | 2.98 |
| iran_war_start | extractive | 4 | 1.44 | 3.40 | 3.04 |
| metals_tariff_2025 | extractive | 4 | 1.02 | 2.34 | 4.31 |
| mexico_president_2024 | extractive | 3 | 1.64 | 1.88 | 2.69 |
| mexico_top_partner_year | extractive | 1 | 4.97 | 0.80 | 0.88 |
| rubio_secretary_of_state | extractive | 1 | 8.32 | 0.78 | 0.53 |
| sheinbaum_took_office | extractive | 2 | 1.83 | 1.40 | 2.40 |
| hamas_attack_date | mixed | 3 | 1.43 | 2.57 | 3.07 |
| mexico_canada_tariff_2025 | mixed | 2 | 0.86 | 1.82 | 5.13 |

Medians per group:

| group | facts | median s | median m0 | median width | width <= 1 |
|---|---|---|---|---|---|
| parametric | 10 | 1.49 | 1.72 | 3.01 | 1/10 |
| extractive | 8 | 1.73 | 1.61 | 2.54 | 2/8 |
| mixed | 2 | 1.14 | 2.19 | 4.10 | 0/2 |

Each wording fitted on its own (6 prompts each):

| group | fact-wordings | median width | width <= 1 | has one step dropping >= 50 points | largest step edits Passage 1 |
|---|---|---|---|---|---|
| parametric | 30 | 1.57 | 12/30 | 20/30 | 15/30 |
| extractive | 24 | 0.73 | 16/24 | 21/24 | 11/24 |
| mixed | 6 | 3.29 | 0/6 | 4/6 | 4/6 |
| all | 60 | 1.38 | 28/60 | 45/60 | 30/60 |

## Position of the edited passages

Slot k = Passage k in the prompt. The retriever puts the most similar passage in slot 1, so position and similarity go together.

Mean retriever similarity by slot (clean prompts):

| slot 1 | slot 2 | slot 3 | slot 4 | slot 5 |
|---|---|---|---|---|
| 0.65 | 0.59 | 0.53 | 0.46 | 0.37 |

m = 1, by which slot holds the one edited passage:

| edited slot | prompts | greedy true | true rate | both rate |
|---|---|---|---|---|
| 1 | 16 | 5 | 0.36 | 0.56 |
| 2 | 17 | 10 | 0.74 | 0.22 |
| 3 | 10 | 7 | 0.59 | 0.31 |
| 4 | 8 | 6 | 0.80 | 0.14 |
| 5 | 9 | 8 | 0.88 | 0.12 |

By m and whether Passage 1 is edited:

| m | Passage 1 | prompts | greedy true | true rate | cf rate |
|---|---|---|---|---|---|
| 0 | clean | 60 | 60 | 0.99 | 0.00 |
| 1 | clean | 44 | 31 | 0.74 | 0.05 |
| 1 | edited | 16 | 5 | 0.36 | 0.08 |
| 2 | clean | 35 | 21 | 0.52 | 0.09 |
| 2 | edited | 25 | 7 | 0.23 | 0.20 |
| 3 | clean | 24 | 12 | 0.35 | 0.16 |
| 3 | edited | 36 | 2 | 0.12 | 0.41 |
| 4 | clean | 12 | 3 | 0.22 | 0.21 |
| 4 | edited | 48 | 3 | 0.05 | 0.71 |
| 5 | edited | 60 | 0 | 0.00 | 0.98 |

Drop in the true-answer rate at a step (m to m + 1), by which passage the step edits (95% interval from 2000 fact resamples):

| added passage | steps | mean drop | 95% interval |
|---|---|---|---|
| Passage 1 | 60 | 0.41 | 0.32 to 0.50 |
| Passage 2 | 60 | 0.19 | 0.13 to 0.26 |
| Passage 3 | 60 | 0.20 | 0.13 to 0.28 |
| Passage 4 | 60 | 0.10 | 0.06 to 0.15 |
| Passage 5 | 60 | 0.09 | 0.05 to 0.12 |

First passage named in the greedy answer (all RAG prompts; Llama sometimes names the wrong one):

| Passage 1 | Passage 2 | Passage 3 | Passage 4 | Passage 5 | names none |
|---|---|---|---|---|---|
| 167 | 29 | 25 | 6 | 1 | 132 |

## Failure types (greedy answers, before hand labelling)

| m | adopted counterfactual | mixed | other wrong answer | refusal |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 4 | 20 | 0 | 0 |
| 2 | 8 | 24 | 0 | 0 |
| 3 | 26 | 20 | 0 | 0 |
| 4 | 37 | 17 | 0 | 0 |
| 5 | 60 | 0 | 0 | 0 |
| closed-book | 1 | 0 | 11 | 16 |
