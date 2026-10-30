"""Probe pilot: can a linear probe on the hidden states predict whether the answer stays true?

    python -m crag.probes colab-l4-run1

For each of the 32 layers we train a separate logistic regression (4096 -> 1, BCE loss,
Adam) on the final prompt token's hidden state. The target is 1 if the greedy answer is
labelled "true" and 0 otherwise ("both" counts as 0 until the hand labels exist).

The split is by fact, so all 18 prompts of a fact are on the same side. With only 20
facts, one 80/20 split leaves 4 test facts, so we repeat it: the facts are cut into 5
folds and each fold is the test set once. Every prompt then gets exactly one prediction
from a probe that never saw its fact.

Baselines, as in the proposal:
    m only        a logistic regression whose only input is m
    m = 0 control a probe on the clean prompts that predicts parametric vs extractive

Diagnostic (not a result): the same probes with prompts split at random, so a test
prompt's fact is also in training. This breaks the rule that a fact stays on one side,
and only shows whether the label is in the hidden states at all.

Output, in results/<run>/analysis/:
    probes.md             tables
    probe_layers.csv      one line per layer
    fig_probe_layers.png  accuracy and AUC against layer
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from crag.analyze import groups, load, md_table, mean, rag_rows, write_csv
from crag.config import K, PROBE_FOLDS, PROBE_LR, PROBE_STEPS, PROBE_WEIGHT_DECAY, RESULTS, SEED
from crag.figures import AQUA, BLUE, DPI, MUTED, ORANGE, line
from crag.schema import Row


def load_hidden(run: str, rows: list[Row]) -> np.ndarray:
    """Load the saved hidden states of the RAG rows, in the same order.

    Args:
        run: The run's folder name in results/.
        rows: RAG rows (Conditions 2 and 3).

    Returns:
        Array of shape (prompts, 32, 4096), as float32.
    """
    arrays = [] # One (32, 4096) array per prompt
    for row in rows: # Same order as the rows
        name = f"{row['fact']}_c{row['condition']}_w{row['wording']}_m{row['m']}.npy" # File name used by run_baseline.rag()
        arrays.append(np.load(RESULTS / run / "hidden" / name).astype(np.float32)) # Saved as float16, train in float32
    return np.stack(arrays) # Stack into one array


def prompt_folds(n: int) -> list[int]:
    """Assign each prompt to a random fold, ignoring its fact (the leaky diagnostic only).

    Args:
        n: Number of prompts.

    Returns:
        The fold of each prompt.
    """
    rng = random.Random(SEED) # Same folds on every rerun
    return [rng.randrange(PROBE_FOLDS) for _ in range(n)] # Independent draw per prompt


def fact_folds(facts: list[str]) -> dict[str, int]:
    """Assign each fact to one of PROBE_FOLDS folds at random (fixed seed).

    Args:
        facts: Fact ids.

    Returns:
        fact -> fold number (0 to PROBE_FOLDS - 1).
    """
    order = sorted(set(facts)) # Fixed starting order
    random.Random(SEED).shuffle(order) # Same folds on every rerun
    return {fact: i % PROBE_FOLDS for i, fact in enumerate(order)} # Deal the facts out like cards


def train_probe(x: torch.Tensor, y: torch.Tensor) -> torch.nn.Linear:
    """Train one logistic-regression probe with BCE loss and Adam.

    Args:
        x: Inputs, shape (examples, features), already standardized.
        y: Targets, 0 or 1, shape (examples,).

    Returns:
        The trained probe (a single linear layer; sigmoid of its output is p).
    """
    torch.manual_seed(SEED) # Same starting weights every time
    probe = torch.nn.Linear(x.shape[1], 1) # w . h + b
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY) # Adam, with an L2 penalty
    loss_fn = torch.nn.BCEWithLogitsLoss() # Binary cross-entropy (applies the sigmoid itself)
    for _ in range(PROBE_STEPS): # Full-batch training: every example in every step
        optimizer.zero_grad() # Clear old gradients
        loss = loss_fn(probe(x).squeeze(1), y) # Loss on all training examples
        loss.backward() # Gradients
        optimizer.step() # Update the weights
    return probe # Return the probe


def out_of_fold(x: np.ndarray, y: np.ndarray, folds: list[int]) -> np.ndarray:
    """Predict every example with a probe trained on the other folds.

    Each feature is standardized with the training folds' mean and spread, because
    hidden-state values differ a lot in scale between layers.

    Args:
        x: Inputs, shape (examples, features).
        y: Targets, 0 or 1.
        folds: The fold of each example (from its fact).

    Returns:
        p for every example, the predicted probability that the answer is true.
    """
    probs = np.zeros(len(y)) # Filled in fold by fold
    fold_ids = np.array(folds) # For boolean masks
    for fold in range(PROBE_FOLDS): # Each fold is the test set once
        test = fold_ids == fold # Examples of the test facts
        train = ~test # Everything else
        if not test.any() or len(set(y[train])) < 2: # Nothing to test, or training data has one class only
            probs[test] = mean([float(v) for v in y[train]]) # Fall back to the training base rate
            continue
        centre = x[train].mean(axis=0) # Training mean of each feature
        spread = x[train].std(axis=0) + 1e-6 # Training spread (+ tiny value so a constant feature does not divide by 0)
        x_train = torch.tensor((x[train] - centre) / spread, dtype=torch.float32) # Standardized training inputs
        x_test = torch.tensor((x[test] - centre) / spread, dtype=torch.float32) # Same scaling for the test inputs
        probe = train_probe(x_train, torch.tensor(y[train], dtype=torch.float32)) # Train on the other folds
        with torch.no_grad(): # No gradients needed to predict
            probs[test] = torch.sigmoid(probe(x_test).squeeze(1)).numpy() # p for the test facts
    return probs # Return the predictions


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve: the chance that a random positive outranks a random negative.

    Ties count as half. 0.5 is chance, 1.0 is perfect ranking.

    Args:
        scores: Predicted probabilities.
        labels: True labels, 0 or 1.

    Returns:
        The AUC, or NaN if only one class is present.
    """
    positives = scores[labels == 1] # Scores of true answers
    negatives = scores[labels == 0] # Scores of the rest
    if len(positives) == 0 or len(negatives) == 0: # AUC needs both classes
        return float("nan")
    wins = 0.0 # Positive-negative pairs ranked correctly
    for p in positives: # Compare every pair
        wins += float((p > negatives).sum()) + 0.5 * float((p == negatives).sum()) # Ties count half
    return wins / (len(positives) * len(negatives)) # Share of pairs ranked correctly


def within_m_auc(scores: np.ndarray, labels: np.ndarray, ms: np.ndarray) -> float:
    """Average AUC computed separately at each m that has both labels (m = 1 to 4 here).

    An input of m alone gives the same score to every prompt with the same m, so its
    within-m AUC is exactly 0.5. A probe above 0.5 here separates prompts with the same
    number of edited passages.

    Args:
        scores: Predicted probabilities.
        labels: True labels, 0 or 1.
        ms: m of each prompt.

    Returns:
        The mean of the per-m AUCs, each weighted by its number of prompts.
    """
    total = 0.0 # Weighted sum of AUCs
    weight = 0 # Prompts counted
    for m in range(K + 1): # m = 0 to 5
        here = ms == m # Prompts at this m
        value = auc(scores[here], labels[here]) # AUC within this m
        if not np.isnan(value): # m = 0 and m = 5 have one label only
            total += value * int(here.sum())
            weight += int(here.sum())
    return total / weight if weight else float("nan") # Weighted mean


def plot_layers(table: list[list[float]], m_auc: float, control: list[float], leaky: list[float], path: str) -> None:
    """Two panels: probe AUC against layer (with baselines), and the m = 0 control accuracy.

    Args:
        table: One [layer, accuracy, AUC, within-m AUC] per layer.
        m_auc: AUC of the m-only baseline.
        control: Accuracy of the m = 0 control probe, one per layer.
        leaky: AUC with the random prompt split (diagnostic), one per layer.
        path: Output PNG.
    """
    layers = [int(t[0]) for t in table] # x values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2)) # Side by side
    ax = axes[0] # Left panel: the main probe
    line(ax, layers, [t[2] for t in table], BLUE, "o", "probe AUC, all prompts") # Overall AUC
    line(ax, layers, [t[3] for t in table], ORANGE, "s", "probe AUC within the same m") # Within-m AUC
    ax.plot(layers, leaky, color=MUTED, linewidth=1.5, linestyle="-.", label="diagnostic: facts shared with training") # Leaky split, drawn quieter
    ax.axhline(m_auc, color=MUTED, linestyle="--", linewidth=1, label=f"m-only baseline ({m_auc:.2f})") # m-only baseline, overall
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1, label="chance, and m only within the same m (0.50)") # Chance
    ax.set_title("Predicting a true greedy answer", loc="left") # Title
    ax.set_xlabel("layer") # x axis
    ax.set_ylabel("AUC on held-out facts") # y axis
    ax.set_ylim(0.3, 1.02) # Room for the baselines
    ax = axes[1] # Right panel: the control
    line(ax, layers, control, AQUA, "^", "m = 0 control") # Control accuracy
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1) # Chance for two balanced groups
    ax.set_title("Control: parametric vs extractive at m = 0", loc="left") # Title
    ax.set_xlabel("layer") # x axis
    ax.set_ylabel("accuracy on held-out facts") # y axis
    ax.set_ylim(0.3, 1.02) # Same scale as the left panel
    handles, texts = axes[0].get_legend_handles_labels() # Left panel's series and baselines
    fig.legend(handles, texts, loc="lower center", ncol=3, fontsize=8) # One legend under both panels
    fig.tight_layout(rect=(0, 0.14, 1, 1)) # Leave room for the legend
    fig.savefig(path, dpi=DPI) # Save
    plt.close(fig) # Free memory


def main() -> None:
    """Train the probes for every layer and write the tables and figure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="folder name in results/, e.g. colab-l4-run1")
    args = parser.parse_args()

    rows = load(args.run) # Every prompt of the run
    group_of = groups(rows) # Groups from Condition 1
    rag = rag_rows(rows) # The 360 RAG prompts
    hidden = load_hidden(args.run, rag) # (360, 32, 4096)
    y = np.array([1.0 if row["label"] == "true" else 0.0 for row in rag]) # Target: greedy answer is true
    ms = np.array([row["m"] for row in rag], dtype=float) # m of each prompt
    fold_of = fact_folds([row["fact"] for row in rag]) # fact -> fold
    folds = [fold_of[row["fact"]] for row in rag] # fold of each prompt

    base = out_of_fold(ms.reshape(-1, 1), y, folds) # m-only baseline
    base_acc = float(((base > 0.5) == (y == 1)).mean()) # Its accuracy
    base_auc = auc(base, y) # Its AUC

    table: list[list[float]] = [] # [layer, accuracy, AUC, within-m AUC]
    leaky: list[float] = [] # AUC with the random prompt split (diagnostic)
    mixed_folds = prompt_folds(len(rag)) # Folds that ignore facts
    for layer in range(hidden.shape[1]): # One probe per layer
        probs = out_of_fold(hidden[:, layer, :], y, folds) # Held-out predictions
        accuracy = float(((probs > 0.5) == (y == 1)).mean()) # Share predicted correctly
        table.append([layer + 1, accuracy, auc(probs, y), within_m_auc(probs, y, ms)]) # Layers numbered from 1
        leaky.append(auc(out_of_fold(hidden[:, layer, :], y, mixed_folds), y)) # Same probe, facts shared between train and test
        print(f"layer {layer + 1:2d}  acc {accuracy:.2f}  AUC {table[-1][2]:.2f}  "
              f"within-m AUC {table[-1][3]:.2f}  leaky AUC {leaky[-1]:.2f}") # Progress

    index = {(row["fact"], row["wording"], row["m"]): i for i, row in enumerate(rag)} # Find a prompt by its identifiers
    middle = hidden.shape[1] // 2 # Layer 16 of 32, for the size comparison
    by_edit, by_wording = [], [] # Distances between two prompts' vectors
    for (fact, wording, m), i in index.items(): # Sweep over prompts
        if m == 0: # Compare from each clean prompt
            by_edit.append(float(np.linalg.norm(hidden[i, middle] - hidden[index[(fact, wording, K)], middle]))) # Same fact and wording, all 5 passages edited
            if wording > 0: # Same fact, another wording
                by_wording.append(float(np.linalg.norm(hidden[i, middle] - hidden[index[(fact, 0, 0)], middle])))
    facts = sorted({row["fact"] for row in rag}) # Fact ids
    by_fact = [float(np.linalg.norm(hidden[index[(a, 0, 0)], middle] - hidden[index[(b, 0, 0)], middle]))
               for a, b in zip(facts, facts[1:])] # Neighbouring facts, first wording, clean

    clean = [i for i, row in enumerate(rag) if row["m"] == 0 and group_of[row["fact"]] != "mixed"] # Control: clean prompts, two groups only
    y_control = np.array([1.0 if group_of[rag[i]["fact"]] == "parametric" else 0.0 for i in clean]) # 1 = parametric
    control = [] # Accuracy per layer
    for layer in range(hidden.shape[1]): # One control probe per layer
        probs = out_of_fold(hidden[clean, layer, :], y_control, [folds[i] for i in clean]) # Held-out predictions
        control.append(float(((probs > 0.5) == (y_control == 1)).mean())) # Accuracy

    out_dir = RESULTS / args.run / "analysis" # Output folder
    out_dir.mkdir(exist_ok=True) # Create it if needed
    write_csv(out_dir / "probe_layers.csv", ["layer", "accuracy", "auc", "within_m_auc", "control_accuracy", "leaky_auc"],
              [t + [c, d] for t, c, d in zip(table, control, leaky)]) # One line per layer
    best = max(table, key=lambda t: t[3]) # Layer with the highest within-m AUC
    out = [f"# Probe pilot: {args.run}", "",
           f"{len(rag)} RAG prompts from {len(fold_of)} facts; target = greedy answer labelled true "
           f"({int(y.sum())} of {len(y)}). {PROBE_FOLDS} folds split by fact; every prompt is predicted by a probe "
           f"that never saw its fact. Adam, {PROBE_STEPS} full-batch steps, learning rate {PROBE_LR}, "
           f"weight decay {PROBE_WEIGHT_DECAY}. Written by `python -m crag.probes`.", "",
           f"m-only baseline: accuracy {base_acc:.2f}, AUC {base_auc:.2f}, within-m AUC 0.50 (by construction).", "",
           f"Best layer by within-m AUC: layer {int(best[0])} (accuracy {best[1]:.2f}, AUC {best[2]:.2f}, within-m AUC {best[3]:.2f}).", "",
           f"m = 0 control (parametric vs extractive, {len(clean)} clean prompts, chance about "
           f"{max(y_control.mean(), 1 - y_control.mean()):.2f}): best accuracy {max(control):.2f} "
           f"at layer {control.index(max(control)) + 1}.", "",
           f"Diagnostic, random prompt split (facts shared between training and test, so NOT a result): "
           f"best AUC {max(leaky):.2f} at layer {leaky.index(max(leaky)) + 1}.", "",
           f"How far the layer-{middle} vector moves (mean distance; mean vector length "
           f"{float(np.linalg.norm(hidden[:, middle], axis=1).mean()):.1f}): editing all 5 passages {mean(by_edit):.1f}, "
           f"another wording of the same question {mean(by_wording):.1f}, another fact {mean(by_fact):.1f}."] # Header and headline numbers
    per_layer = [[int(t[0]), t[1], t[2], t[3], c, d] for t, c, d in zip(table, control, leaky)] # Layer number as a whole number
    out += md_table(["layer", "accuracy", "AUC", "within-m AUC", "control accuracy", "leaky AUC (diagnostic)"], per_layer) # Per-layer table
    (out_dir / "probes.md").write_text("\n".join(out) + "\n", encoding="utf-8") # Save the tables
    plot_layers(table, base_auc, control, leaky, str(out_dir / "fig_probe_layers.png")) # Save the figure
    print(f"wrote {out_dir}") # Say where


if __name__ == "__main__":
    main()
