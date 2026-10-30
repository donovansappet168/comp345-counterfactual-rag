"""Draw the report figures for one run (the tables come from analyze.py).

    python -m crag.figures colab-l4-run1

Output, in results/<run>/analysis/:
    fig_dose_by_group.png     true / counterfactual / both rates against m, parametric vs extractive
    fig_greedy_labels.png     greedy label counts at each m (all facts)
    fig_passage1.png          true-answer rate against m, split by whether Passage 1 is edited
    fig_step_drops.png        drop in the true-answer rate when a step edits Passage 1, 2, ... 5
    fig_fact_curves.png       every fact's true-answer rate per wording, with the pooled sigmoid fit
"""

import argparse

import matplotlib

matplotlib.use("Agg") # Draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from crag.analyze import (dose_table, fit_fact, groups, load, mean, rag_rows, sigmoid, step_drop_ci,
                          step_drops, width)
from crag.config import K, RESULTS
from crag.schema import Row

# Chart style, the same as the demo notebook: blue = true value, orange = counterfactual,
# aqua = both. Each line also has its own marker, so the colours are never the only cue.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RATES = [("true_rate", "states the true value", BLUE, "o"),
         ("cf_rate", "states the counterfactual", ORANGE, "s"),
         ("both_rate", "states both", AQUA, "^")] # (row key, legend text, colour, marker)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "legend.frameon": False,
})
DPI = 200 # Resolution of the saved PNGs


def line(ax: Axes, x: list[int], y: list[float], color: str, marker: str, label: str) -> None:
    """Draw one series: a 2px line with markers ringed in the background colour.

    Args:
        ax: Where to draw.
        x: x values.
        y: y values.
        color: Line colour.
        marker: Marker shape, e.g. "o".
        label: Legend text.
    """
    ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label) # Ring keeps overlapping markers apart


def dose_axes(ax: Axes, title: str) -> None:
    """Shared look for a rate-against-m panel.

    Args:
        ax: The panel.
        title: Panel title.
    """
    ax.set_title(title, loc="left") # Title at the top left
    ax.set_xlabel("m = edited passages (of 5)") # x axis
    ax.set_xticks(range(K + 1)) # One tick per m
    ax.set_ylim(-0.03, 1.03) # Rates from 0 to 1, with room for markers
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)) # 0.5 -> 50%


def plot_dose_by_group(rows: list[Row], group_of: dict[str, str], path: str) -> None:
    """Two panels (parametric, extractive) of the three sampled rates against m.

    Args:
        rows: All rows of a run.
        group_of: Groups from analyze.groups().
        path: Output PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True) # Side by side, same y axis
    for ax, name in zip(axes, ["parametric", "extractive"]): # One panel per group
        facts = [f for f in group_of if group_of[f] == name] # The group's facts
        table = dose_table(rag_rows(rows, facts)) # Its dose table
        ms = [int(t["m"]) for t in table] # x values
        for key, text, color, marker in RATES: # The three rates
            line(ax, ms, [t[key] for t in table], color, marker, text) # Draw the rate
        dose_axes(ax, f"{name.capitalize()} facts ({len(facts)})") # Titles and ticks
    axes[0].set_ylabel("share of sampled answers") # y label once
    fig.suptitle("Sampled answers by m, averaged over all facts in each group", x=0.02, ha="left") # Say what each point averages
    handles, texts = axes[0].get_legend_handles_labels() # Both panels use the same series
    fig.legend(handles, texts, loc="lower center", ncol=3) # One legend under both panels, clear of the lines
    fig.tight_layout(rect=(0, 0.08, 1, 1)) # Leave room for the legend
    fig.savefig(path, dpi=DPI) # Save
    plt.close(fig) # Free memory


def plot_greedy_labels(rows: list[Row], path: str) -> None:
    """Stacked bars of greedy label counts at each m, all facts together.

    Args:
        rows: All rows of a run.
        path: Output PNG.
    """
    table = dose_table(rag_rows(rows)) # All facts
    ms = [int(t["m"]) for t in table] # x values
    fig, ax = plt.subplots(figsize=(6, 3.6)) # One panel
    bottom = np.zeros(len(ms)) # Where the next segment starts
    for lab, text, color in [("true", "true", BLUE), ("both", "both", AQUA), ("cf", "counterfactual", ORANGE)]: # Bottom to top
        counts = np.array([t[lab] for t in table]) # Segment heights
        ax.bar(ms, counts, bottom=bottom, color=color, width=0.6, edgecolor=SURFACE, linewidth=2, label=text) # 2px gap between segments
        for x, n, b in zip(ms, counts, bottom): # Label segments big enough to hold a number
            if n >= 4:
                ax.text(x, b + n / 2, str(int(n)), ha="center", va="center", color="white", fontsize=9)
        bottom += counts # Stack the next label on top
    ax.set_title("Greedy answer labels by m (20 facts × 3 wordings)", loc="left") # Title
    ax.set_xlabel("m = edited passages (of 5)") # x axis
    ax.set_ylabel("prompts") # y axis
    ax.set_xticks(ms) # One tick per m
    ax.grid(axis="x", visible=False) # Only horizontal grid lines
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1)) # Legend outside the bars
    fig.tight_layout() # Fit everything
    fig.savefig(path, dpi=DPI) # Save
    plt.close(fig) # Free memory


def plot_passage1(rows: list[Row], path: str) -> None:
    """True-answer rate against m, split by whether Passage 1 (the most similar) is edited.

    Args:
        rows: All rows of a run.
        path: Output PNG.
    """
    rag = rag_rows(rows) # All RAG prompts
    fig, ax = plt.subplots(figsize=(6, 3.6)) # One panel
    for edited, text, color, marker in [(False, "Passage 1 unedited", BLUE, "o"),
                                        (True, "Passage 1 edited", ORANGE, "s")]: # The two cases
        ms, rates = [], [] # Points of this line
        for m in range(K + 1): # m = 0 to 5
            here = [row for row in rag if row["m"] == m and (0 in row.get("edited_slots", [])) == edited] # Matching prompts
            if here: # m = 0 has no edited case and m = 5 no unedited one
                ms.append(m)
                rates.append(mean([row["true_rate"] for row in here])) # type: ignore[misc]
        line(ax, ms, rates, color, marker, text) # Draw the line
    dose_axes(ax, "True-answer rate, by whether Passage 1 is edited") # Titles and ticks
    ax.set_ylabel("share of sampled answers\nstating the true value") # y axis
    ax.legend(loc="upper right") # Legend
    fig.tight_layout() # Fit everything
    fig.savefig(path, dpi=DPI) # Save
    plt.close(fig) # Free memory


def plot_step_drops(rows: list[Row], path: str) -> None:
    """Bars of the mean drop in true-answer rate when a step edits each passage, with 95% intervals.

    Args:
        rows: All rows of a run.
        path: Output PNG.
    """
    rag = rag_rows(rows) # All RAG prompts
    drops = step_drops(rag) # Mean drop per added passage
    intervals = step_drop_ci(rag) # Their 95% intervals
    below = [d - lo for d, (lo, hi) in zip(drops, intervals)] # Bar top down to the interval's low end
    above = [hi - d for d, (lo, hi) in zip(drops, intervals)] # Bar top up to the high end
    scores = [mean([row["retrieval_scores"][k] for row in rag if row["m"] == 0]) for k in range(K)] # Mean similarity of each slot
    slots = list(range(1, K + 1)) # x values
    fig, ax = plt.subplots(figsize=(6, 3.6)) # One panel
    ax.bar(slots, drops, color=BLUE, width=0.6) # One bar per passage
    ax.errorbar(slots, drops, yerr=[below, above], fmt="none", ecolor=INK, elinewidth=1, capsize=4) # 95% intervals
    for x, d in zip(slots, drops): # Value beside each bar top
        ax.text(x + 0.18, d + 0.01, f"{d:.2f}", color=INK, fontsize=9)
    ax.set_xticks(slots, [f"Passage {k}\nsim. {s:.2f}" for k, s in zip(slots, scores)]) # Slot and mean similarity
    ax.set_ylabel("drop in true-answer rate") # y axis
    ax.set_title("Drop when one more passage is edited, by which passage", loc="left") # Title
    ax.grid(axis="x", visible=False) # Only horizontal grid lines
    fig.tight_layout() # Fit everything
    fig.savefig(path, dpi=DPI) # Save
    plt.close(fig) # Free memory


def plot_fact_curves(rows: list[Row], group_of: dict[str, str], path: str) -> None:
    """Small multiples: each fact's true-answer rate, one thin line per wording, and its pooled sigmoid fit.

    A point is an orange square once Passage 1 is edited and a blue circle before, so the
    panels show whether a wording's big drop lines up with the edit to Passage 1.

    Args:
        rows: All rows of a run.
        group_of: Groups from analyze.groups().
        path: Output PNG.
    """
    facts = sorted(group_of, key=lambda f: (group_of[f] != "parametric", f)) # Parametric facts first
    cols = 5 # Panels per row
    n_rows = (len(facts) + cols - 1) // cols # Enough rows for every fact
    fig, axes = plt.subplots(n_rows, cols, figsize=(12, 2.3 * n_rows), sharex=True, sharey=True) # Grid of panels
    grid_m = np.linspace(0, K, 100) # Smooth x for the fitted curve
    for ax, fact in zip(axes.flat, facts): # One panel per fact
        fact_rows = rag_rows(rows, [fact]) # The fact's 18 prompts
        for wording in sorted({row["wording"] for row in fact_rows}): # One line per wording
            one = sorted([row for row in fact_rows if row["wording"] == wording], key=lambda row: row["m"] or 0) # Its 6 prompts, m = 0 first
            ax.plot([row["m"] for row in one], [row["true_rate"] for row in one], color=MUTED, linewidth=0.8, alpha=0.6) # Thin connecting line
            for row in one: # Mark each point by whether Passage 1 is edited yet
                hit = 0 in row.get("edited_slots", []) # Passage 1 edited?
                ax.plot(row["m"], row["true_rate"], linestyle="none", marker="s" if hit else "o", markersize=4,
                        color=ORANGE if hit else BLUE, markeredgecolor=SURFACE, markeredgewidth=0.6)
        s, m0 = fit_fact(fact_rows) # Pooled fit over the 3 wordings
        ax.plot(grid_m, sigmoid(grid_m, s, m0), color=INK, linewidth=1.5) # Draw it
        ax.set_title(f"{fact}\n{group_of[fact]}, s = {s:.1f}, width {width(s):.1f}", fontsize=8, loc="left") # Fact, group, slope, width
        ax.set_xticks(range(K + 1)) # One tick per m
    for ax in axes.flat[len(facts):]: # Hide unused panels
        ax.set_visible(False)
    fig.supxlabel("m = edited passages (of 5)", color=MUTED) # Shared x label
    fig.supylabel("true-answer rate", color=MUTED) # Shared y label
    fig.suptitle("Blue circle: Passage 1 unedited. Orange square: Passage 1 edited. Black: sigmoid fit to all 3 wordings.",
                 color=MUTED, fontsize=9) # Key for the markers
    fig.tight_layout() # Fit everything
    fig.savefig(path, dpi=DPI) # Save
    plt.close(fig) # Free memory


def main() -> None:
    """Draw every figure for one run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="folder name in results/, e.g. colab-l4-run1")
    args = parser.parse_args()

    rows = load(args.run) # Every prompt of the run
    group_of = groups(rows) # Groups from this run's Condition 1
    out_dir = RESULTS / args.run / "analysis" # Output folder
    out_dir.mkdir(exist_ok=True) # Create it if needed
    plot_dose_by_group(rows, group_of, str(out_dir / "fig_dose_by_group.png")) # Figure: dose curves
    plot_greedy_labels(rows, str(out_dir / "fig_greedy_labels.png")) # Figure: greedy labels
    plot_passage1(rows, str(out_dir / "fig_passage1.png")) # Figure: Passage 1 split
    plot_step_drops(rows, str(out_dir / "fig_step_drops.png")) # Figure: drop by added passage
    plot_fact_curves(rows, group_of, str(out_dir / "fig_fact_curves.png")) # Figure: per-fact fits
    print(f"wrote figures to {out_dir}") # Say where


if __name__ == "__main__":
    main()
