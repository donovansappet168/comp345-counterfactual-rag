"""Turn one run's baseline.jsonl into the tables for the report.

    python -m crag.analyze colab-l4-run1     # reads results/colab-l4-run1/baseline.jsonl
    python -m crag.figures colab-l4-run1     # the figures (figures.py uses the functions here)

Output, in results/<run>/analysis/:
    summary.md            every table in Markdown, ready to copy into the report
    fact_fits.csv         one line per fact: group, flip point, sigmoid s, m0, and width
    both_to_label.csv     greedy answers labelled "both", with an empty column for the hand label
    handcheck_sample.csv  a random 50 of the "true" and "cf" greedy labels, to check by hand

Terms used below:
    truth-support rate   share of GREEDY answers labelled "true" (one answer per prompt)
    true-answer rate     share of the 10 SAMPLED answers labelled "true", averaged over prompts
    flip point           smallest m where a rate drops below 50%
    width                passages it takes a fact's fitted curve to fall from 90% to 10%
"""

import argparse
import csv
import json
import math
import random
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from crag.config import BOOTSTRAP_ROUNDS, FLIP_THRESHOLD, HANDCHECK_SAMPLE, K, MAX_SLOPE, RESULTS, SEED
from crag.schema import Row

LABELS = ["true", "cf", "both", "neither"]                 # the four script labels (score.py)
GROUPS = ["parametric", "extractive", "mixed"]             # set by Condition 1, see groups()


def load(run: str) -> list[Row]:
    """Read every row of one run.

    Args:
        run: The run's folder name in results/, e.g. "colab-l4-run1".

    Returns:
        The rows of results/<run>/baseline.jsonl.
    """
    rows: list[Row] = [] # One row per prompt
    with open(RESULTS / run / "baseline.jsonl", encoding="utf-8") as f: # Open the run's output
        for line in f: # One JSON object per line
            rows.append(json.loads(line)) # Parse and keep
    return rows # Return the rows


def groups(rows: list[Row]) -> dict[str, str]:
    """Sort the facts into groups from their Condition 1 (closed-book) greedy answers.

    A fact is "parametric" if Llama answers correctly with no passages for every
    wording, "extractive" if it answers correctly for none, and "mixed" if in between.

    Args:
        rows: All rows of a run.

    Returns:
        The group of each fact, e.g. {"nafta_start": "parametric", ...}.
    """
    correct: dict[str, int] = {} # Correct closed-book answers per fact
    asked: dict[str, int] = {} # Closed-book prompts per fact (one per wording)
    for row in rows: # Sweep over rows
        if row["condition"] != 1: # Only closed-book rows decide the group
            continue
        fact = row["fact"] # The fact's id
        asked[fact] = asked.get(fact, 0) + 1 # Count the wording
        correct[fact] = correct.get(fact, 0) + (row["label"] == "true") # Count it again if correct
    result: dict[str, str] = {} # fact -> group
    for fact in asked: # Sweep over facts
        if correct[fact] == asked[fact]: # Correct for every wording
            result[fact] = "parametric"
        elif correct[fact] == 0: # Correct for no wording
            result[fact] = "extractive"
        else: # Correct for some wordings
            result[fact] = "mixed"
    return result # Return the groups


def mean(values: list[float]) -> float:
    """Average a list, or NaN for an empty list.

    Args:
        values: The numbers.

    Returns:
        Their mean, or NaN if there are none.
    """
    if not values: # Nothing to average
        return math.nan
    return sum(values) / len(values) # Plain average


def rag_rows(rows: list[Row], facts: list[str] | None = None) -> list[Row]:
    """Keep the RAG rows (Conditions 2 and 3), optionally for some facts only.

    Args:
        rows: All rows of a run.
        facts: Fact ids to keep, or None for every fact.

    Returns:
        The rows with passages (m = 0 to 5).
    """
    kept: list[Row] = [] # The RAG rows
    for row in rows: # Sweep over rows
        if row["condition"] == 1: # Closed-book rows have no m
            continue
        if facts is not None and row["fact"] not in facts: # Not one of the requested facts
            continue
        kept.append(row) # Keep it
    return kept # Return the RAG rows


def dose_table(rows: list[Row]) -> list[dict[str, float]]:
    """Summarize RAG rows at each m: greedy label counts and the three sampled rates.

    Args:
        rows: RAG rows (from rag_rows()).

    Returns:
        One dict per m = 0 to 5, with keys "m", "prompts", one count per label,
        "truth_support", "true_rate", "cf_rate", and "both_rate".
    """
    table: list[dict[str, float]] = [] # One line per m
    for m in range(K + 1): # m = 0 to 5
        at_m = [row for row in rows if row["m"] == m] # Prompts with m edited passages
        line: dict[str, float] = {"m": m, "prompts": len(at_m)} # Start the line
        for lab in LABELS: # Count each greedy label
            line[lab] = sum(1 for row in at_m if row["label"] == lab)
        line["truth_support"] = line["true"] / len(at_m) if at_m else math.nan # Share of greedy answers labelled true
        for rate in ["true_rate", "cf_rate", "both_rate"]: # The three sampled rates
            line[rate] = mean([row[rate] for row in at_m if row.get(rate) is not None]) # type: ignore[literal-required] # Average over prompts that have samples
        table.append(line) # Add the line
    return table # Return the table


def flip_point(table: list[dict[str, float]], key: str) -> int | None:
    """Find the smallest m where a rate drops below FLIP_THRESHOLD (50%).

    Args:
        table: A dose_table().
        key: Which rate to use, e.g. "truth_support" or "true_rate".

    Returns:
        The flip point, or None if the rate never drops below the threshold.
    """
    for line in table: # m in increasing order
        if line[key] < FLIP_THRESHOLD: # Dropped below 50%
            return int(line["m"])
    return None # Never flipped


def sigmoid(m: np.ndarray, s: float, m0: float) -> np.ndarray:
    """The proposal's curve r(m) = 1 / (1 + exp(s (m - m0))).

    Args:
        m: Number of edited passages (one value per point).
        s: Slope. Large s is a sudden threshold, small s a gradual decline.
        m0: Midpoint, the m where the curve crosses 50%.

    Returns:
        r(m) for each point.
    """
    return 1 / (1 + np.exp(s * (m - m0))) # Falls from 1 to 0 around m0


def fit_fact(rows: list[Row]) -> tuple[float, float]:
    """Fit the sigmoid to one fact's true-answer rates against m.

    Pass all of the fact's RAG rows (18 for 3 wordings, as the proposal fits), or the
    6 rows of one wording to fit that wording on its own.

    Args:
        rows: The fact's RAG rows.

    Returns:
        (s, m0). s is capped at MAX_SLOPE, because a fact that drops in a single step
        has no finite best s.
    """
    m = np.array([row["m"] for row in rows], dtype=float) # x: edited passages
    r = np.array([row["true_rate"] for row in rows], dtype=float) # y: true-answer rate
    (s, m0), _ = curve_fit(sigmoid, m, r, p0=[1.0, K / 2], bounds=([0.0, -K], [MAX_SLOPE, 2 * K]), maxfev=50000) # Least squares, starting from a gentle curve centred at m = 2.5
    return float(s), float(m0) # Return slope and midpoint


def width(s: float) -> float:
    """Convert a slope s into the passages the curve takes to fall from 90% to 10%.

    r = 0.9 at m0 - ln(9)/s and r = 0.1 at m0 + ln(9)/s, so the width is 2 ln(9) / s
    (about 4.4 / s). Width 1 means the fact flips within one added passage.

    Args:
        s: The fitted slope.

    Returns:
        The width in passages (infinite for s = 0).
    """
    return 2 * math.log(9) / s if s > 0 else math.inf # 2 ln 9 ~ 4.39


def steps(rows: list[Row]) -> list[tuple[str, int, int, int, float]]:
    """List every one-passage step: going from m to m + 1 for one fact and wording.

    The edited slots are nested (edit.py), so each step adds exactly one edited passage
    and changes nothing else. That makes the drop at a step the effect of that one passage.

    Args:
        rows: RAG rows (Conditions 2 and 3).

    Returns:
        One (fact, wording, m, added slot, drop) per step, where the slot is 0-based and
        drop = true-answer rate at m minus the rate at m + 1 (positive = the rate fell).
    """
    by_prompt: dict[tuple[str, int, int], Row] = {} # (fact, wording, m) -> row
    for row in rows: # Sweep over prompts
        if row["m"] is not None: # Closed-book rows have no m
            by_prompt[(row["fact"], row["wording"], row["m"])] = row # Index the row
    result = [] # The steps
    for (fact, wording, m), before in sorted(by_prompt.items()): # Every prompt, in order
        after = by_prompt.get((fact, wording, m + 1)) # The same prompt with one more edit
        if after is None: # m = 5 has no next step
            continue
        old = before.get("edited_slots", []) # Slots edited at m
        added = [slot for slot in after.get("edited_slots", []) if slot not in old][0] # The one new slot
        drop = (before["true_rate"] or 0.0) - (after["true_rate"] or 0.0) # Fall in the true-answer rate
        result.append((fact, wording, m, added, drop)) # Keep the step
    return result # Return the steps


def step_drops(rows: list[Row]) -> list[float]:
    """Average drop in the true-answer rate at a step, by which passage the step adds.

    Args:
        rows: RAG rows (Conditions 2 and 3).

    Returns:
        One mean drop per slot, Passage 1 first.
    """
    drops: list[list[float]] = [[] for _ in range(K)] # Drops sorted by the added slot
    for _, _, _, added, drop in steps(rows): # Sweep over steps
        drops[added].append(drop) # File the drop under its slot
    return [mean(d) for d in drops] # Average per slot


def step_drop_ci(rows: list[Row]) -> list[tuple[float, float]]:
    """95% intervals for step_drops(), by resampling facts (bootstrap).

    Each round draws as many facts as there are, with replacement, and recomputes the
    averages. Facts are resampled rather than prompts because prompts of the same fact
    are not independent.

    Args:
        rows: RAG rows (Conditions 2 and 3).

    Returns:
        (low, high) for each slot, Passage 1 first.
    """
    rng = random.Random(SEED) # Same intervals on every rerun
    by_fact: dict[str, list[Row]] = {} # Rows grouped by fact
    for row in rows: # Sweep over rows
        by_fact.setdefault(row["fact"], []).append(row) # Add the row to its fact
    facts = sorted(by_fact) # Fact ids in a fixed order
    draws = [] # One set of averages per round
    for _ in range(BOOTSTRAP_ROUNDS): # Resample many times
        picked = [rng.choice(facts) for _ in facts] # Facts drawn with replacement
        drops: list[list[float]] = [[] for _ in range(K)] # Drops by added slot for this round
        for fact in picked: # A fact drawn twice counts twice
            for _, _, _, added, drop in steps(by_fact[fact]): # Its steps
                drops[added].append(drop)
        draws.append([mean(d) for d in drops]) # This round's averages
    low = np.nanpercentile(draws, 2.5, axis=0) # Lower edge of the 95% interval
    high = np.nanpercentile(draws, 97.5, axis=0) # Upper edge
    return [(float(a), float(b)) for a, b in zip(low, high)] # One (low, high) per slot


def first_cited(answer: str) -> int | None:
    """Find the first passage an answer names, e.g. 1 for "According to Passage 1, ...".

    Llama sometimes names the wrong passage, so this is a rough signal only.

    Args:
        answer: The greedy answer.

    Returns:
        The passage number (1 to 5), or None if the answer names no passage.
    """
    match = re.search(r"Passage (\d)", answer) # First "Passage N"
    return int(match.group(1)) if match else None # Its number


def md_table(header: list[str], lines: Sequence[Sequence[object]]) -> list[str]:
    """Format a Markdown table. Floats are printed with 2 decimals.

    Args:
        header: Column names.
        lines: One list of cells per row.

    Returns:
        The table as Markdown lines, with a blank line before it.
    """
    out = ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)] # Blank line, header, divider
    for cells in lines: # One table row per line
        text = [f"{c:.2f}" if isinstance(c, float) else str(c) for c in cells] # Round floats
        out.append("| " + " | ".join(text) + " |") # Add the row
    return out # Return the lines


def write_csv(path: Path, header: list[str], lines: Sequence[Sequence[object]]) -> None:
    """Write a CSV file with a header line.

    Args:
        path: Where to write.
        header: Column names.
        lines: One list of cells per row.
    """
    with open(path, "w", newline="", encoding="utf-8") as f: # newline="" stops blank lines on Windows
        writer = csv.writer(f) # Standard CSV quoting
        writer.writerow(header) # Column names
        writer.writerows(lines) # The data


def condition1_section(rows: list[Row], group_of: dict[str, str], planned: dict[str, str]) -> list[str]:
    """Markdown for Condition 1: closed-book answers per fact and the resulting groups.

    Args:
        rows: All rows of a run.
        group_of: Groups from groups().
        planned: The group each fact was picked for, from run_info.json.

    Returns:
        Markdown lines.
    """
    out = ["## Condition 1: closed-book", "",
           "Greedy correct = wordings (of 3) answered correctly with no passages. "
           "Rates are over the 10 samples per wording, averaged over wordings."] # Section intro
    lines = [] # One table row per fact
    for fact in sorted(group_of, key=lambda f: (GROUPS.index(group_of[f]), f)): # Parametric first, then by name
        c1 = [row for row in rows if row["condition"] == 1 and row["fact"] == fact] # The fact's closed-book rows
        correct = sum(1 for row in c1 if row["label"] == "true") # Greedy answers that are correct
        true_rate = mean([row["true_rate"] for row in c1 if row["true_rate"] is not None]) # Sampled true rate
        cf_rate = mean([row.get("cf_rate") or 0.0 for row in c1]) # Sampled counterfactual rate
        lines.append([fact, planned.get(fact, "?"), f"{correct}/{len(c1)}", true_rate, cf_rate, group_of[fact]]) # The row
    out += md_table(["fact", "picked as", "greedy correct", "true rate", "cf rate", "group (this run)"], lines) # The table
    counts = [f"{sum(1 for g in group_of.values() if g == name)} {name}" for name in GROUPS] # e.g. "10 parametric"
    out += ["", "Groups: " + ", ".join(counts) + "."] # Summary line
    return out # Return the section


def condition2_section(rows: list[Row], group_of: dict[str, str]) -> list[str]:
    """Markdown for Condition 2: clean passages (m = 0), per group, plus any misses.

    Args:
        rows: All rows of a run.
        group_of: Groups from groups().

    Returns:
        Markdown lines.
    """
    out = ["## Condition 2: clean passages (m = 0)"] # Section heading
    lines = [] # One table row per group
    for name in GROUPS + ["all"]: # Each group, then everything
        facts = [f for f in group_of if name in ("all", group_of[f])] # Facts in this group
        c2 = [row for row in rag_rows(rows, facts) if row["m"] == 0] # Their clean-RAG rows
        if not c2: # Group is empty in this run
            continue
        table = dose_table(c2)[0] # The m = 0 line
        lines.append([name, len(facts), int(table["prompts"]), table["truth_support"], table["true_rate"], table["cf_rate"]]) # The row
    out += md_table(["group", "facts", "prompts", "truth-support", "true rate", "cf rate"], lines) # The table
    misses = [row for row in rag_rows(rows) if row["m"] == 0 and (row["label"] != "true" or (row["true_rate"] or 0) < 1)] # Anything short of perfect
    out += ["", "Prompts with a greedy label other than true, or a sampled true rate below 1:"] # Lead-in
    for row in misses: # List each one
        out.append(f"- {row['fact']} wording {row['wording']}: greedy {row['label']}, true rate {row['true_rate']:.1f}")
    return out # Return the section


def dose_section(rows: list[Row], group_of: dict[str, str]) -> list[str]:
    """Markdown for the dose curves (m = 0 to 5) and flip points, per group.

    Args:
        rows: All rows of a run.
        group_of: Groups from groups().

    Returns:
        Markdown lines.
    """
    out = ["## Condition 3: dose (m = 0 to 5)", "",
           "Greedy label counts, truth-support (greedy), and the three sampled rates. "
           "m = 0 is Condition 2, repeated so each curve starts from the clean prompt."] # Section intro
    for name in GROUPS + ["all"]: # Each group, then everything
        facts = [f for f in group_of if name in ("all", group_of[f])] # Facts in this group
        table = dose_table(rag_rows(rows, facts)) # Its dose table
        if not table[0]["prompts"]: # Group is empty in this run
            continue
        lines = [[int(t["m"]), int(t["prompts"])] + [int(t[lab]) for lab in LABELS]
                 + [t["truth_support"], t["true_rate"], t["cf_rate"], t["both_rate"]] for t in table] # One row per m
        out += ["", f"### {name} ({len(facts)} facts)"] # Sub-heading
        out += md_table(["m", "prompts"] + LABELS + ["truth-support", "true rate", "cf rate", "both rate"], lines) # The table
        out += ["", f"Flip point: m = {flip_point(table, 'truth_support')} by truth-support, "
                    f"m = {flip_point(table, 'true_rate')} by true-answer rate."] # Both flip points
    return out # Return the section


def fits_section(rows: list[Row], group_of: dict[str, str], out_dir: Path) -> list[str]:
    """Markdown and fact_fits.csv for the per-fact flip points and sigmoid fits.

    Args:
        rows: All rows of a run.
        group_of: Groups from groups().
        out_dir: Where fact_fits.csv goes.

    Returns:
        Markdown lines.
    """
    out = ["## Per-fact sigmoid fits", "",
           "r(m) = 1 / (1 + exp(s (m - m0))) fitted to the true-answer rate of all 18 RAG prompts of a fact. "
           "Width = 2 ln 9 / s = passages to fall from 90% to 10% (width 1 or less = flips within one passage). "
           "Flip point = smallest m where the fact's mean true-answer rate is below 50%."] # Section intro
    fits: list[tuple[str, str, int | None, float, float, float]] = [] # One (fact, group, flip, s, m0, width) per fact
    for fact in sorted(group_of, key=lambda f: (GROUPS.index(group_of[f]), f)): # Parametric first
        fact_rows = rag_rows(rows, [fact]) # The fact's 18 RAG rows
        s, m0 = fit_fact(fact_rows) # Fit the curve
        flip = flip_point(dose_table(fact_rows), "true_rate") # Per-fact flip point
        fits.append((fact, group_of[fact], flip, s, m0, width(s))) # The row
    header = ["fact", "group", "flip point", "s", "m0", "width"] # Column names
    out += md_table(header, fits) # The table
    write_csv(out_dir / "fact_fits.csv", header, fits) # Same table as CSV
    medians: list[list[object]] = [] # One row per group
    for name in GROUPS: # Each group
        mine = [fit for fit in fits if fit[1] == name] # The group's facts
        if not mine: # Group is empty in this run
            continue
        sharp = sum(1 for fit in mine if fit[5] <= 1) # Facts that flip within one passage
        medians.append([name, len(mine), float(np.median([fit[3] for fit in mine])),
                        float(np.median([fit[4] for fit in mine])), float(np.median([fit[5] for fit in mine])),
                        f"{sharp}/{len(mine)}"]) # Median s, m0, width, and the sharp count
    out += ["", "Medians per group:"] # Lead-in
    out += md_table(["group", "facts", "median s", "median m0", "median width", "width <= 1"], medians) # The medians

    # The fits above pool 3 wordings, and each wording edits Passage 1 at a different m.
    # Fitting each wording on its own shows whether a single wording drops gradually or in one step.
    all_steps = steps(rag_rows(rows)) # Every m -> m + 1 step
    by_wording: list[list[object]] = [] # One row per group
    for name in GROUPS + ["all"]: # Each group, then everything
        widths: list[float] = [] # One fitted width per (fact, wording)
        biggest: list[tuple[float, int]] = [] # (drop, added slot) of each (fact, wording)'s largest step
        for fact in group_of: # Sweep over facts
            if name not in ("all", group_of[fact]): # Not in this group
                continue
            for wording in sorted({row["wording"] for row in rag_rows(rows, [fact])}): # Each wording
                one = [row for row in rag_rows(rows, [fact]) if row["wording"] == wording] # Its 6 prompts
                widths.append(width(fit_fact(one)[0])) # Fit this wording alone
                its_steps = [(step[4], step[3]) for step in all_steps if step[0] == fact and step[1] == wording] # Its 5 steps
                biggest.append(max(its_steps)) # The largest drop and which passage it added
        if not widths: # Group is empty in this run
            continue
        by_wording.append([name, len(widths), float(np.median(widths)),
                      f"{sum(1 for w in widths if w <= 1)}/{len(widths)}",
                      f"{sum(1 for drop, _ in biggest if drop >= FLIP_THRESHOLD)}/{len(biggest)}",
                      f"{sum(1 for _, slot in biggest if slot == 0)}/{len(biggest)}"]) # Summary for the group
    out += ["", "Each wording fitted on its own (6 prompts each):"] # Lead-in
    out += md_table(["group", "fact-wordings", "median width", "width <= 1",
                     "has one step dropping >= 50 points", "largest step edits Passage 1"], by_wording) # The table
    return out # Return the section


def position_section(rows: list[Row]) -> list[str]:
    """Markdown for the position results: which edited slot moves the answer most.

    Args:
        rows: All rows of a run.

    Returns:
        Markdown lines.
    """
    rag = rag_rows(rows) # All RAG prompts
    out = ["## Position of the edited passages", "",
           "Slot k = Passage k in the prompt. The retriever puts the most similar passage in slot 1, "
           "so position and similarity go together."] # Section intro

    scores = [mean([row["retrieval_scores"][k] for row in rag if row["m"] == 0]) for k in range(K)] # Average score of each slot
    out += ["", "Mean retriever similarity by slot (clean prompts):"] # Lead-in
    out += md_table([f"slot {k + 1}" for k in range(K)], [scores]) # One row of scores

    m1 = [row for row in rag if row["m"] == 1] # Prompts with exactly one edited passage
    lines: list[list[object]] = [] # One row per edited slot
    for k in range(K): # Each slot
        here = [row for row in m1 if row.get("edited_slots") == [k]] # Prompts whose one edit is in slot k
        lines.append([k + 1, len(here), sum(1 for row in here if row["label"] == "true"),
                      mean([row["true_rate"] for row in here]), mean([row["both_rate"] for row in here])]) # type: ignore[misc]
    out += ["", "m = 1, by which slot holds the one edited passage:"] # Lead-in
    out += md_table(["edited slot", "prompts", "greedy true", "true rate", "both rate"], lines) # The table

    lines = [] # One row per (m, Passage 1 edited or not)
    for m in range(K + 1): # m = 0 to 5
        for edited in [False, True]: # Passage 1 clean, then edited
            here = [row for row in rag if row["m"] == m and (0 in row.get("edited_slots", [])) == edited] # Matching prompts
            if here: # Skip impossible combinations (m = 0 edited, m = 5 clean)
                lines.append([m, "edited" if edited else "clean", len(here), sum(1 for row in here if row["label"] == "true"),
                              mean([row["true_rate"] for row in here]), mean([row["cf_rate"] for row in here])]) # type: ignore[misc]
    out += ["", "By m and whether Passage 1 is edited:"] # Lead-in
    out += md_table(["m", "Passage 1", "prompts", "greedy true", "true rate", "cf rate"], lines) # The table

    all_steps = steps(rag) # Every m -> m + 1 step (one passage added each)
    drops = step_drops(rag) # Mean drop by added slot
    intervals = step_drop_ci(rag) # Their 95% intervals
    lines = [] # One row per slot
    for k in range(K): # Each slot
        count = sum(1 for step in all_steps if step[3] == k) # Steps that add this slot
        lines.append([f"Passage {k + 1}", count, drops[k], f"{intervals[k][0]:.2f} to {intervals[k][1]:.2f}"]) # The row
    out += ["", f"Drop in the true-answer rate at a step (m to m + 1), by which passage the step edits "
                f"(95% interval from {BOOTSTRAP_ROUNDS} fact resamples):"] # Lead-in
    out += md_table(["added passage", "steps", "mean drop", "95% interval"], lines) # The table

    cited: dict[str, int] = {} # First passage named in the greedy answer -> count
    for row in rag: # Sweep over prompts
        n = first_cited(row["greedy"]) # e.g. 1 for "According to Passage 1"
        key = f"Passage {n}" if n else "names none" # Table label
        cited[key] = cited.get(key, 0) + 1 # Count
    out += ["", "First passage named in the greedy answer (all RAG prompts; Llama sometimes names the wrong one):"] # Lead-in
    out += md_table(sorted(cited), [[cited[key] for key in sorted(cited)]]) # One row of counts
    return out # Return the section


def failure_section(rows: list[Row]) -> list[str]:
    """Markdown for the failure types (score.failure_type) of greedy answers by m.

    Args:
        rows: All rows of a run.

    Returns:
        Markdown lines.
    """
    rag = rag_rows(rows) # All RAG prompts
    kinds = sorted({row["failure"] for row in rows if row["failure"]}) # Failure types that occur
    lines: list[Sequence[object]] = [] # One row per m
    for m in range(K + 1): # m = 0 to 5
        at_m = [row for row in rag if row["m"] == m] # Prompts at this m
        lines.append([m] + [sum(1 for row in at_m if row["failure"] == kind) for kind in kinds]) # Count each type
    c1 = [row for row in rows if row["condition"] == 1] # Closed-book prompts
    lines.append(["closed-book"] + [sum(1 for row in c1 if row["failure"] == kind) for kind in kinds]) # Same for Condition 1
    out = ["## Failure types (greedy answers, before hand labelling)"] # Section heading
    out += md_table(["m"] + kinds, lines) # The table
    return out # Return the section


def export_handcheck(rows: list[Row], out_dir: Path) -> None:
    """Write the two hand-labelling files: every "both" answer, and a random 50 script labels.

    Args:
        rows: All rows of a run.
        out_dir: Where the CSV files go.
    """
    header = ["fact", "condition", "wording", "m", "edited_slots", "label", "greedy", "hand_label"] # hand_label is left empty
    both = [row for row in rows if row["label"] == "both"] # Answers the script cannot decide
    write_csv(out_dir / "both_to_label.csv", header,
              [[r["fact"], r["condition"], r["wording"], r["m"], r.get("edited_slots"), r["label"], r["greedy"], ""] for r in both]) # One line each
    decided = [row for row in rows if row["label"] in ("true", "cf")] # Labels the script decided
    sample = random.Random(SEED).sample(decided, min(HANDCHECK_SAMPLE, len(decided))) # Same 50 on every rerun
    write_csv(out_dir / "handcheck_sample.csv", header,
              [[r["fact"], r["condition"], r["wording"], r["m"], r.get("edited_slots"), r["label"], r["greedy"], ""] for r in sample]) # One line each


def main() -> None:
    """Read one run and write summary.md and the CSV files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="folder name in results/, e.g. colab-l4-run1")
    args = parser.parse_args()

    rows = load(args.run) # Every prompt of the run
    info = json.loads((RESULTS / args.run / "run_info.json").read_text(encoding="utf-8")) # Settings written by the Colab notebook
    planned = {f: g for g, facts in info["facts"].items() for f in facts} # Group each fact was picked for
    group_of = groups(rows) # Group from this run's Condition 1
    out_dir = RESULTS / args.run / "analysis" # Output folder
    out_dir.mkdir(exist_ok=True) # Create it if needed

    out = [f"# Results: {args.run}", "",
           f"{len(rows)} prompts. Commit {info['commit'][:7]}, {info['gpu']}, {info['generator']}, "
           f"chat date {info['chat_date']}, {info['max_new_tokens']} new tokens, {info['n_samples']} samples, seed {info['seed']}. "
           "Written by `python -m crag.analyze`; do not edit by hand.", ""] # Header
    out += condition1_section(rows, group_of, planned) + [""] # Condition 1
    out += condition2_section(rows, group_of) + [""] # Condition 2
    out += dose_section(rows, group_of) + [""] # Condition 3
    out += fits_section(rows, group_of, out_dir) + [""] # Sigmoid fits
    out += position_section(rows) + [""] # Slot position
    out += failure_section(rows) # Failure types
    (out_dir / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8") # Save the summary
    export_handcheck(rows, out_dir) # Save the hand-labelling files
    print(f"wrote {out_dir}") # Say where


if __name__ == "__main__":
    main()
