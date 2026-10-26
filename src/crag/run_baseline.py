"""Condition 1 (closed-book) and Condition 2 (clean RAG, m = 0).

    python -m crag.run_baseline                 # every fact in data/facts.json
    python -m crag.run_baseline --samples 0     # greedy answers only (much faster on CPU)
    python -m crag.run_baseline --facts nafta_start usmca_start

Output:
    results/baseline.jsonl   one line per (fact, condition, wording)
    results/hidden/*.npy     hidden states (32 x 4096) for each Condition 2 prompt

If the run stops, start it again: lines already in baseline.jsonl are skipped.
"""

import argparse
import json
import time
from typing import TYPE_CHECKING

import numpy as np

from crag import facts
from crag.config import N_SAMPLES, RESULTS
from crag.edit import build_context
from crag.passages import rank
from crag.prompt import closed_book_prompt, rag_prompt
from crag.schema import Answer, Fact, Passage, Row
from crag.score import failure_type, label

if TYPE_CHECKING: # Only for type hints; the real import is slow, so it happens in main()
    from crag.generate import Llama # Only for type hints; the real import is slow, so it happens in main()

OUTPUT = RESULTS / "baseline.jsonl"
HIDDEN_DIR = RESULTS / "hidden"


def top5(fact: Fact, wording: int) -> list[Passage]:
    """Order the fact's passages by retriever similarity to one wording of the question.

    Args:
        fact: The fact.
        wording: Which question wording (0, 1, or 2).

    Returns:
        The 5 most similar of the fact's passages, slot 1 first, each with a "score".
    """
    question = fact["questions"][wording] # The question in this wording
    return rank(question, fact["passages"])[:5] # Rank the fact's passages by similarity to the question, keep the top 5 (slot 1 = most similar)


def make_row(fact: Fact, condition: int, wording: int, m: int | None,
             prompt: str, result: Answer) -> Row:
    """Score Llama's answers for one prompt and package them as an output row.

    Args:
        fact: The fact.
        condition: 1, 2, or 3.
        wording: Which question wording (0, 1, or 2).
        m: Number of edited passages (None for closed-book).
        prompt: The exact prompt text.
        result: Llama's output for the prompt.

    Returns:
        The row, with the greedy label, sample labels, and the true, counterfactual,
        and mixed ("both") rates of the samples. We report all three because many
        answers with edited passages name both values (decided 2026-10-26).
    """
    greedy_label = label(result["greedy"], fact) # Label the greedy answer (true / cf / both / neither)
    sample_labels = [label(s, fact) for s in result["samples"]] # Label each sampled answer the same way
    true_rate = cf_rate = both_rate = None # Stay None when no samples were drawn (--samples 0)
    if sample_labels: # If there are samples
        true_rate = sample_labels.count("true") / len(sample_labels) # Fraction stating only the true value
        cf_rate = sample_labels.count("cf") / len(sample_labels) # Fraction stating only the counterfactual
        both_rate = sample_labels.count("both") / len(sample_labels) # Fraction stating both (mixed)
    return {
        "fact": fact["id"],
        "condition": condition,
        "wording": wording,
        "m": m,
        "question": fact["questions"][wording],
        "prompt": prompt,
        "prompt_tokens": result["prompt_tokens"],
        "greedy": result["greedy"],
        "label": greedy_label,
        "failure": failure_type(result["greedy"], greedy_label),
        "samples": result["samples"],
        "sample_labels": sample_labels,
        "true_rate": true_rate,
        "cf_rate": cf_rate,
        "both_rate": both_rate,
    }


def closed_book(llm: "Llama", fact: Fact, wording: int, n_samples: int) -> Row:
    """Run one Condition 1 example: the question with no passages.

    Args:
        llm: The loaded model.
        fact: The fact.
        wording: Which question wording (0, 1, or 2).
        n_samples: Number of sampled answers.

    Returns:
        The scored row.
    """
    prompt = closed_book_prompt(fact["questions"][wording]) # Question only, no passages
    result = llm.answer(prompt, n_samples=n_samples, keep_hidden=False) # Ask Llama (greedy + samples; no hidden states needed for closed-book)
    return make_row(fact, 1, wording, None, prompt, result) # Score and package as a row (m = None for closed-book)


def rag(llm: "Llama", fact: Fact, wording: int, m: int, n_samples: int) -> Row:
    """Run one RAG example with m edited passages and save its hidden states.

    Condition 2 is m = 0. Condition 3 will call this same function with m = 1 to 5.

    Args:
        llm: The loaded model.
        fact: The fact.
        wording: Which question wording (0, 1, or 2).
        m: Number of passages to edit.
        n_samples: Number of sampled answers.

    Returns:
        The scored row, including the edited slots, passage ids, and retrieval scores.
    """
    passages = top5(fact, wording) # The fact's 5 passages in slot order for this wording
    texts, slots = build_context(passages, fact, wording, m) # Edit m of them (nested slots, see edit.py)
    prompt = rag_prompt(fact["questions"][wording], texts) # Fill the RAG template
    result = llm.answer(prompt, n_samples=n_samples, keep_hidden=True) # Ask Llama (greedy + samples + hidden states)

    condition = 2 if m == 0 else 3 # m = 0 is Condition 2, m > 0 is Condition 3
    if result["hidden"] is not None: # If hidden states came back
        np.save(HIDDEN_DIR / f"{fact['id']}_c{condition}_w{wording}_m{m}.npy",
                result["hidden"].astype(np.float16)) # Save them as 16-bit to halve the file size, e.g. nafta_start_c2_w0_m0.npy

    row = make_row(fact, condition, wording, m, prompt, result) # Score and package as a row
    row["edited_slots"] = slots # Record which slots were edited (0-based)
    row["passage_ids"] = [p["id"] for p in passages] # Record which passages were used
    row["retrieval_scores"] = [round(p.get("score", 0.0), 4) for p in passages] # Record their retrieval similarity scores
    return row # Return the row


def already_done() -> set[tuple[str, int, int]]:
    """Find which (fact, condition, wording) combinations are already in the output file.

    Returns:
        The set of finished (fact id, condition, wording) triples.
    """
    done = set() # (fact, condition, wording) triples already saved
    if OUTPUT.exists(): # If there is an output file from an earlier run
        with open(OUTPUT, encoding="utf-8") as f: # Open it
            for line in f: # Sweep over saved rows
                row = json.loads(line) # Parse one row
                done.add((row["fact"], row["condition"], row["wording"])) # Mark its (fact, condition, wording) as done
    return done # Return the finished triples


def main() -> None:
    """Run Conditions 1 and 2 for every fact and wording not already done."""
    parser = argparse.ArgumentParser() # Command-line options
    parser.add_argument("--samples", type=int, default=N_SAMPLES) # --samples N: sampled answers per prompt (0 = greedy only)
    parser.add_argument("--facts", nargs="*", help="fact ids to run (default: all)") # --facts a b: only run these fact ids
    args = parser.parse_args() # Read the options

    from crag.generate import Llama # Imported here because loading torch/transformers is slow

    all_facts = facts.load() # Load every fact
    if args.facts: # If --facts was given
        all_facts = [f for f in all_facts if f["id"] in args.facts] # Keep only those facts

    RESULTS.mkdir(exist_ok=True) # Make results/ if needed
    HIDDEN_DIR.mkdir(exist_ok=True) # Make results/hidden/ if needed
    done = already_done() # What an earlier run already finished
    llm = Llama() # Load Llama once (~45 s on CPU)

    with open(OUTPUT, "a", encoding="utf-8") as out: # Append to results/baseline.jsonl
        for fact in all_facts: # Sweep over facts
            for wording in range(len(fact["questions"])): # Sweep over the 3 question wordings
                for condition in (1, 2): # Condition 1 (closed-book) then 2 (clean RAG)
                    if (fact["id"], condition, wording) in done: # If this one is already saved
                        continue # Skip it (lets a stopped run resume)
                    start = time.time() # Start timer
                    if condition == 1: # Closed-book
                        row = closed_book(llm, fact, wording, args.samples) # Run Condition 1
                    else: # Clean RAG
                        row = rag(llm, fact, wording, 0, args.samples) # Run Condition 2 (m = 0)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n") # Save the row as one JSON line
                    out.flush() # Write it to disk now, so a crash loses at most one row
                    seconds = time.time() - start # Time taken for this prompt
                    print(f"C{condition} {fact['id']} w{wording} {row['label']:8} "
                          f"{seconds:5.0f}s  {row['greedy'][:60]!r}") # Progress line: condition, fact, wording, label, time, start of the answer


if __name__ == "__main__":
    main() # Run Conditions 1 and 2
