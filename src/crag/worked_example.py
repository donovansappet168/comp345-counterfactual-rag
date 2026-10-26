"""Write the worked example for one fact as a Markdown file in examples/.

    python -m crag.worked_example nafta_start           # passages, edits, prompts
    python -m crag.worked_example nafta_start --run     # also run Llama at every m

The file walks through one fact: its 5 passages, the edited passages at m = 0 to 5,
the exact prompt, the scoring rules, and (with --run) Llama's answers.
"""

import argparse
import json
from typing import TYPE_CHECKING

from crag import facts
from crag.config import CHAT_DATE, K, N_SAMPLES, ROOT
from crag.edit import build_context
from crag.prompt import closed_book_prompt, rag_prompt
from crag.run_baseline import top5
from crag.schema import Fact, Passage
from crag.score import label

if TYPE_CHECKING: # Only for type hints; the real import is slow, so it happens in main()
    from crag.generate import Llama # Only for type hints; the real import is slow, so it happens in main()

EXAMPLES_DIR = ROOT / "examples"


def blockquote(text: str) -> str:
    """Format text as a Markdown blockquote.

    Args:
        text: The text.

    Returns:
        The text with "> " in front of every line.
    """
    return "> " + text.replace("\n", "\n> ") # Put '> ' in front of every line


def bold_edits(text: str, fact: Fact) -> str:
    """Make the inserted counterfactual values bold so they are easy to spot.

    Args:
        text: An edited passage.
        fact: The fact, whose "edits" list gives the counterfactual forms.

    Returns:
        The text with each counterfactual form wrapped in **...**.
    """
    for _, cf_form in fact["edits"]: # Sweep over the counterfactual forms
        text = text.replace(cf_form, f"**{cf_form}**") # Wrap each one in ** ** (bold in Markdown)
    return text # Return the marked-up text


def fact_section(fact: Fact) -> list[str]:
    """Section 1: the true and counterfactual values, the questions, and the edit rule.

    Args:
        fact: The fact.

    Returns:
        Markdown lines.
    """
    lines = ["## 1. The fact", ""] # Section heading
    lines.append(f"- True value (what the unedited corpus says): **{fact['true_value']}**") # True value
    lines.append(f"- Counterfactual value (what we insert): **{fact['cf_value']}**") # Counterfactual value
    lines.append(f"- Group: {fact.get('group') or 'decided by Condition 1 (section 6)'}") # Group, or a note that Condition 1 decides it
    if "note" in fact: # If the fact has a note
        lines.append(f"- Note: {fact['note']}") # Show it
    lines += ["", "The three question wordings:", ""] # Heading for the question list
    for i, q in enumerate(fact["questions"], start=1): # Sweep over wordings, counting from 1
        lines.append(f"{i}. {q}") # Numbered list item
    lines += ["", "Edit rule: in an edited passage, each form on the left becomes the form on the right. "
              "Nothing else in the passage changes.", "",
              "| In the corpus | After the edit |", "|---|---|"] # Edit rule + table header
    for true_form, cf_form in fact["edits"]: # Sweep over edit pairs
        lines.append(f"| `{true_form}` | `{cf_form}` |") # One table row per pair
    return lines + [""] # Return the lines, with a blank line at the end


def passages_section(passages: list[Passage], wording: int) -> list[str]:
    """Section 2: the 5 unedited passages in slot order, with their sources.

    Args:
        passages: The fact's passages, ranked by top5().
        wording: Which question wording they were ranked against.

    Returns:
        Markdown lines.
    """
    lines = ["## 2. The 5 passages (m = 0)", "",
             f"Ranked by all-MiniLM-L6-v2 cosine similarity to question wording {wording + 1}. "
             "Slot 1 is the most similar.", ""] # Section heading + how the slots are ordered
    for i, p in enumerate(passages, start=1): # Sweep over passages in slot order
        lines.append(f"**Slot {i}** (similarity {p.get('score', 0.0):.3f}): "
                     f"[{p['title']}]({p['url']}), {p['date']}") # Slot number, similarity, linked title, date
        lines += ["", blockquote(p["text"]), ""] # The passage text as a quote
    return lines # Return the lines


def edits_section(passages: list[Passage], fact: Fact,
                  wording: int) -> tuple[list[str], list[list[str]]]:
    """Section 3: which slots are edited at each m, and the edited passages.

    Args:
        passages: The fact's passages, ranked by top5().
        fact: The fact.
        wording: Which question wording.

    Returns:
        Markdown lines, and the 5 passage texts for each m (index m of the list).
    """
    lines = ["## 3. Edited passages at m = 0 to 5", "",
             "The edited slots are nested: one random order of the 5 slots is drawn for each fact and "
             "wording (fixed seed, `edited_slots` in `src/crag/edit.py`), and the first m slots of that "
             "order are edited, so each step of m adds exactly one wrong passage.", "",
             "| m | Edited slots | Fraction of wrong passages |", "|---|---|---|"] # Section heading + table header
    contexts = [] # The 5 passage texts for each m
    all_slots = [] # The edited slots for each m
    for m in range(K + 1): # Sweep over m = 0 to 5
        texts, slots = build_context(passages, fact, wording, m) # Edit m of the passages
        contexts.append(texts) # Keep the texts
        all_slots.append(slots) # Keep the slots
        slot_names = ", ".join(str(s + 1) for s in slots) or "none" # Slots as 1-based numbers, e.g. 2, 3
        lines.append(f"| {m} | {slot_names} | {m / K:.1f} |") # One table row per m
    lines.append("") # Blank line after the table

    for m in range(1, K + 1): # Sweep over m = 1 to 5
        slot_names = ", ".join(str(s + 1) for s in all_slots[m]) # Edited slots, 1-based
        lines += [f"<details><summary>m = {m} (slots {slot_names} edited)</summary>", ""] # Collapsible block per m
        for i, text in enumerate(contexts[m]): # Sweep over the 5 slots
            if i in all_slots[m]: # If this slot is edited
                lines += [f"**Slot {i + 1} (edited)**", "", blockquote(bold_edits(text, fact)), ""] # Show it with the inserted value in bold
            else: # Otherwise
                lines += [f"**Slot {i + 1}**", "", blockquote(text), ""] # Show it unchanged
        lines += ["</details>", ""] # Close the block
    return lines, contexts # Return the lines and the texts for each m


def prompt_section(question: str, clean_texts: list[str], llm: "Llama | None") -> list[str]:
    """Section 4: the exact prompt text (and, if Llama is loaded, the exact chat string).

    Args:
        question: The question wording used.
        clean_texts: The 5 unedited passage texts.
        llm: The loaded model, or None if we are not running it.

    Returns:
        Markdown lines.
    """
    prompt = rag_prompt(question, clean_texts) # The m = 0 prompt
    lines = ["## 4. The exact prompt", "",
             "Conditions 2 and 3 (`RAG_TEMPLATE` in `src/crag/prompt.py`), filled in for m = 0. "
             "For m > 0 only the edited passages change.", "",
             "```text", prompt, "```", "",
             "Condition 1 (closed-book) uses the same layout without the passages:", "",
             "```text", closed_book_prompt(question), "```", ""] # Section heading + RAG prompt + closed-book prompt
    if llm is not None: # If Llama is loaded
        raw = llm.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, tokenize=False,
                                                date_string=CHAT_DATE) # Same text encode() tokenizes, incl. the hidden date
        n_tokens = llm.encode(prompt).shape[1] # Prompt length in tokens
        lines += [f"Llama receives the prompt as one user message in its chat format ({n_tokens} tokens). "
                  "The saved hidden states come from the last token of this string.", "",
                  "<details><summary>Exact string given to the tokenizer</summary>", "",
                  "```text", raw, "```", "", "</details>", ""] # Show the exact chat-format string
    return lines # Return the lines


def scoring_section(fact: Fact) -> list[str]:
    """Section 5: the regex patterns the scoring script uses for this fact.

    Args:
        fact: The fact.

    Returns:
        Markdown lines.
    """
    return ["## 5. Scoring", "",
            "The script removes commas from the answer, then looks for these regex patterns. "
            "If it finds both values or neither, the answer is labelled by hand.", "",
            f"- True value: `{'`, `'.join(fact['true_patterns'])}`",
            f"- Counterfactual value: `{'`, `'.join(fact['cf_patterns'])}`", ""] # Section heading + the regex patterns


def outputs_section(llm: "Llama | None", fact: Fact, question: str,
                    contexts: list[list[str]], n_samples: int) -> list[str]:
    """Section 6: Llama's answers closed-book and at every m (only with --run).

    Also saves every sampled answer to examples/<fact id>_outputs.json.

    Args:
        llm: The loaded model, or None to skip this section.
        fact: The fact.
        question: The question wording used.
        contexts: The 5 passage texts for each m, from edits_section().
        n_samples: Number of sampled answers per prompt.

    Returns:
        Markdown lines.
    """
    lines = ["## 6. Llama 3.1 8B Instruct outputs", ""] # Section heading
    if llm is None: # If Llama is not loaded (no --run)
        return lines + ["Not run yet. Run again with `--run`.", ""] # Say so and stop

    lines += [f"Greedy answer plus {n_samples} sampled answers per prompt. "
              "True-answer rate = fraction of the sampled answers labelled `true`.", "",
              "| Condition | m | Greedy answer | Label | True-answer rate |", "|---|---|---|---|---|"] # Explanation + table header
    runs: list[tuple[str, str, str]] = [("1 (closed-book)", "-", closed_book_prompt(question))] # Closed-book first
    for m in range(K + 1): # Sweep over m = 0 to 5
        name = "2 (clean)" if m == 0 else "3 (edited)" # Condition name
        runs.append((name, str(m), rag_prompt(question, contexts[m]))) # Add the RAG prompt for this m

    saved = [] # Every answer, for the outputs JSON file
    for name, m_text, prompt in runs: # Sweep over the 7 prompts
        result = llm.answer(prompt, n_samples=n_samples, keep_hidden=False) # Ask Llama (greedy + samples; no hidden states needed here)
        sample_labels = [label(s, fact) for s in result["samples"]] # Label each sampled answer
        rate = sample_labels.count("true") / len(sample_labels) if sample_labels else float("nan") # True-answer rate (NaN if no samples)
        answer = result["greedy"].replace("\n", " ").replace("|", "\\|") # Greedy answer on one line, | escaped for the table
        lines.append(f"| {name} | {m_text} | {answer} | {label(result['greedy'], fact)} | {rate:.1f} |") # One table row per prompt
        saved.append({"condition": name, "m": m_text, "greedy": result["greedy"],
                      "samples": result["samples"], "sample_labels": sample_labels}) # Keep every answer
        print(name, m_text, repr(result["greedy"])) # Progress

    with open(EXAMPLES_DIR / f"{fact['id']}_outputs.json", "w", encoding="utf-8") as f: # Open the outputs JSON file
        json.dump(saved, f, indent=2, ensure_ascii=False) # Save every answer
    lines += ["", f"Every sampled answer is in `examples/{fact['id']}_outputs.json`.", ""] # Point to the file
    return lines # Return the lines


def main() -> None:
    """Build the worked example for the fact named on the command line."""
    parser = argparse.ArgumentParser() # Command-line options
    parser.add_argument("fact") # Fact id, e.g. nafta_start
    parser.add_argument("--wording", type=int, default=0) # --wording N: which question wording (default the first)
    parser.add_argument("--run", action="store_true") # --run: also run Llama (slow)
    parser.add_argument("--samples", type=int, default=N_SAMPLES) # --samples N: sampled answers per prompt
    args = parser.parse_args() # Read the options

    fact = facts.get(args.fact) # Load the fact
    question = fact["questions"][args.wording] # The question in this wording
    passages = top5(fact, args.wording) # Its 5 passages in slot order

    llm = None # No model unless --run
    if args.run: # If --run
        from crag.generate import Llama # Imported here because loading torch/transformers is slow
        llm = Llama() # Load Llama

    EXAMPLES_DIR.mkdir(exist_ok=True) # Make examples/ if needed
    lines = [f"# Worked example: {fact['id']}", "",
             "One real fact from the Baker Institute corpus, followed through each step of the experiment. "
             f"Generated by `python -m crag.worked_example {fact['id']}` from `data/facts.json`.", ""] # Title + one-line intro
    lines += fact_section(fact) # Section 1
    lines += passages_section(passages, args.wording) # Section 2
    edit_lines, contexts = edits_section(passages, fact, args.wording) # Section 3 (also returns the texts for each m)
    lines += edit_lines # Add its lines
    lines += prompt_section(question, contexts[0], llm) # Section 4
    lines += scoring_section(fact) # Section 5
    lines += outputs_section(llm, fact, question, contexts, args.samples) # Section 6 (runs Llama if loaded)

    path = EXAMPLES_DIR / f"{fact['id']}.md" # examples/<fact id>.md
    path.write_text("\n".join(lines), encoding="utf-8") # Write the Markdown file
    print(f"wrote {path}") # Report where


if __name__ == "__main__":
    main() # Build the worked example for the fact on the command line
