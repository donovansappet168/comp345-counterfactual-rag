"""All settings in one place - change values in this file ONLY."""

from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
RAW_PAGES = ROOT / "data" / "raw" / "baker_pages.jsonl"         # scraped pages
PDF_DIR = ROOT / "data" / "raw" / "pdfs"                        # downloaded PDF files
RAW_PDFS = ROOT / "data" / "raw" / "baker_pdfs.jsonl"           # text pulled from the PDFs
PASSAGES = ROOT / "data" / "raw" / "passages.jsonl"             # chunked passages
INDEX = ROOT / "data" / "raw" / "passage_embeddings.npy"        # MiniLM vectors
FACTS = ROOT / "data" / "facts.json"                            # hand-checked facts
FACT_CANDIDATES = ROOT / "data" / "fact_candidates.csv"         # ranked list from mine_facts.py
RESULTS = ROOT / "results"

# Models
RETRIEVER = "sentence-transformers/all-MiniLM-L6-v2"
# unsloth's copy of llama has the same weights as meta-llama/Llama-3.1-8B-Instruct but does not
# need a Hugging Face access request. If you have access, switch to the Meta directory.
GENERATOR = "unsloth/Llama-3.1-8B-Instruct"

# Experiment
K = 5                   # passages per prompt
N_SAMPLES = 10          # sampled answers per prompt (true-answer rate)
MAX_NEW_TOKENS = 192    # max number of tokens used for answer - room for Llama to go through every passage
                        # (32 -> 64 on 2026-10-25, 64 -> 192 on 2026-10-28: mixed answers often name
                        # the second value after token 64; in a 50-answer test at 512 tokens, every
                        # label was already final by 128 and 96% of answers ended on their own by 192)
SEED = 345              # uniform seed for repeatability
# Llama 3.1's chat template always adds a system message "Today Date: ...". Its default is
# 26 Jul 2024, which made Llama call 2026 events "hypothetical". We pin one fixed date, the
# same in every condition (decided 2026-10-26). Format must be "DD Mon YYYY".
CHAT_DATE = "26 Oct 2026"
CHUNK_WORDS = 120       # minimum passage length in words; passages run longer because
                        # chunk() never splits a paragraph (see passages.chunk)
MAX_FACT_PASSAGE_WORDS = 200  # a fact's passages should fit in what MiniLM reads (~256 tokens)

# Fact mining (mine_facts.py)
MINE_MIN_DOCUMENTS = 5  # keep candidates stated in at least this many different documents
MINE_MIN_LIFT = 3.0    # how much more often the value appears with this subject than in general
MINE_WINDOW = 12        # a subject and a value must be within this many words in one sentence
