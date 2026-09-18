# SEC 10-K RAG System

A local Retrieval-Augmented Generation system for analyzing five years of SEC 10-K filings from one bank.

## Architecture

1. Parse local PDF/HTML filings
2. Extract 10-K sections
3. Create overlapping chunks with year/section/source metadata
4. Generate OpenAI embeddings
5. Build a FAISS vector index
6. Combine semantic and BM25 lexical retrieval
7. Generate grounded answers with source IDs
8. Evaluate retrieval and generation quality

## Project structure

```text
sec_rag/
├── data/
│   └── filings/
├── evaluation/
│   └── questions.json
├── indexes/
├── src/
│   ├── parser.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── generator.py
│   ├── rag.py
│   └── evaluate.py
├── build_index.py
├── query.py
├── run_evaluation.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key.

Put the five local filings into:

```text
data/filings/
```

Filenames should contain the filing year, for example:

```text
2021_10k.pdf
2022_10k.pdf
2023_10k.pdf
2024_10k.pdf
2025_10k.pdf
```

## Build the index

```bash
python build_index.py
```

This creates:

```text
indexes/faiss.index
indexes/metadata.json
```

## Query the system

```bash
python query.py
```

## Run evaluation

```bash
python run_evaluation.py
```

The evaluation includes:

- Recall@5
- MRR
- Faithfulness
- Relevance
- Completeness
- Citation accuracy
- Hallucination rate

For meaningful retrieval metrics, populate `relevant_chunks` in `evaluation/questions.json` with manually verified chunk IDs.

## Recommended evaluation dataset

For a serious benchmark, use 50–100 questions covering:

- financial statements
- revenue and net income
- net interest income
- credit losses
- capital ratios
- liquidity
- risk factors
- regulatory matters
- multi-year comparisons
- numerical questions
- unanswerable questions

For financial RAG, also add a separate numerical-accuracy benchmark where each question has an expected value and unit.
