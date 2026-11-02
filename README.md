# MAESTRO Embedding Evaluation — UET Mardan Post-Graduate Prospectus 2025–26

This package benchmarks embedding models for **document retrieval** in MAESTRO using the supplied UET Mardan Post-Graduate Prospectus 2025–26. The script runs embedding inference on the CPU so the results can help assess models for a CPU-only deployment.

> **Scope:** The supplied PDF is the **Post-Graduate Prospectus 2025–26**, not an undergraduate prospectus. This benchmark does not represent the full UET Mardan knowledge base or all MAESTRO use cases.

## 1. Package structure

```text
MAESTRO_UETMardan_PG_Prospectus_Evaluation/
├── documents/
│   └── prospectus_chunks.csv
├── test_data/
│   ├── questions.csv
│   └── ground_truth.csv
├── evaluate_embeddings.py
├── requirements.txt
└── README.md
```

### Input files and expected columns

| File | Current size | Required columns | Purpose |
|---|---:|---|---|
| `documents/prospectus_chunks.csv` | 276 chunks | `chunk_id`, `text` | Text passages to embed and retrieve. Other metadata columns are retained in the CSV but are not required by the script. |
| `test_data/questions.csv` | 45 questions | `query_id`, `query` | User-style questions to embed and search. |
| `test_data/ground_truth.csv` | 122 mappings | `query_id`, `relevant_chunk_id` | Gold relevance labels: one row for each question-to-relevant-chunk relationship. |

The script uses these three paths automatically when run from the package. `ground_truth.csv` is the **source of truth** for relevance labels. The `relevant_chunk_ids` column in `questions.csv` is not used for scoring.

## 2. How the evaluation works

For each selected embedding model, the script:

1. Loads the model with `device="cpu"`.
2. Embeds all 276 document chunks.
3. Embeds all test queries (with model-specific prefixes where defined).
4. Calculates cosine similarity between each query vector and all document vectors.
5. Ranks document chunks for each query and evaluates the top K results (default K = 5).
6. Joins `questions.csv` with `ground_truth.csv` using `query_id` and checks that every gold chunk ID exists in the document CSV.

### Metrics

- **Hit@K:** share of questions with at least one gold chunk in the top K. Least affected by page-level labels, so use it with MRR as the main comparison.
- **Recall@K:** relevant chunks retrieved in the top K divided by all gold relevant chunks for that query.
- **Precision@K:** relevant chunks retrieved in the top K divided by K.
- **MRR (Mean Reciprocal Rank):** average reciprocal rank of the first relevant retrieved chunk.

Questions with no ground-truth mapping are still included in retrieval details, but excluded from the aggregate Recall, Precision, and MRR averages. The script prints a warning when such questions exist. It also checks duplicate document chunk IDs, duplicate question IDs, unknown query IDs, and gold chunk IDs missing from the document collection.

## 3. Install and run

**Recommended:** Python 3.10 or 3.11 in a clean virtual environment.

### Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python evaluate_embeddings.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python evaluate_embeddings.py
```

The first run downloads the embedding models from Hugging Face and therefore requires internet access. Model files are cached locally for later runs. The script explicitly selects CPU inference; it does not use a GPU.

**PyTorch installation note:** `requirements.txt` installs PyTorch as a dependency. If a strictly CPU-only PyTorch wheel is required on the evaluator's machine, install the appropriate CPU build using the official PyTorch installation selector for that operating system and Python version before installing/running the remaining requirements. Verify with:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The script still sets `device="cpu"` regardless of CUDA availability.

## 4. Optional command-line settings

Run one model at a time on a machine with limited RAM:

```bash
python evaluate_embeddings.py --models sentence-transformers/all-MiniLM-L6-v2 --batch-size 4
```

Compare a smaller set of models:

```bash
python evaluate_embeddings.py --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 --top-k 5 --batch-size 8
```

Override input paths if the files are stored elsewhere:

```bash
python evaluate_embeddings.py --documents documents/prospectus_chunks.csv --questions test_data/questions.csv --ground-truth test_data/ground_truth.csv
```

Available options:

| Option | Default | Meaning |
|---|---:|---|
| `--documents` | `documents/prospectus_chunks.csv` | CSV containing document chunks. |
| `--questions` | `test_data/questions.csv` | CSV containing query IDs and query text. |
| `--ground-truth` | `test_data/ground_truth.csv` | CSV containing query-to-relevant-chunk mappings. |
| `--top-k` | `5` | Number of top-ranked chunks used for metrics. Must be positive. |
| `--batch-size` | `8` | Number of texts encoded per batch. Reduce this if RAM is limited. Must be positive. |
| `--max-seq-length` | `512` | Token limit applied to every model so long chunks are truncated equally. `0` keeps each model's own limit. |
| `--models` | The 12 models listed below | One or more Hugging Face model IDs. |
| `--list-models` | – | Print the built-in model list and exit. |

### Default candidate models (12)

| # | Model | Group | Hugging Face ID |
|---|---|---|---|
| 1 | all-MiniLM-L6-v2 | Lightweight | `sentence-transformers/all-MiniLM-L6-v2` |
| 2 | BGE-small-en-v1.5 | Lightweight | `BAAI/bge-small-en-v1.5` |
| 3 | Snowflake Arctic S | Lightweight | `Snowflake/snowflake-arctic-embed-s` |
| 4 | GTE-small | Lightweight | `thenlper/gte-small` |
| 5 | Multilingual-E5-small | Multilingual | `intfloat/multilingual-e5-small` |
| 6 | Nomic Embed v1.5 | General | `nomic-ai/nomic-embed-text-v1.5` |
| 7 | Nomic Embed v2 MoE | Multilingual | `nomic-ai/nomic-embed-text-v2-moe` |
| 8 | BGE-base-en-v1.5 | General | `BAAI/bge-base-en-v1.5` |
| 9 | GTE-base | General | `thenlper/gte-base` |
| 10 | Jina Embeddings v5 Text Nano | Multilingual | `jinaai/jina-embeddings-v5-text-nano-retrieval` |
| 11 | Qwen3-Embedding-0.6B | Advanced | `Qwen/Qwen3-Embedding-0.6B` |
| 12 | BGE-M3 | Advanced | `BAAI/bge-m3` |

The Nomic and Jina models run custom code (`trust_remote_code=True`). Nomic needs `einops`. Nomic v2 MoE, BGE-M3 and Qwen3 are slow on CPU; use `--batch-size 4` or run them one at a time with `--models` if memory is limited.

These models differ in size, embedding dimension, architecture, and supported input length. Long chunks are truncated at `--max-seq-length` (default 512 tokens) for every model so the comparison is equal; the results record each model's native and effective limit. Note that all-MiniLM-L6-v2 was trained at 256 tokens.

## 5. Output files

After a run, the script creates a `results/` directory beside `evaluate_embeddings.py`:

- `results/model_comparison.csv` — one summary row per successfully evaluated model, including retrieval metrics, embedding times, embedding dimension, and process memory measurements.
- `results/retrieval_details.csv` — per-question gold IDs, retrieved IDs, top-1 result, and per-query metrics (including hit@K).
- `results/errors.csv` — model-specific errors, if any model fails to load or evaluate.

If no model completes successfully, check the terminal output and `errors.csv` (if created). Do not compare incomplete runs as if every model succeeded.

## 6. Important limitations and fair-comparison guidance

1. **Gold-label quality:** The current relevance mappings were created from source-page locations in the prospectus. They are a page-level baseline; a page can produce multiple chunks, and not every chunk from that page necessarily answers the question. Manually review and narrow the gold labels to directly relevant chunks before treating metric scores as final. Until then, Recall@K is capped by the over-inclusive labels; compare models mainly on Hit@K and MRR. Labels for Q001, Q026 and Q033–Q037 were corrected against the PDF.
2. **Unlabeled questions:** A question without a row in `ground_truth.csv` is not scored. Add one or more correct `relevant_chunk_id` mappings before using that question in aggregate metric comparisons.
3. **Model-specific instructions:** The script applies each model's documented query/document prefixes or named prompts (for example BGE's query instruction, E5 `query:`/`passage:`, Nomic `search_query:`/`search_document:`, and the Qwen3/Jina query prompts). If a model has no matching prompt defined, the script prints a note and encodes without it. Report any change to these settings alongside the results.
4. **Same conditions:** Compare models using the same document chunks, questions, gold labels, machine, CPU setting, top-K, and comparable batch settings. Record CPU model, RAM, operating system, Python, PyTorch, and sentence-transformers versions.
5. **Timing:** Model download time is not included in model-load timing if the model is already cached. First-time downloads require internet and can substantially increase the time before evaluation starts.
6. **Memory:** RSS values are process memory snapshots, not a complete measure of peak RAM or model file size. Cached model size is best-effort and may be unavailable.
7. **Vector database:** Embeddings from different models belong to different vector spaces. If the production embedding model changes, re-embed the complete MAESTRO document collection before using that model for retrieval in ChromaDB.
8. **Evaluation scope:** This script measures embedding-based retrieval only. It does not evaluate generated answer correctness, faithfulness, citation accuracy, or end-to-end RAG quality.
9. **Duplicate PDF pages:** PDF pages 24–28 repeat as pages 48–52, so the chunk file contains duplicated text. No test question is labelled on those pages.
10. **PDF extraction:** Check tables, equations, scanned pages, and complex layouts against the original PDF. Text extraction can lose table relationships or reading order.

## 7. Suggested additional MAESTRO documents

To evaluate a broader university assistant, add verified and current versions of:

- Undergraduate prospectus.
- Current admission policies and admission advertisements.
- Official fee notifications and fee schedules.
- Academic calendar and examination notifications.
- Student handbook and updated university rules.

Create corresponding chunks, questions, and carefully reviewed ground-truth mappings for each added source. Keep the source document and page metadata so retrieval results can be audited.