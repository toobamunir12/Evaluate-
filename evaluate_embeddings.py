#!/usr/bin/env python3
"""CPU-focused embedding and retrieval evaluation for MAESTRO.

Default inputs (relative to this file; falls back to the same file name, or
<name>_corrected.csv, placed directly next to this script):
  documents/prospectus_chunks.csv : chunk_id, text, ...
  test_data/questions.csv         : query_id, query, ...
  test_data/ground_truth.csv      : query_id, relevant_chunk_id

Run from this folder:
  python evaluate_embeddings.py

Optional:
  python evaluate_embeddings.py --top-k 5 --batch-size 8 --max-seq-length 512
  python evaluate_embeddings.py --models BAAI/bge-small-en-v1.5 thenlper/gte-small
  python evaluate_embeddings.py --list-models

Extra packages some models need:
  pip install einops           (Nomic v1.5 / v2 MoE)
  pip install -U sentence-transformers transformers   (Qwen3, Jina v5)

The script uses CPU intentionally so model comparisons reflect MAESTRO's
CPU-only deployment constraint. It does not evaluate generated answers.
"""
import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Per-model settings. Retrieval quality depends heavily on using each model's
# documented query/document prefixes, so they are applied here.
#   label            : short name shown in the results
#   group            : Lightweight / General / Multilingual / Advanced
#   query_prefix     : text prepended to every query
#   doc_prefix       : text prepended to every document chunk
#   query_prompt     : sentence-transformers prompt_name for queries (if the
#                      model ships named prompts); skipped if not available
#   doc_prompt       : same, for documents
#   trust_remote_code: needed by models with custom code
MODEL_REGISTRY = {
    # 1 Lightweight
    "sentence-transformers/all-MiniLM-L6-v2": {
        "label": "all-MiniLM-L6-v2", "group": "Lightweight"},
    # 2 Lightweight
    "BAAI/bge-small-en-v1.5": {
        "label": "BGE-small-en-v1.5", "group": "Lightweight",
        "query_prefix": BGE_QUERY_PREFIX},
    # 3 Lightweight
    "Snowflake/snowflake-arctic-embed-s": {
        "label": "Snowflake Arctic S", "group": "Lightweight",
        "query_prefix": BGE_QUERY_PREFIX},
    # 4 Lightweight
    "thenlper/gte-small": {
        "label": "GTE-small", "group": "Lightweight"},
    # 5 Multilingual
    "intfloat/multilingual-e5-small": {
        "label": "Multilingual-E5-small", "group": "Multilingual",
        "query_prefix": "query: ", "doc_prefix": "passage: "},
    # 6 General
    "nomic-ai/nomic-embed-text-v1.5": {
        "label": "Nomic Embed v1.5", "group": "General",
        "query_prefix": "search_query: ", "doc_prefix": "search_document: ",
        "trust_remote_code": True},
    # 7 Multilingual
    "nomic-ai/nomic-embed-text-v2-moe": {
        "label": "Nomic Embed v2 MoE", "group": "Multilingual",
        "query_prefix": "search_query: ", "doc_prefix": "search_document: ",
        "trust_remote_code": True},
    # 8 General
    "BAAI/bge-base-en-v1.5": {
        "label": "BGE-base-en-v1.5", "group": "General",
        "query_prefix": BGE_QUERY_PREFIX},
    # 9 General
    "thenlper/gte-base": {
        "label": "GTE-base", "group": "General"},
    # 10 Multilingual (retrieval adapter of Jina v5 text nano)
    "jinaai/jina-embeddings-v5-text-nano-retrieval": {
        "label": "Jina Embeddings v5 Text Nano", "group": "Multilingual",
        "query_prompt": "query", "doc_prompt": "document",
        "trust_remote_code": True},
    # 11 Advanced
    "Qwen/Qwen3-Embedding-0.6B": {
        "label": "Qwen3-Embedding-0.6B", "group": "Advanced",
        "query_prompt": "query"},
    # 12 Advanced
    "BAAI/bge-m3": {
        "label": "BGE-M3", "group": "Advanced"},
}

DEFAULT_MODELS = list(MODEL_REGISTRY.keys())

ROOT = Path(__file__).resolve().parent
DEFAULT_DOCUMENTS = ROOT / "documents" / "prospectus_chunks.csv"
DEFAULT_QUESTIONS = ROOT / "test_data" / "questions.csv"
DEFAULT_GROUND_TRUTH = ROOT / "test_data" / "ground_truth.csv"


def resolve_path(path_str, kind):
    """Use the given path, or fall back to flat / *_corrected.csv next to this script."""
    path = Path(path_str)
    if path.is_file():
        return path
    for candidate in (ROOT / path.name, ROOT / f"{path.stem}_corrected{path.suffix}"):
        if candidate.is_file():
            print(f"NOTE: {kind} CSV not found at {path}; using {candidate}")
            return candidate
    raise FileNotFoundError(f"{kind} CSV not found: {path}")


def process_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def cached_model_size_mb(model_name):
    """Best-effort size of a model's Hugging Face cache entry."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        matches = [repo for repo in info.repos if repo.repo_id == model_name]
        if matches:
            return round(matches[0].size_on_disk / (1024 ** 2), 2)
    except Exception:
        pass
    return None


def parse_relevant(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip()}


def encode_texts(model, texts, prefix, prompt_name, batch_size):
    """Encode with the model's prefix / named prompt when configured."""
    if prefix:
        texts = [prefix + text for text in texts]
    kwargs = dict(
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if prompt_name:
        available = getattr(model, "prompts", None) or {}
        if prompt_name in available:
            kwargs["prompt_name"] = prompt_name
        else:
            print(f"  NOTE: prompt '{prompt_name}' not defined for this model; encoding without it.")
    return model.encode(texts, **kwargs)


def evaluate_one(model_name, docs, questions, top_k, batch_size, max_seq_length):
    cfg = MODEL_REGISTRY.get(model_name, {})
    label = cfg.get("label", model_name)
    print(f"\n--- Evaluating {label} ({model_name}) on CPU ---")
    gc.collect()
    rss_before = process_rss_mb()
    load_start = time.perf_counter()
    model = SentenceTransformer(
        model_name, device="cpu", trust_remote_code=cfg.get("trust_remote_code", False)
    )
    load_seconds = time.perf_counter() - load_start
    rss_after_load = process_rss_mb()

    native_max_seq = getattr(model, "max_seq_length", None)
    if max_seq_length:
        try:
            model.max_seq_length = max_seq_length
        except Exception as exc:
            print(f"  NOTE: could not set max_seq_length ({exc}); keeping {native_max_seq}.")
    effective_max_seq = getattr(model, "max_seq_length", None)

    document_texts = docs["text"].astype(str).tolist()
    document_start = time.perf_counter()
    document_vectors = encode_texts(
        model, document_texts, cfg.get("doc_prefix", ""), cfg.get("doc_prompt"), batch_size
    )
    document_seconds = time.perf_counter() - document_start

    query_texts = questions["query"].astype(str).tolist()
    query_start = time.perf_counter()
    query_vectors = encode_texts(
        model, query_texts, cfg.get("query_prefix", ""), cfg.get("query_prompt"), batch_size
    )
    query_seconds = time.perf_counter() - query_start

    similarities = cosine_similarity(query_vectors, document_vectors)
    k = min(top_k, len(docs))
    ranked_indices = np.argsort(-similarities, axis=1)[:, :k]
    chunk_ids = docs["chunk_id"].astype(str).tolist()

    recalls, precisions, reciprocal_ranks, hit_flags = [], [], [], []
    detail_rows = []
    for row_index, (_, question_row) in enumerate(questions.iterrows()):
        relevant = parse_relevant(question_row["relevant_chunk_ids"])
        ranked_ids = [chunk_ids[index] for index in ranked_indices[row_index]]
        hits = [chunk_id for chunk_id in ranked_ids if chunk_id in relevant]

        # Questions without a gold label are retrieved but excluded from aggregate
        # quality metrics; otherwise they would incorrectly count as zero scores.
        if relevant:
            recall = len(set(hits)) / len(relevant)
            precision = len(hits) / k if k else 0.0
            hit = 1.0 if hits else 0.0
            reciprocal_rank = next(
                (1.0 / rank for rank, chunk_id in enumerate(ranked_ids, start=1)
                 if chunk_id in relevant),
                0.0,
            )
            recalls.append(recall)
            precisions.append(precision)
            hit_flags.append(hit)
            reciprocal_ranks.append(reciprocal_rank)
        else:
            recall = precision = hit = reciprocal_rank = np.nan

        detail_rows.append({
            "model": model_name,
            "model_label": label,
            "query_id": question_row.get("query_id", row_index),
            "query": question_row["query"],
            "has_ground_truth": bool(relevant),
            "relevant_chunk_ids": ";".join(sorted(relevant)),
            "retrieved_chunk_ids": ";".join(ranked_ids),
            "top1_chunk_id": ranked_ids[0] if ranked_ids else "",
            f"recall_at_{k}": recall,
            f"precision_at_{k}": precision,
            f"hit_at_{k}": hit,
            "reciprocal_rank": reciprocal_rank,
        })

    metrics = {
        "model": model_name,
        "model_label": label,
        "group": cfg.get("group", ""),
        "device": "CPU",
        "embedding_dimension": int(document_vectors.shape[1]),
        "native_max_seq_length": native_max_seq,
        "effective_max_seq_length": effective_max_seq,
        "model_load_seconds": round(load_seconds, 3),
        "document_embedding_seconds": round(document_seconds, 3),
        "document_chunks_per_second": round(len(document_texts) / document_seconds, 3)
        if document_seconds else None,
        "query_embedding_seconds_total": round(query_seconds, 3),
        "average_query_embedding_ms": round((query_seconds / len(query_texts)) * 1000, 3)
        if query_texts else None,
        f"recall_at_{k}": round(float(np.mean(recalls)), 4) if recalls else None,
        f"precision_at_{k}": round(float(np.mean(precisions)), 4) if precisions else None,
        f"hit_at_{k}": round(float(np.mean(hit_flags)), 4) if hit_flags else None,
        "mrr": round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else None,
        "rss_before_load_mb": round(rss_before, 2),
        "rss_after_load_mb": round(rss_after_load, 2),
        "rss_after_embeddings_mb": round(process_rss_mb(), 2),
        "cached_model_size_mb_best_effort": cached_model_size_mb(model_name),
        "number_of_document_chunks": len(docs),
        "number_of_test_queries": len(questions),
        "number_of_queries_with_ground_truth": len(recalls),
        "number_of_queries_without_ground_truth": len(questions) - len(recalls),
        "top_k": k,
    }

    del model, document_vectors, query_vectors, similarities
    gc.collect()
    return metrics, detail_rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare embedding models on the MAESTRO prospectus chunks (CPU)."
    )
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS),
                        help="CSV containing chunk_id and text (default: documents/prospectus_chunks.csv)")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS),
                        help="CSV containing query_id and query (default: test_data/questions.csv)")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH),
                        help="CSV containing query_id and relevant_chunk_id (default: test_data/ground_truth.csv)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Token limit applied to every model so long chunks are "
                             "truncated equally (default 512; 0 keeps each model's own limit).")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--list-models", action="store_true",
                        help="Print the built-in model list and exit.")
    args = parser.parse_args()

    if args.list_models:
        for number, (name, cfg) in enumerate(MODEL_REGISTRY.items(), start=1):
            print(f"{number:>2}. {cfg['label']:<30} {cfg['group']:<13} {name}")
        return

    if args.top_k < 1 or args.batch_size < 1 or args.max_seq_length < 0:
        parser.error("--top-k and --batch-size must be positive; --max-seq-length must be >= 0.")

    documents_path = resolve_path(args.documents, "Documents")
    questions_path = resolve_path(args.questions, "Questions")
    ground_truth_path = resolve_path(args.ground_truth, "Ground-truth")

    docs = pd.read_csv(documents_path).fillna("")
    questions = pd.read_csv(questions_path).fillna("")
    ground_truth = pd.read_csv(ground_truth_path).fillna("")
    required_docs = {"chunk_id", "text"}
    required_questions = {"query_id", "query"}
    required_ground_truth = {"query_id", "relevant_chunk_id"}
    if not required_docs.issubset(docs.columns):
        raise ValueError(f"Documents CSV must contain columns: {sorted(required_docs)}")
    if not required_questions.issubset(questions.columns):
        raise ValueError(f"Questions CSV must contain columns: {sorted(required_questions)}")
    if not required_ground_truth.issubset(ground_truth.columns):
        raise ValueError(f"Ground-truth CSV must contain columns: {sorted(required_ground_truth)}")
    if docs.empty or questions.empty or ground_truth.empty:
        raise ValueError("Documents, questions, and ground-truth CSV files must contain data rows.")
    if docs["chunk_id"].astype(str).duplicated().any():
        raise ValueError("chunk_id values in the documents CSV must be unique.")

    if questions["query_id"].astype(str).duplicated().any():
        raise ValueError("query_id values in questions.csv must be unique.")
    available_ids = set(docs["chunk_id"].astype(str))
    questions["query_id"] = questions["query_id"].astype(str)
    ground_truth["query_id"] = ground_truth["query_id"].astype(str)
    ground_truth["relevant_chunk_id"] = ground_truth["relevant_chunk_id"].astype(str).str.strip()
    ground_truth = ground_truth[ground_truth["relevant_chunk_id"] != ""]

    unknown_queries = set(ground_truth["query_id"]) - set(questions["query_id"])
    if unknown_queries:
        raise ValueError("Ground-truth query IDs are missing from questions.csv: "
                         + ", ".join(sorted(unknown_queries)))
    missing_gold_ids = set(ground_truth["relevant_chunk_id"]) - available_ids
    if missing_gold_ids:
        raise ValueError("Ground-truth chunk IDs are missing from documents CSV: "
                         + ", ".join(sorted(missing_gold_ids)))

    grouped_gold = (ground_truth.groupby("query_id")["relevant_chunk_id"]
                    .apply(lambda values: ";".join(sorted(set(values))))
                    .to_dict())
    questions["relevant_chunk_ids"] = questions["query_id"].map(grouped_gold).fillna("")
    unlabeled_count = int((questions["relevant_chunk_ids"] == "").sum())
    if unlabeled_count:
        print(f"WARNING: {unlabeled_count} question(s) have no ground-truth mapping. "
              "They will be excluded from Recall/Precision/MRR averages.")
    print(f"Loaded {len(docs)} chunks, {len(questions)} questions, "
          f"and {len(ground_truth)} ground-truth mappings.")

    all_metrics, all_details, errors = [], [], []
    for model_name in args.models:
        try:
            metrics, details = evaluate_one(
                model_name, docs, questions, args.top_k, args.batch_size,
                args.max_seq_length or None,
            )
            all_metrics.append(metrics)
            all_details.extend(details)
        except Exception as exc:
            errors.append({"model": model_name, "error": repr(exc)})
            print(f"ERROR for {model_name}: {exc}")

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    if all_metrics:
        k = min(args.top_k, len(docs))
        comparison = pd.DataFrame(all_metrics)
        comparison.to_csv(output_dir / "model_comparison.csv", index=False)
        pd.DataFrame(all_details).to_csv(output_dir / "retrieval_details.csv", index=False)
        metric_columns = ["model_label", "group", f"hit_at_{k}", f"recall_at_{k}",
                          f"precision_at_{k}", "mrr",
                          "document_embedding_seconds", "rss_after_load_mb"]
        print("\nModel comparison:")
        print(comparison[metric_columns].to_string(index=False))
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)

    print("\nEvaluation finished.")
    print(f"Results folder: {output_dir}")
    if not all_metrics:
        print("No model completed successfully. See the errors above.")


if __name__ == "__main__":
    main()
