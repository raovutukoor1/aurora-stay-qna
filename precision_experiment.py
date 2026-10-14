"""Controlled retrieval experiment for contextual-precision improvements.

This is a retrieval screening experiment. It does not alter the production
defaults. It compares the current configuration with stricter routing,
lexical exact-match support, context compression, and a smaller final context.
"""

import csv
import os
import re
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
DATA = ROOT / "data"


def tokens(value):
    return set(re.findall(r"[a-z0-9$-]+", str(value).lower()))


def run_config(frame, store, retrieve_documents, config):
    rows = []
    for _, row in frame.iterrows():
        docs = retrieve_documents(
            query=row["question"],
            k=config["k"],
            vectorstore=store,
            lambda_mult=config["lambda_mult"],
            fetch_k=config["fetch_k"],
        )
        retrieved = " ".join(doc.page_content for doc in docs)
        expected = tokens(row["expected_answer"])
        retrieved_tokens = tokens(retrieved)
        coverage = len(expected & retrieved_tokens) / max(1, len(expected))
        # Relevant-context proxy: expected-answer vocabulary not present in the
        # retrieved context is a useful signal for recall; context length and
        # non-overlap approximate noise for precision screening.
        overlap_by_doc = []
        for doc in docs:
            overlap_by_doc.append(len(expected & tokens(doc.page_content)) / max(1, len(expected)))
        rows.append({
            "question": row["question"],
            "coverage_proxy": coverage,
            "mean_doc_overlap": sum(overlap_by_doc) / max(1, len(overlap_by_doc)),
            "chunks_returned": len(docs),
            "context_chars": len(retrieved),
            "sources": " | ".join(Path(d.metadata.get("source", "")).name for d in docs),
        })
    return rows


def main():
    # Keep the Hugging Face cache on a short local path; the workspace path is
    # long enough to exceed Windows' maximum path length during downloads.
    hf_cache = Path(os.environ.get("TEMP", str(ROOT))) / "aurora-hf-cache"
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
    import sys
    sys.path.insert(0, str(BACKEND))
    from rag_utils import retrieve_documents

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBED_MODEL", "HarishMaths/Hotel-Policy-Embedding"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.load_local(
        str(BACKEND / "faiss_index"), embeddings, allow_dangerous_deserialization=True
    )
    datasets = {
        "dev": pd.read_csv(DATA / "evaluation.csv"),
        "heldout": pd.read_excel(DATA / "test.xlsx"),
    }
    configs = [
        {"name": "baseline_k5", "k": 5, "lambda_mult": 0.50, "fetch_k": 20,
         "env": {"HIERARCHICAL_ROUTING": "true", "STRICT_ROUTING": "true", "STRICT_FINAL_FILTER": "false", "HYBRID_RETRIEVAL": "false", "EXACT_MATCH_BOOST": "false", "CONTEXT_COMPRESSION": "false"}},
        {"name": "strict_k3", "k": 3, "lambda_mult": 0.50, "fetch_k": 20,
         "env": {"HIERARCHICAL_ROUTING": "true", "STRICT_ROUTING": "true", "STRICT_FINAL_FILTER": "true", "HYBRID_RETRIEVAL": "false", "EXACT_MATCH_BOOST": "false", "CONTEXT_COMPRESSION": "false"}},
        {"name": "hybrid_exact_k3", "k": 3, "lambda_mult": 0.50, "fetch_k": 20,
         "env": {"HIERARCHICAL_ROUTING": "true", "STRICT_ROUTING": "true", "STRICT_FINAL_FILTER": "true", "HYBRID_RETRIEVAL": "true", "EXACT_MATCH_BOOST": "true", "CONTEXT_COMPRESSION": "false"}},
        {"name": "hybrid_exact_compressed_k3", "k": 3, "lambda_mult": 0.50, "fetch_k": 20,
         "env": {"HIERARCHICAL_ROUTING": "true", "STRICT_ROUTING": "true", "STRICT_FINAL_FILTER": "true", "HYBRID_RETRIEVAL": "true", "EXACT_MATCH_BOOST": "true", "CONTEXT_COMPRESSION": "true"}},
        {"name": "hybrid_exact_compressed_k2", "k": 2, "lambda_mult": 0.50, "fetch_k": 20,
         "env": {"HIERARCHICAL_ROUTING": "true", "STRICT_ROUTING": "true", "STRICT_FINAL_FILTER": "true", "HYBRID_RETRIEVAL": "true", "EXACT_MATCH_BOOST": "true", "CONTEXT_COMPRESSION": "true"}},
    ]
    summary, details = [], []
    for config in configs:
        os.environ.update(config["env"])
        for dataset, frame in datasets.items():
            rows = run_config(frame, store, retrieve_documents, config)
            mean_coverage = sum(r["coverage_proxy"] for r in rows) / len(rows)
            mean_overlap = sum(r["mean_doc_overlap"] for r in rows) / len(rows)
            mean_chars = sum(r["context_chars"] for r in rows) / len(rows)
            at75 = sum(r["coverage_proxy"] >= 0.75 for r in rows)
            summary.append({"config": config["name"], "dataset": dataset,
                            "mean_coverage_proxy": round(mean_coverage, 4),
                            "mean_doc_overlap": round(mean_overlap, 4),
                            "mean_context_chars": round(mean_chars, 1),
                            "coverage_at_75": f"{at75}/10"})
            details.extend([{**config, "dataset": dataset, **row} for row in rows])
            print(f"{config['name']:32} {dataset:7} coverage={mean_coverage:.3f} doc_overlap={mean_overlap:.3f} chars={mean_chars:.0f} >=.75={at75}/10", flush=True)
    with (ROOT / "precision_experiment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    with (ROOT / "precision_experiment_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details[0].keys()); writer.writeheader(); writer.writerows(details)


if __name__ == "__main__":
    main()
