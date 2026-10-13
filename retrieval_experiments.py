"""Screen retrieval configurations and save comparable diagnostics.

This is a retrieval-only screening harness. Final candidates should be validated
with the full DeepEval notebook because lexical coverage is only a proxy for
Contextual Precision.
"""

import csv
import os
import re
from pathlib import Path

# Use the Windows certificate store for Hugging Face downloads.  The local
# Python OpenSSL bundle does not include the certificate chain used here.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DATA_DIR = ROOT / "data"


def words(value):
    stop = set("the a an and or is are to of for at in on what how can may be with from this that it as by do does will not only their its into after before one two".split())
    return {x for x in re.findall(r"[a-z0-9$]+", str(value).lower()) if len(x) > 2 and x not in stop}


def source_boost(query, source, amount=0.25):
    query = query.lower()
    source = Path(source).name.lower()
    topics = {
        ("cancellation", "cancel"): ("booking_terms", "cancellation"),
        ("refund", "charge"): ("refunds", "additional_charges"),
        ("room", "property"): ("room_and_property",),
        ("accessibility", "conduct"): ("guest_conduct", "accessibility"),
    }
    for query_terms, source_terms in topics.items():
        if any(term in query for term in query_terms) and any(term in source for term in source_terms):
            return amount
    return 0.0


def retrieve(query, store, reranker, k, fetch_k, lambda_mult, metadata_boost,
             rerank_threshold=None, deduplicate=True):
    candidates = store.max_marginal_relevance_search(
        query, k=max(k * 2, 10), fetch_k=fetch_k, lambda_mult=lambda_mult
    )
    scores = reranker.predict([(query, doc.page_content) for doc in candidates])
    ranked = sorted(
        zip(scores, candidates),
        key=lambda item: item[0] + metadata_boost * source_boost(
            query, item[1].metadata.get("source", "")
        ),
        reverse=True,
    )
    selected = []
    seen_tokens = []
    for score, doc in ranked:
        if rerank_threshold is not None and score < rerank_threshold:
            continue
        if deduplicate:
            tokens = set(words(doc.page_content))
            if tokens and any(len(tokens & previous) / max(1, len(tokens | previous)) >= 0.85 for previous in seen_tokens):
                continue
            seen_tokens.append(tokens)
        selected.append(doc)
        if len(selected) == k:
            break
    return selected


def evaluate_proxy(frame, store, reranker, config):
    rows = []
    for _, row in frame.iterrows():
        docs = retrieve(
            row["question"], store, reranker, config["k"], config["fetch_k"],
            config["lambda_mult"], config["metadata_boost"],
            config.get("rerank_threshold"), config.get("deduplicate", True)
        )
        retrieved = " ".join(doc.page_content for doc in docs)
        expected = words(row["expected_answer"])
        coverage = len(expected & words(retrieved)) / max(1, len(expected))
        rows.append({
            "question": row["question"],
            "coverage_proxy": coverage,
            "retrieved_sources": " | ".join(Path(d.metadata.get("source", "")).name for d in docs),
        })
    return rows


def main():
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    reranker = CrossEncoder(RERANKER_MODEL)
    docs = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        from docling.document_converter import DocumentConverter
        text = DocumentConverter().convert(str(pdf)).document.export_to_markdown()
        docs.append(Document(page_content=text, metadata={"source": str(pdf)}))

    frame = pd.read_csv(DATA_DIR / "evaluation.csv")
    configs = [
        {"name": "validated_baseline", "chunk_size": 512, "chunk_overlap": 64, "lambda_mult": 0.50, "fetch_k": 20, "metadata_boost": 0.25, "k": 5, "rerank_threshold": None, "deduplicate": False},
        {"name": "fetch40_lambda050", "chunk_size": 512, "chunk_overlap": 64, "lambda_mult": 0.50, "fetch_k": 40, "metadata_boost": 0.25, "k": 5, "rerank_threshold": None, "deduplicate": True},
        {"name": "fetch30_lambda055", "chunk_size": 512, "chunk_overlap": 64, "lambda_mult": 0.55, "fetch_k": 30, "metadata_boost": 0.25, "k": 5, "rerank_threshold": None, "deduplicate": True},
        {"name": "fetch40_lambda060", "chunk_size": 512, "chunk_overlap": 64, "lambda_mult": 0.60, "fetch_k": 40, "metadata_boost": 0.25, "k": 5, "rerank_threshold": None, "deduplicate": True},
        {"name": "fetch40_lambda060_threshold010", "chunk_size": 512, "chunk_overlap": 64, "lambda_mult": 0.60, "fetch_k": 40, "metadata_boost": 0.25, "k": 5, "rerank_threshold": 0.10, "deduplicate": True},
    ]
    summary, details = [], []
    for i, config in enumerate(configs, 1):
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", "##", ". ", " ", ""],
        )
        store = FAISS.from_documents(splitter.split_documents(docs), embeddings)
        rows = evaluate_proxy(frame, store, reranker, config)
        mean = sum(r["coverage_proxy"] for r in rows) / len(rows)
        high = sum(r["coverage_proxy"] >= 0.75 for r in rows)
        run_id = f"run_{i:02d}"
        summary.append({**config, "run_id": run_id, "chunks": store.index.ntotal, "mean_coverage_proxy": round(mean, 4), "coverage_at_75": high})
        details.extend([{**config, "run_id": run_id, **row} for row in rows])
        print(f"{run_id}: proxy={mean:.3f}, >=.75={high}/10, chunks={store.index.ntotal}", flush=True)

    summary.sort(key=lambda row: row["mean_coverage_proxy"], reverse=True)
    with (ROOT / "retrieval_experiment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader(); writer.writerows(summary)
    with (ROOT / "retrieval_experiment_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details[0].keys())
        writer.writeheader(); writer.writerows(details)
    with (ROOT / "retrieval_experiment_results.md").open("w", encoding="utf-8") as f:
        f.write("# Retrieval experiment results\n\n")
        f.write("These are lexical coverage screening results; validate finalists with DeepEval.\n\n")
        f.write("| Rank | Run | Chunk | Overlap | Lambda | Fetch k | Proxy | >=.75 |\n|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(summary, 1):
            f.write(f"| {rank} | {row['run_id']} | {row['chunk_size']} | {row['chunk_overlap']} | {row['lambda_mult']} | {row['fetch_k']} | {row['mean_coverage_proxy']:.3f} | {row['coverage_at_75']}/10 |\n")
        f.write("\nThe detailed per-question diagnostics are in `retrieval_experiment_details.csv`.\n")


if __name__ == "__main__":
    main()
