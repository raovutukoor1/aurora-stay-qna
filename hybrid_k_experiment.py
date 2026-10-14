"""Compare hybrid dense+BM25 retrieval at several final k values."""

import csv
import os
import sys
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


def main():
    hf_cache = Path(os.environ.get("TEMP", str(ROOT))) / "aurora-hf-cache"
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
    sys.path.insert(0, str(BACKEND))
    from rag_utils import retrieve_documents
    from precision_experiment import run_config

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
    configs = []
    for k in (4, 5, 6):
        for exact in (False, True):
            configs.append({"name": f"hybrid_k{k}_{'exact' if exact else 'noexact'}",
                            "k": k, "lambda_mult": 0.50, "fetch_k": 20,
                            "env": {"HIERARCHICAL_ROUTING": "true",
                                    "STRICT_ROUTING": "false",
                                    "STRICT_FINAL_FILTER": "false",
                                    "HYBRID_RETRIEVAL": "true",
                                    "EXACT_MATCH_BOOST": str(exact).lower(),
                                    "CONTEXT_COMPRESSION": "false"}})
    summary, details = [], []
    for config in configs:
        os.environ.update(config["env"])
        for dataset, frame in datasets.items():
            rows = run_config(frame, store, retrieve_documents, config)
            coverage = sum(r["coverage_proxy"] for r in rows) / len(rows)
            overlap = sum(r["mean_doc_overlap"] for r in rows) / len(rows)
            chars = sum(r["context_chars"] for r in rows) / len(rows)
            at75 = sum(r["coverage_proxy"] >= 0.75 for r in rows)
            summary.append({"config": config["name"], "dataset": dataset,
                            "mean_coverage_proxy": round(coverage, 4),
                            "mean_doc_overlap": round(overlap, 4),
                            "mean_context_chars": round(chars, 1),
                            "coverage_at_75": f"{at75}/10"})
            details.extend([{**config, "dataset": dataset, **row} for row in rows])
            print(f"{config['name']:24} {dataset:7} coverage={coverage:.3f} overlap={overlap:.3f} chars={chars:.0f} >=.75={at75}/10", flush=True)
    with (ROOT / "hybrid_k_experiment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    with (ROOT / "hybrid_k_experiment_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details[0].keys()); writer.writeheader(); writer.writerows(details)


if __name__ == "__main__":
    main()
