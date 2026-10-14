"""Run DeepEval on held-out test.xlsx for retrieval option comparison."""

import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.evaluate.configs import AsyncConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
DATA = ROOT / "data"


def load_config():
    config = json.loads((ROOT / "config" / "config.json").read_text(encoding="utf-8"))
    supplied = ROOT / "Supplied files" / "confignewproject4.txt"
    if not config.get("OPENAI_API_KEY") and supplied.exists():
        fallback = json.loads(supplied.read_text(encoding="utf-8-sig"))
        config.update({key: value for key, value in fallback.items() if value})
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = config.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return config


def metrics(eval_model):
    return [
        GEval(name="Faithfulness",
              criteria="Determine whether the actual output is factually supported by the provided retrieval context.",
              evaluation_steps=["Break the actual output into individual factual claims.", "Check whether each claim is supported by the retrieval context.", "Penalize unsupported or contradicted claims.", "Do not penalize wording differences when meaning is supported.", "Give a higher score when all important claims are supported by the context."],
              evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT], threshold=0.85, model=eval_model),
        GEval(name="Contextual Recall",
              criteria="Determine whether the retrieval context contains the information needed to support the expected output.",
              evaluation_steps=["Break the expected output into important factual claims.", "Check whether each claim is present or supported by the retrieval context.", "Penalize missing information required for the expected output.", "Do not penalize wording differences.", "Give a higher score when the retrieval context contains most or all required information."],
              evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT], threshold=0.80, model=eval_model),
        GEval(name="Contextual Precision",
              criteria="Determine whether the retrieved context is relevant to answering the input question, with relevant information prioritized over irrelevant information.",
              evaluation_steps=["Identify information relevant to the input question.", "Check for unnecessary or unrelated information.", "Penalize retrievals containing a large amount of irrelevant information.", "Give more credit when relevant information is prioritized.", "Give a higher score when most retrieved context is useful."],
              evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.RETRIEVAL_CONTEXT], threshold=0.75, model=eval_model),
        GEval(name="Answer Relevancy",
              criteria="Determine whether the actual output directly addresses the input question without unnecessary or unrelated information.",
              evaluation_steps=["Check whether the actual output directly answers the input question.", "Check whether the response contains relevant information.", "Penalize unnecessary or off-topic information.", "Do not penalize concise answers that adequately answer the question.", "Give a higher score when the response is focused and directly relevant."],
              evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT], threshold=0.75, model=eval_model),
    ]


def main():
    config = load_config()
    hf_cache = Path(os.environ.get("TEMP", str(ROOT))) / "aurora-hf-cache"
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
    sys.path.insert(0, str(BACKEND))
    from rag_utils import rag

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBED_MODEL", "HarishMaths/Hotel-Policy-Embedding"),
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.load_local(str(BACKEND / "faiss_index"), embeddings, allow_dangerous_deserialization=True)
    frame = pd.read_excel(DATA / "test.xlsx")
    judge = OpenAIModel(model="gpt-4o", api_key=config["OPENAI_API_KEY"], base_url=config["OPENAI_BASE_URL"])
    eval_metrics = metrics(judge)
    options = [
        ("baseline_k5", 5, {"HYBRID_RETRIEVAL": "false", "EXACT_MATCH_BOOST": "false", "HIERARCHICAL_ROUTING": "true", "STRICT_ROUTING": "true", "STRICT_FINAL_FILTER": "false", "CONTEXT_COMPRESSION": "false"}),
        ("hybrid_k5", 5, {"HYBRID_RETRIEVAL": "true", "EXACT_MATCH_BOOST": "false", "HIERARCHICAL_ROUTING": "true", "STRICT_ROUTING": "false", "STRICT_FINAL_FILTER": "false", "CONTEXT_COMPRESSION": "false"}),
        ("hybrid_k6", 6, {"HYBRID_RETRIEVAL": "true", "EXACT_MATCH_BOOST": "false", "HIERARCHICAL_ROUTING": "true", "STRICT_ROUTING": "false", "STRICT_FINAL_FILTER": "false", "CONTEXT_COMPRESSION": "false"}),
    ]
    aggregate_rows, case_rows = [], []
    for name, k, env in options:
        os.environ.update(env)
        cases = []
        print(f"\nRunning {name} (k={k})", flush=True)
        for _, row in frame.iterrows():
            answer, retrieved = rag(query=row["question"], k=k, model_name="gpt-4o-mini", temperature=0.0, top_p=1.0, max_tokens=512, vectorstore=store)
            cases.append(LLMTestCase(input=row["question"], actual_output=answer, expected_output=row["expected_answer"], retrieval_context=retrieved, context=[x.strip() for x in row["relevant_context"].split("<>") if x.strip()]))
        result = evaluate(test_cases=cases, metrics=eval_metrics, async_config=AsyncConfig(run_async=False))
        scores = []
        for idx, (case, test_result) in enumerate(zip(cases, result.test_results), 1):
            values = {m.name.replace(" [GEval]", "").strip().lower(): m.score for m in test_result.metrics_data}
            composite = 0.35 * values["faithfulness"] + 0.25 * values["contextual recall"] + 0.20 * values["contextual precision"] + 0.20 * values["answer relevancy"]
            passed = values["faithfulness"] >= 0.85 and values["contextual recall"] >= 0.80 and values["contextual precision"] >= 0.75 and values["answer relevancy"] >= 0.75
            row_out = {"option": name, "case": idx, "question": case.input, **values, "composite": composite, "passed_all_metrics": passed}
            case_rows.append(row_out); scores.append(row_out)
        aggregate_rows.append({"option": name, "cases": len(scores), "passed_all_metrics": sum(x["passed_all_metrics"] for x in scores), **{metric: sum(x[metric] for x in scores) / len(scores) for metric in ["faithfulness", "contextual recall", "contextual precision", "answer relevancy", "composite"]}})
        print(aggregate_rows[-1], flush=True)
    with (ROOT / "deepeval_options_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys()); writer.writeheader(); writer.writerows(aggregate_rows)
    with (ROOT / "deepeval_options_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=case_rows[0].keys()); writer.writeheader(); writer.writerows(case_rows)


if __name__ == "__main__":
    main()
