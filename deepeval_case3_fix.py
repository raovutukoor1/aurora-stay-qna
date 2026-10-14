"""Validate the targeted Double Queen retrieval fix on held-out case 3."""

import os
import sys
from pathlib import Path

import pandas as pd
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
from rag_utils import rag
from deepeval_options_experiment import load_config, metrics
from deepeval.models import OpenAIModel

config = load_config()
hf_cache = Path(os.environ.get("TEMP", str(ROOT))) / "aurora-hf-cache"
os.environ.setdefault("HF_HOME", str(hf_cache))
os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
os.environ.update({
    "HYBRID_RETRIEVAL": "false",
    "EXACT_MATCH_BOOST": "false",
    "HIERARCHICAL_ROUTING": "true",
    "STRICT_ROUTING": "true",
    "STRICT_FINAL_FILTER": "false",
    "CONTEXT_COMPRESSION": "false",
})

embeddings = HuggingFaceEmbeddings(
    model_name="HarishMaths/Hotel-Policy-Embedding",
    model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True},
)
store = FAISS.load_local(str(ROOT / "backend" / "faiss_index"), embeddings, allow_dangerous_deserialization=True)
row = pd.read_excel(ROOT / "data" / "test.xlsx").iloc[2]
answer, retrieved = rag(row["question"], k=5, model_name="gpt-4o-mini", temperature=0.0, top_p=1.0, max_tokens=512, vectorstore=store)
case = LLMTestCase(input=row["question"], actual_output=answer, expected_output=row["expected_answer"], retrieval_context=retrieved, context=[x.strip() for x in row["relevant_context"].split("<>") if x.strip()])
judge = OpenAIModel(model="gpt-4o", api_key=config["OPENAI_API_KEY"], base_url=config["OPENAI_BASE_URL"])
result = evaluate(test_cases=[case], metrics=metrics(judge), async_config=AsyncConfig(run_async=False))
scores = {m.name.replace(" [GEval]", "").strip(): m.score for m in result.test_results[0].metrics_data}
print({"case": 3, **scores})
print("Retrieved chunks:", len(retrieved))
