import json
from pathlib import Path

import truststore
from langsmith import Client

truststore.inject_into_ssl()

root = Path(__file__).resolve().parent
config = json.loads(
    (root / "Supplied files" / "confignew.txt").read_text(encoding="utf-8-sig")
)
api_key = config.get("LANGCHAIN_API_KEY")
if not api_key:
    raise RuntimeError("No LANGCHAIN_API_KEY found in the supplied configuration")

project_name = "Aurora Stays QnA K5 Baseline"
client = Client(
    api_url=config.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    api_key=api_key,
    auto_batch_tracing=False,
)

project = client.create_project(
    project_name,
    description=(
        "Aurora Stays Q&A evaluation project for the K=5 baseline "
        "using chunk_size=512 and chunk_overlap=64."
    ),
    metadata={
        "retrieval_k": 5,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "purpose": "dev and held-out test evaluation",
    },
    upsert=False,
)

print(f"Created LangSmith project: {project.name}")
print(f"Project ID: {project.id}")
