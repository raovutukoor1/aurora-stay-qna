import copy
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "Aurora_Stays_RAG_POC_Local_executed.ipynb"
SUPPLIED_CONFIG_PATH = ROOT / "Supplied files" / "confignewproject4.txt"


def source_for(number: int, source: str) -> str:
    replacements = {
        13: '''# Code cell 013
import re
from copy import deepcopy

raw_docs = [
    LCDocument(page_content=doc.export_to_markdown(), metadata={"source": path})
    for path, doc in docs.items()
]

def annotate_chunk(chunk):
    text = chunk.page_content
    codes = sorted(set(re.findall(r"ASH-[A-Z]{3}", text.upper())))
    headings = re.findall(r"^#{1,4}\\s+(.+)$", text, flags=re.MULTILINE)
    section = headings[-1].strip() if headings else ""
    lower = text.lower()
    topic_map = {
        "room_and_property": ("breakfast", "room", "parking", "pet", "accessible", "property", "occupancy"),
        "booking_terms": ("cancellation", "advance purchase", "reservation", "no-show", "check-in"),
        "refunds": ("refund", "charge", "incidentals", "hold", "smoking", "parking"),
        "guest_conduct": ("service animal", "accessibility", "quiet hours", "conduct", "safety"),
    }
    topics = [topic for topic, terms in topic_map.items() if any(term in lower for term in terms)]
    enriched = deepcopy(chunk)
    enriched.metadata.update({"property_codes": codes, "section": section, "topics": topics})
    return enriched

base_chunks = splitter.split_documents(raw_docs)
atomic_rows = []
for raw in raw_docs:
    if "room_and_property" not in raw.metadata.get("source", ""):
        continue
    heading = "Property-specific matrix"
    for line in raw.page_content.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
        if line.strip().startswith("|") and re.search(r"ASH-[A-Z]{3}", line.upper()):
            atomic_rows.append(LCDocument(
                page_content=f"## {heading}\\n{line}",
                metadata={"source": raw.metadata["source"]},
            ))

''',
        12: '''# Code cell 012
import os

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
    separators=["\\n\\n", "\\n", "##", ". ", " ", ""]
)
''',
        18: '''# Code cell 018
from backend.rag_utils import retrieve_documents as hierarchical_retrieve_documents

def retrieve(query: str, k: int = 5, vectorstore=None) -> list[str]:
    results = hierarchical_retrieve_documents(query=query, k=k, vectorstore=vectorstore)
    return [
        f"Content:\\n{r.page_content}\\n\\nMetadata:\\n{r.metadata}"
        for r in results
    ]
''',
        13: '''# Code cell 013
import re
from copy import deepcopy

raw_docs = [
    LCDocument(page_content=doc.export_to_markdown(), metadata={"source": path})
    for path, doc in docs.items()
]

def annotate_chunk(chunk):
    text = chunk.page_content
    codes = sorted(set(re.findall(r"ASH-[A-Z]{3}", text.upper())))
    headings = re.findall(r"^#{1,4}\\s+(.+)$", text, flags=re.MULTILINE)
    section = headings[-1].strip() if headings else ""
    lower = text.lower()
    topic_map = {
        "room_and_property": ("breakfast", "room", "parking", "pet", "accessible", "property", "occupancy"),
        "booking_terms": ("cancellation", "advance purchase", "reservation", "no-show", "check-in"),
        "refunds": ("refund", "charge", "incidentals", "hold", "smoking", "parking"),
        "guest_conduct": ("service animal", "accessibility", "quiet hours", "conduct", "safety"),
    }
    topics = [topic for topic, terms in topic_map.items() if any(term in lower for term in terms)]
    enriched = deepcopy(chunk)
    enriched.metadata.update({"property_codes": codes, "section": section, "topics": topics})
    return enriched

base_chunks = splitter.split_documents(raw_docs)
atomic_rows = []
for raw in raw_docs:
    if "room_and_property" not in raw.metadata.get("source", ""):
        continue
    heading = "Property-specific matrix"
    for line in raw.page_content.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
        if line.strip().startswith("|") and re.search(r"ASH-[A-Z]{3}", line.upper()):
            property_code = re.search(r"ASH-[A-Z]{3}", line.upper()).group(0)
            atomic_rows.append(LCDocument(
                page_content=f"## {heading}\\nProperty: {property_code}\\n{line}",
                metadata={"source": raw.metadata["source"], "property_codes": [property_code], "topics": ["room_and_property"]},
            ))
if os.getenv("ATOMIC_PROPERTY_ROWS", "true").lower() == "true":
    base_chunks.extend(atomic_rows)
lc_docs = [annotate_chunk(chunk) for chunk in base_chunks]
print(f"Total chunks: {len(lc_docs)} (atomic rows enabled={os.getenv('ATOMIC_PROPERTY_ROWS', 'true')})")
''',
        20: '''# Code cell 020
# Load local API configuration, with the supplied assignment config as a fallback.
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
SUPPLIED_CONFIG_PATH = PROJECT_ROOT / "Supplied files" / "confignewproject4.txt"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)
if not config.get("OPENAI_API_KEY") and SUPPLIED_CONFIG_PATH.exists():
    with open(SUPPLIED_CONFIG_PATH, "r", encoding="utf-8-sig") as f:
        supplied_config = json.load(f)
    config.update({k: v for k, v in supplied_config.items() if v})

os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["OPENAI_BASE_URL"] = config.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
os.environ["LANGCHAIN_TRACING_V2"] = config.get("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = config.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "Aurora Stays QnA K5 MMR Rerank Trial"
os.environ["LANGCHAIN_ENDPOINT"] = "https://apac.api.smith.langchain.com"
os.environ["ADAPTIVE_K"] = "false"
os.environ["HYBRID_RETRIEVAL"] = "false"

print(f"Configuration loaded from {CONFIG_PATH} with supplied fallback")
''',
        21: '''# Code cell 021
# Define the model configuration
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512

llm = ChatOpenAI(
    model=MODEL,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    openai_api_key=config["OPENAI_API_KEY"],
    base_url=config["OPENAI_BASE_URL"]
)
''',
        22: '''# Code cell 022
# Define a simple test query to check whether the LLM is working correctly
TEST_QUERY_LLM = "Reply with exactly: LLM connection successful."
response = llm.invoke(TEST_QUERY_LLM)
print(response.content)
''',
        24: '''# Code cell 024
def rag(
    query: str,
    k: int = 5,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )
    return answer.content, retrieved_chunks
''',
        25: '''# Code cell 025
# Test the RAG pipeline with a sample query
query = "What is the cancellation policy for a flexible reservation?"
response, rel_chunks = rag(query, vectorstore=vectorstore)
''',
        29: '''# Code cell 029
# Verify that the evaluation model is responding correctly
eval_model.generate("Reply with exactly: Evaluation model connection successful.")
''',
        32: '''# Code cell 032
# Define the parameters for test case generation
K = 5
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512

test_cases = test_case_generator(
    df,
    vectorstore,
    k=K,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS
)

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=__import__("deepeval.evaluate.configs", fromlist=["AsyncConfig"]).AsyncConfig(
        run_async=False
    )
)
''',
        35: '''# Code cell 035
# Load the supplied test data workbook
df = pd.read_excel("data/test.xlsx")
''',
        36: '''# Code cell 036
# Use the same retrieval and generation settings for the held-out test set.
K = 5
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512

test_cases = test_case_generator(
    df,
    vectorstore,
    k=K,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS
)

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=__import__("deepeval.evaluate.configs", fromlist=["AsyncConfig"]).AsyncConfig(
        run_async=False
    )
)
''',
    }
    updated = replacements.get(number, source)
    updated = updated.replace(
        "answering questions about medical insurance policies",
        "answering questions about Aurora Stays hotel policies",
    )
    updated = updated.replace(
        "e.g., What is the waiting period for hospitalization?",
        "e.g., What is the cancellation policy for a flexible reservation?",
    )
    return updated


def main() -> None:
    supplied = json.loads(SUPPLIED_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    for key, value in supplied.items():
        if value:
            os.environ[key] = str(value)

    notebook = nbformat.read(NOTEBOOK_PATH.open("r", encoding="utf-8-sig"), as_version=4)
    by_number = {
        int(cell.metadata["code_cell_number"]): (index, cell)
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and "code_cell_number" in cell.metadata
    }

    for number, (_, cell) in by_number.items():
        cell.source = source_for(number, cell.source)

    # Rebuild only the state needed for the unfinished test-evaluation and
    # local-artifact groups, then resume at code cell 035.
    numbers = [
        2, 3, 4, 5, 6, 10, 12, 13, 16, 17, 18,
        20, 21, 23, 24, 28, 30, 31, 33,
        35, 36, 37,
    ]
    mini = nbformat.v4.new_notebook(metadata=copy.deepcopy(notebook.metadata))
    mini.cells = [copy.deepcopy(by_number[number][1]) for number in numbers]

    client = NotebookClient(
        mini,
        timeout=1800,
        allow_errors=False,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )

    with client.setup_kernel():
        for position, number in enumerate(numbers):
            print(f"Executing code cell {number:03d}...", flush=True)
            executed_cell = client.execute_cell(mini.cells[position], position)
            original_index, _ = by_number[number]
            notebook.cells[original_index] = copy.deepcopy(executed_cell)
            notebook.metadata["last_execution_group"] = (
                "Group 4" if number <= 26 else
                "Group 5" if number <= 34 else
                "Group 6" if number <= 37 else
                "Group 7"
            )
            nbformat.write(notebook, NOTEBOOK_PATH.open("w", encoding="utf-8"))
            print(
                f"Completed code cell {number:03d}; "
                f"outputs={len(executed_cell.get('outputs', []))}",
                flush=True,
            )

    notebook.metadata["last_execution_group"] = "Groups 4-7 complete through code cell 047"
    nbformat.write(notebook, NOTEBOOK_PATH.open("w", encoding="utf-8"))
    print("Completed local execution through code cell 047.", flush=True)


if __name__ == "__main__":
    main()
