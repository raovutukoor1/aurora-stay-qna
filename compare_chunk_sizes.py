import os
import re
import sys
from pathlib import Path

import pandas as pd
import truststore

truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parent
TEMP_CACHE = Path(os.environ.get("TEMP", str(ROOT))) / "aurora-hf-cache"
os.environ.setdefault("HF_HOME", str(TEMP_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(TEMP_CACHE / "hub"))

from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

sys.stdout.reconfigure(encoding="utf-8")

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
PDFS = sorted((ROOT / "data").glob("*.pdf"))
converter = DocumentConverter()
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

docs = []
for pdf in PDFS:
    converted = converter.convert(str(pdf)).document
    docs.append(Document(page_content=converted.export_to_markdown(), metadata={"source": str(pdf)}))

dev = pd.read_csv(ROOT / "data" / "evaluation.csv")
test = pd.read_excel(ROOT / "data" / "test.xlsx")
stop = set("the a an and or is are to of for at in on what how can may be with from this that it as by do does will not only their its into after before one two".split())


def words(value):
    return {x for x in re.findall(r"[a-z0-9$]+", str(value).lower()) if len(x) > 2 and x not in stop}


def score(frame, store, k=5):
    coverages = []
    high = 0
    evidence = 0
    for _, row in frame.iterrows():
        retrieved = store.similarity_search(row["question"], k=k)
        text = " ".join(doc.page_content for doc in retrieved)
        expected = words(row["expected_answer"])
        coverage = len(expected & words(text)) / max(1, len(expected))
        coverages.append(coverage)
        high += coverage >= 0.75
        important = re.findall(
            r"ASH-[A-Z]+|\$\d+|\d:\d+\s*(?:AM|PM)|\d+\s*PM|\d+\s*AM",
            row["expected_answer"],
            re.I,
        )
        evidence += bool(important) and all(token.lower() in text.lower() for token in important)
    return sum(coverages) / len(coverages), high, evidence


print("dataset | chunk_size | overlap | chunks | avg coverage | >=75% | important evidence")
for chunk_size, overlap in [(400, 64), (450, 80), (512, 64)]:
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "##", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    for name, frame in [("dev", dev), ("test", test)]:
        coverage, high, evidence = score(frame, store)
        print(f"{name:>4} | {chunk_size:>10} | {overlap:>7} | {len(chunks):>6} | {coverage:.3f}        | {high}/10  | {evidence}/10")
