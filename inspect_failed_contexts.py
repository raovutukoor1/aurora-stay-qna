import os
import sys
from pathlib import Path

os.environ.update({
    "HYBRID_RETRIEVAL": "false",
    "EXACT_MATCH_BOOST": "false",
    "HIERARCHICAL_ROUTING": "true",
    "STRICT_ROUTING": "true",
    "STRICT_FINAL_FILTER": "false",
    "CONTEXT_COMPRESSION": "false",
})
hf_cache = Path(os.environ.get("TEMP", ".")) / "aurora-hf-cache"
os.environ.setdefault("HF_HOME", str(hf_cache))
os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, "backend")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_utils import retrieve_documents

embeddings = HuggingFaceEmbeddings(
    model_name="HarishMaths/Hotel-Policy-Embedding",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
store = FAISS.load_local("backend/faiss_index", embeddings, allow_dangerous_deserialization=True)

for query in [
    "How many people can stay in a Double Queen room?",
    "What exactly is included in an Accessible King room?",
]:
    print(f"\n### {query}")
    for index, doc in enumerate(retrieve_documents(query, k=5, vectorstore=store), 1):
        source = Path(doc.metadata.get("source", "")).name
        print(f"--- {index} source={source} section={doc.metadata.get('section')} codes={doc.metadata.get('property_codes')}")
        print(doc.page_content[:1200].replace("\n", " "))
