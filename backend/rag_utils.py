# Code cell 040

from langchain_openai import ChatOpenAI
import os
import re
import math
from functools import lru_cache
from pathlib import Path
from langchain_core.documents import Document


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=4)
def _get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def _topic_source_terms(query: str):
    query = query.lower()
    topic_terms = {
        "cancellation": ("booking_terms", "cancellation"),
        "cancel": ("booking_terms", "cancellation"),
        "refund": ("refunds", "additional_charges"),
        "charge": ("refunds", "additional_charges"),
        "room": ("room_and_property",),
        "property": ("room_and_property",),
        "breakfast": ("room_and_property",),
        "pet": ("room_and_property", "refunds"),
        "dog": ("room_and_property", "refunds"),
        "parking": ("room_and_property", "refunds"),
        "accessible": ("room_and_property", "guest_conduct"),
        "accessibility": ("guest_conduct", "accessibility"),
        "conduct": ("guest_conduct",),
    }
    return {term for word, terms in topic_terms.items() if word in query for term in terms}


def _property_codes(query: str):
    query = query.lower()
    aliases = {
        "miami brickell": "ASH-MIA",
        "ash-mia": "ASH-MIA",
        "chi loop": "ASH-CHI",
        "ash-chi": "ASH-CHI",
        "atlanta": "ASH-ATL",
        "atl midtown": "ASH-ATL",
        "boston": "ASH-BOS",
        "bos harbor": "ASH-BOS",
        "denver": "ASH-DEN",
        "den union": "ASH-DEN",
        "seattle": "ASH-SEA",
        "sea pike": "ASH-SEA",
    }
    codes = {match.upper() for match in __import__("re").findall(r"ASH-[A-Z]{3}", query.upper())}
    codes.update(code for alias, code in aliases.items() if alias in query)
    return codes


def _topic_matches(query: str, doc) -> bool:
    text = doc.page_content.lower()
    source = Path(doc.metadata.get("source", "")).name.lower()
    terms = _topic_source_terms(query)
    if not terms:
        return True
    metadata_topics = {str(value).lower() for value in doc.metadata.get("topics", [])}
    metadata_section = str(doc.metadata.get("section", "")).lower()
    return (
        bool(metadata_topics.intersection(terms))
        or any(term in source for term in terms)
        or any(term in metadata_section for term in terms)
        or any(word in text for word in terms)
    )


def _property_matches(query: str, doc) -> bool:
    codes = _property_codes(query)
    if not codes:
        return True
    metadata_codes = {str(value).upper() for value in doc.metadata.get("property_codes", [])}
    if metadata_codes:
        return bool(metadata_codes.intersection(codes))
    text = doc.page_content.upper()
    return any(code in text for code in codes)


def _adaptive_k(query: str, requested_k: int) -> int:
    """Use a smaller context for focused factual questions."""
    lower = query.lower()
    complex_terms = (" and ", " or ", "refund", "cancel", "reservation", "policy", "both", "also")
    if _property_codes(query) and not any(term in lower for term in complex_terms):
        return min(requested_k, 3)
    return requested_k


def _bm25_rank(query: str, documents, limit: int):
    """Small dependency-free BM25 implementation for exact-term matching."""
    tokenized = [re.findall(r"[a-z0-9$-]+", doc.page_content.lower()) for doc in documents]
    query_terms = re.findall(r"[a-z0-9$-]+", query.lower())
    n = len(tokenized)
    avgdl = sum(len(tokens) for tokens in tokenized) / max(n, 1)
    doc_freq = {}
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    scored = []
    for index, tokens in enumerate(tokenized):
        counts = {term: tokens.count(term) for term in set(query_terms)}
        score = 0.0
        for term, tf in counts.items():
            if not tf:
                continue
            idf = math.log(1 + (n - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5))
            score += idf * (tf * 2.0) / (tf + 1.5 * (0.5 + 0.5 * len(tokens) / max(avgdl, 1)))
        scored.append((score, index))
    return [documents[index] for _, index in sorted(scored, reverse=True)[:limit]]


def _strict_topic_matches(query: str, doc) -> bool:
    query = query.lower()
    topic_categories = {
        "room_and_property": ("breakfast", "room", "property", "occupancy", "pet", "dog", "parking", "accessible"),
        "booking_terms": ("cancel", "cancellation", "advance purchase", "reservation", "no-show", "check-in"),
        "refunds": ("refund", "charge", "incidentals", "hold", "smoking", "parking", "pet", "dog"),
        "guest_conduct": ("service animal", "accessibility", "quiet", "conduct", "safety"),
    }
    requested = {category for category, words in topic_categories.items() if any(word in query for word in words)}
    if not requested:
        return True
    stored = doc.metadata.get("topics", [])
    if isinstance(stored, str):
        stored = [stored]
    return bool(requested.intersection({str(value) for value in stored})) if stored else _topic_matches(query, doc)


def _section_metadata(doc):
    text = doc.page_content
    headings = re.findall(r"^#{2,4}\s+(.+)$", text, re.MULTILINE)
    properties = sorted(set(re.findall(r"ASH-[A-Z]{3}", text.upper())))
    return {
        **doc.metadata,
        "section": headings[-1].strip() if headings else "unknown",
        "property_codes": ",".join(properties),
    }


def _compress_context(query: str, doc):
    """Keep the section label and the highest-overlap policy sentences."""
    text = doc.page_content
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) <= 6:
        return Document(page_content=text, metadata=_section_metadata(doc))
    query_terms = set(re.findall(r"[a-z0-9$-]+", query.lower()))
    scored = []
    for index, sentence in enumerate(sentences):
        terms = set(re.findall(r"[a-z0-9$-]+", sentence.lower()))
        scored.append((len(query_terms & terms), -index, sentence))
    selected = {item[2] for item in sorted(scored, reverse=True)[:6] if item[0] > 0}
    if not selected:
        selected = set(sentences[:6])
    compressed = " ".join(sentence for sentence in sentences if sentence in selected)
    return Document(page_content=compressed, metadata=_section_metadata(doc))


def _source_boost(query: str, source: str) -> float:
    """Small topic-aware boost; global retrieval remains available."""
    source = Path(source).name.lower()
    return 0.25 if any(term in source for term in _topic_source_terms(query)) else 0.0


def _exact_match_boost(query: str, doc) -> float:
    query_lower = query.lower()
    text_lower = doc.page_content.lower()
    boost = 0.0
    for code in _property_codes(query):
        if code.lower() in text_lower:
            boost += 1.5
    for term in re.findall(r"[a-z0-9]{4,}", query_lower):
        if term in text_lower:
            boost += 0.08
    return boost


def retrieve_documents(
    query: str,
    k: int = 5,
    vectorstore=None,
    lambda_mult: float | None = None,
    fetch_k: int | None = None,
    reranker_model: str | None = None,
    metadata_boost: float | None = None,
):
    """MMR candidate retrieval followed by relevance reranking."""
    if os.getenv("ADAPTIVE_K", "false").lower() == "true":
        k = _adaptive_k(query, k)
    lambda_mult = float(os.getenv("MMR_LAMBDA", "0.5")) if lambda_mult is None else lambda_mult
    fetch_k = int(os.getenv("MMR_FETCH_K", str(max(k * 4, 20)))) if fetch_k is None else fetch_k
    reranker_model = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL) if reranker_model is None else reranker_model
    metadata_boost = float(os.getenv("METADATA_BOOST", "0.25")) if metadata_boost is None else metadata_boost

    dense_candidates = vectorstore.max_marginal_relevance_search(
        query,
        k=max(k * 3, 15),
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    candidates = dense_candidates
    if os.getenv("HYBRID_RETRIEVAL", "false").lower() == "true":
        all_documents = list(vectorstore.docstore._dict.values())
        lexical_candidates = _bm25_rank(query, all_documents, max(k * 3, 15))
        ranked = {}
        for rank, doc in enumerate(dense_candidates, 1):
            ranked.setdefault(doc.page_content, {"doc": doc, "score": 0.0})["score"] += 1 / (60 + rank)
        for rank, doc in enumerate(lexical_candidates, 1):
            ranked.setdefault(doc.page_content, {"doc": doc, "score": 0.0})["score"] += 1 / (60 + rank)
        candidates = [item["doc"] for item in sorted(ranked.values(), key=lambda item: item["score"], reverse=True)]
    # Route property/topic questions into a focused candidate pool, while
    # retaining global candidates as a fallback for ambiguous questions.
    routed_terms = _topic_source_terms(query)
    property_codes = _property_codes(query)
    if os.getenv("HIERARCHICAL_ROUTING", "true").lower() == "true" and (routed_terms or property_codes):
        routed = vectorstore.similarity_search(query, k=max(k * 4, 20))
        focused = [doc for doc in routed if _topic_matches(query, doc) and _property_matches(query, doc)]
        if focused:
            multi_policy = any(word in query.lower() for word in ("cancel", "refund", "advance", "reservation", "rate"))
            strict = os.getenv("STRICT_ROUTING", "true").lower() == "true"
            if strict and property_codes and not multi_policy and len(focused) >= min(k, 3):
                candidates = focused
            else:
                candidates = focused + [
                    doc for doc in candidates
                    if doc.page_content not in {item.page_content for item in focused}
                ]
            if os.getenv("STRICT_FINAL_FILTER", "false").lower() == "true":
                strict_candidates = [
                    doc for doc in candidates
                    if _property_matches(query, doc) and _strict_topic_matches(query, doc)
                ]
                if len(strict_candidates) >= min(2, k):
                    strict_ids = {doc.page_content for doc in strict_candidates}
                    candidates = strict_candidates + [doc for doc in candidates if doc.page_content not in strict_ids]
    reranker = _get_reranker(reranker_model)
    scores = reranker.predict([(query, doc.page_content) for doc in candidates])
    ranked = sorted(
        zip(scores, candidates),
        key=lambda item: item[0]
        + metadata_boost * _source_boost(query, item[1].metadata.get("source", ""))
        + (0.35 if _property_matches(query, item[1]) and _property_codes(query) else 0.0)
        + (_exact_match_boost(query, item[1]) if os.getenv("EXACT_MATCH_BOOST", "false").lower() == "true" else 0.0),
        reverse=True,
    )
    results = [doc for _, doc in ranked[:k]]
    if os.getenv("CONTEXT_COMPRESSION", "false").lower() == "true":
        return [_compress_context(query, doc) for doc in results]
    return [Document(page_content=doc.page_content, metadata=_section_metadata(doc)) for doc in results]

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 5,vectorstore=None) -> list[str]:
    results = retrieve_documents(query=query, k=k, vectorstore=vectorstore)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about Aurora Stays hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations, exceptions, or coverage details.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent benefits, exclusions, limits, waiting periods, conditions, or coverage details.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 5,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    try:
        # Create the generator model inside the guarded block. This allows the
        # local UI to return a useful fallback when API configuration is absent.
        model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )

        # Generate answer using retrieved chunks.
        answer = generate(
            query=query,
            retrieved_chunks=retrieved_chunks,
            model=model
        )
        return answer.content, retrieved_chunks
    except Exception as exc:
        # Keep local endpoint smoke tests available when the model service is offline.
        print(f"RAG generation failed: {type(exc).__name__}: {exc}", flush=True)
        return f"Local smoke-test fallback: model service unavailable ({type(exc).__name__}).", retrieved_chunks
