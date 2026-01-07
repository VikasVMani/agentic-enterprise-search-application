import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# Config

EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
COLLECTION_NAME = "enterprise_docs"


# Init

client = chromadb.Client()
collection = client.get_or_create_collection(COLLECTION_NAME)

embedder = SentenceTransformer(EMBED_MODEL)

# BM25 indexes per partition
bm25_indexes = {}      # partition -> BM25
bm25_corpus = {}       # partition -> list of tokenized docs
bm25_ids = {}          # partition -> list of ids



# Ingestion

def ingest_documents(chunks):
    """
    chunks: list of dicts
    {
      id,
      document_name,
      page_no,
      text,
      partition
    }
    """
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "document_name": c["document_name"],
                "page_no": c["page_no"],
                "partition": c["partition"]
            }
            for c in chunks
        ]
    )

    # ---- Build BM25 per partition ----
    for c in chunks:
        p = c["partition"]

        bm25_corpus.setdefault(p, [])
        bm25_ids.setdefault(p, [])

        bm25_corpus[p].append(c["text"].lower().split())
        bm25_ids[p].append(c["id"])

    for p in bm25_corpus:
        bm25_indexes[p] = BM25Okapi(bm25_corpus[p])



# Hybrid Search (Partition-Aware)

def hybrid_search(query, partition, top_k=5, alpha=0.6):
    """
    Search within a specific logical partition.
    """
    # collection = client.get_or_create_collection(COLLECTION_NAME)

    # ---- Semantic search ----
    query_emb = embedder.encode([query], normalize_embeddings=True)
    semantic = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=top_k,
        where={"partition": partition}
    )

    semantic_scores = {}
    for i, doc_id in enumerate(semantic["ids"][0]):
        semantic_scores[doc_id] = 1 - semantic["distances"][0][i]

    # ---- Keyword search ----
    bm25 = bm25_indexes.get(partition)
    if not bm25:
        return []

    scores = bm25.get_scores(query.lower().split())
    keyword_scores = {
        bm25_ids[partition][i]: scores[i]
        for i in range(len(scores))
    }

    # ---- Combine ----
    combined = {}
    for doc_id in set(semantic_scores) | set(keyword_scores):
        combined[doc_id] = (
            alpha * semantic_scores.get(doc_id, 0) +
            (1 - alpha) * keyword_scores.get(doc_id, 0)
        )

    ranked_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]

    results = collection.get(ids=ranked_ids)

    final = []
    for i in range(len(results["ids"])):
        final.append({
            "id": results["ids"][i],
            "text": results["documents"][i],
            "document_name": results["metadatas"][i]["document_name"],
            "page_no": results["metadatas"][i]["page_no"],
            "partition": results["metadatas"][i]["partition"],
            "score": combined[results["ids"][i]]
        })

    return final
