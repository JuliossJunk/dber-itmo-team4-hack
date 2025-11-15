import chromadb
from datetime import datetime
import hashlib
import os

host = os.getenv("CHROMA_HOST", "localhost")
port = int(os.getenv("CHROMA_PORT", "8000"))

client = chromadb.HttpClient(host=host, port=port)

cache_collection = client.get_or_create_collection(
    name="verified_cache"
)


def make_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cache_get(query: str):
    h = make_hash(query)
    result = cache_collection.get(ids=[h])

    if not result["documents"]:
        return None

    return {
        "facts": result["documents"][0],
        "source": result["metadatas"][0]["source"],
        "date": result["metadatas"][0]["date"]
    }


def cache_set(query: str, facts: str, source: str):
    h = make_hash(query)
    cache_collection.upsert(
        ids=[h],
        documents=[facts],
        metadatas=[{
            "source": source,
            "date": datetime.utcnow().isoformat()
        }]
    )


def cache_search(query: str, n_results=3):
    results = cache_collection.query(
        query_texts=[query],
        n_results=n_results
    )

    if not results["documents"]:
        return []

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append({
            "facts": doc,
            "source": meta["source"],
            "date": meta["date"]
        })

    return output
