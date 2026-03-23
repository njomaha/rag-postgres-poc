import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME
from db import get_schema_as_text

def build_schema_index():
    """Run this once to embed your schema into ChromaDB."""
    print("Reading schema from Postgres...")
    schema_docs = get_schema_as_text()

    print(f"Found {len(schema_docs)} tables. Embedding...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete and recreate collection (fresh index)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    for doc in schema_docs:
        embedding = model.encode(doc["text"]).tolist()
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["text"]],
            metadatas=[{"table": doc["table"]}]
        )
        print(f"  ✅ Embedded: {doc['table']}")

    print("\nSchema index built successfully!")


def search_schema(question: str, top_k: int = 4) -> list[str]:
    """Find the most relevant tables for a given question."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    return results["documents"][0]   # list of matching schema strings


if __name__ == "__main__":
    build_schema_index()