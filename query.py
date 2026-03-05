import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from collections import defaultdict

load_dotenv()


# ---- Load components ----

def load_components(persist_dir="chroma_db"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("documents")
    print(f"Connected to ChromaDB. Total chunks: {collection.count()}")
    return model, collection


# ---- Retrieval ----

def retrieve(query, model, collection, top_k=5, filter_doc=None):
    """
    Retrieve relevant chunks from ChromaDB.
    
    filter_doc: optionally restrict search to one document.
    None = search across ALL documents simultaneously.
    """
    query_embedding = model.encode([query]).tolist()

    # Build the query
    query_params = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }

    # Optional: filter by specific document
    if filter_doc:
        query_params["where"] = {"doc_id": filter_doc}

    results = collection.query(**query_params)

    # Format results cleanly
    retrieved = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        retrieved.append({
            "text":       results["documents"][0][i],
            "source":     results["metadatas"][0][i]["source"],
            "doc_id":     results["metadatas"][0][i]["doc_id"],
            "page":       results["metadatas"][0][i]["page"],
            "similarity": round(1 / (1 + distance), 4)
        })

    return retrieved


# ---- Generation ----

def generate(query, retrieved_chunks):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Build context with source labels
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += (
            f"\n--- Source {i+1}: "
            f"{chunk['source']} (page {chunk['page']}) ---\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""You are a helpful assistant analyzing documents.
Answer the question using ONLY the context provided below.
When referencing information, mention which source it came from.
If the answer is not in the context, say "I don't have enough 
information in these documents to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ---- Confidence thresholding ----

def filter_by_confidence(retrieved, min_similarity=0.45):
    """
    Remove chunks below confidence threshold.
    
    Why this matters:
    FAISS/ChromaDB always returns top_k results
    even if they're completely irrelevant.
    A question about dinosaurs returns Japan chunks
    with score 0.2 — technically "nearest" but useless.
    
    Threshold prevents generating answers from bad context.
    """
    filtered = [r for r in retrieved if r["similarity"] >= min_similarity]
    return filtered


# ---- Full RAG pipeline ----

def rag(query, model, collection, top_k=5,
        min_similarity=0.45, filter_doc=None):

    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"{'='*60}")

    # Step 1: Retrieve
    retrieved = retrieve(query, model, collection, top_k, filter_doc)

    print(f"\nRetrieved {len(retrieved)} chunks:")
    for r in retrieved:
        print(f"  [{r['similarity']:.4f}] {r['source']} "
              f"p.{r['page']} → {r['text'][:60]}...")

    # Step 2: Filter by confidence
    confident = filter_by_confidence(retrieved, min_similarity)
    print(f"\nAfter confidence filter ({min_similarity}): "
          f"{len(confident)} chunks remain")

    if not confident:
        print("Answer: No confident results found in documents.")
        return None

    # Step 3: Show which documents contributed
    sources_used = defaultdict(int)
    for chunk in confident:
        sources_used[chunk["source"]] += 1

    print(f"\nSources contributing to answer:")
    for source, count in sources_used.items():
        print(f"  {source}: {count} chunks")

    # Step 4: Generate
    print("\nGenerating answer...")
    answer = generate(query, confident)

    print(f"\nAnswer:\n{answer}")
    return answer


if __name__ == "__main__":
    model, collection = load_components()

    # ---- Test 1: Single topic question ----
    rag(
        "What is the role of religion in Japanese culture?",
        model, collection
    )

    # ---- Test 2: Cross-document comparison ----
    rag(
        "How do India and Japan differ in their traditional clothing?",
        model, collection
    )

    # ---- Test 3: Document-specific filter ----
    rag(
        "What is the national language?",
        model, collection,
        filter_doc="france_culture"   # only search France doc
    )

    # ---- Test 4: Out of scope question ----
    rag(
        "What is the speed of light?",
        model, collection
    )