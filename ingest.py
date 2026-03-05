import os
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


# ---- PDF Processing ----

def extract_text_with_pages(pdf_path):
    """
    Extract text page by page.
    Returns list of (page_num, text) tuples.
    Unlike Project 2, we track pages individually
    so metadata can include page numbers.
    """
    pages = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():  # skip empty pages
                pages.append((page_num + 1, text))
    return pages


def split_page_into_chunks(text, page_num, doc_id, source,
                            chunk_size=800, overlap=100):
    """
    Split a single page's text into chunks.
    Each chunk gets metadata about its origin.
    
    Why 800 instead of 1000?
    Smaller chunks = more precise retrieval.
    With metadata overhead we keep total size manageable.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:  # skip empty chunks
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source,       # filename
                    "doc_id": doc_id,       # clean name
                    "page": page_num,       # page number
                    "start_char": start,    # position in page
                }
            })
        start += chunk_size - overlap

    return chunks


def process_pdf(pdf_path):
    """Process a single PDF into chunks with metadata."""
    source = Path(pdf_path).name          # "japan_culture.pdf"
    doc_id = Path(pdf_path).stem          # "japan_culture"

    print(f"\nProcessing: {source}")

    pages = extract_text_with_pages(pdf_path)
    print(f"  Pages extracted: {len(pages)}")

    all_chunks = []
    for page_num, page_text in pages:
        chunks = split_page_into_chunks(
            page_text, page_num, doc_id, source
        )
        all_chunks.extend(chunks)

    print(f"  Chunks created: {len(all_chunks)}")
    return all_chunks


# ---- ChromaDB Storage ----

def get_or_create_collection(persist_dir="chroma_db"):
    """
    Get existing ChromaDB collection or create new one.
    
    persist_dir: where ChromaDB saves data on disk.
    collection: like a table in a database — holds all our chunks.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )
    return collection


def ingest_chunks(collection, chunks, model):
    """
    Add chunks to ChromaDB.
    ChromaDB needs: ids, embeddings, documents, metadatas
    """
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Generate unique IDs
    # Format: doc_id_page_chunknum
    ids = [
        f"{c['metadata']['doc_id']}_p{c['metadata']['page']}_c{i}"
        for i, c in enumerate(chunks)
    ]

    print(f"  Embedding {len(chunks)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = [e.tolist() for e in embeddings]

    # Add to ChromaDB in batches of 100
    # ChromaDB can be slow with very large single inserts
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )

    print(f"  Stored in ChromaDB successfully")


def check_already_indexed(collection, doc_id):
    """
    Check if a document is already in the collection.
    This prevents re-indexing the same PDF twice.
    This is called incremental indexing.
    """
    results = collection.get(
        where={"doc_id": doc_id},
        limit=1
    )
    return len(results["ids"]) > 0


# ---- Main ingestion pipeline ----

if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    collection = get_or_create_collection()

    # Find all PDFs in current directory
    pdf_files = list(Path(".").glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF files: "
          f"{[p.name for p in pdf_files]}")

    # Process each PDF
    total_chunks = 0
    for pdf_path in pdf_files:
        doc_id = pdf_path.stem

        # Skip if already indexed
        if check_already_indexed(collection, doc_id):
            print(f"\nSkipping {pdf_path.name} — already indexed")
            continue

        chunks = process_pdf(str(pdf_path))
        ingest_chunks(collection, chunks, model)
        total_chunks += len(chunks)

    # Summary
    total_in_db = collection.count()
    print(f"\n=== INGESTION COMPLETE ===")
    print(f"New chunks added:     {total_chunks}")
    print(f"Total chunks in DB:   {total_in_db}")

    # Show what's in the database
    print(f"\n=== DATABASE CONTENTS ===")
    all_items = collection.get()
    
    from collections import Counter
    sources = Counter(m["source"] for m in all_items["metadatas"])
    for source, count in sources.most_common():
        print(f"  {source}: {count} chunks")