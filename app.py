import os
import chromadb
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import PyPDF2
from pathlib import Path

load_dotenv()


# ---- Core functions ----

def extract_text_with_pages(pdf_file):
    pages = []
    reader = PyPDF2.PdfReader(pdf_file)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append((page_num + 1, text))
    return pages


def split_page_into_chunks(text, page_num, doc_id,
                            source, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "doc_id": doc_id,
                    "page": page_num,
                    "start_char": start
                }
            })
        start += chunk_size - overlap
    return chunks


def ingest_pdf(pdf_file, collection, model, filename):
    doc_id = Path(filename).stem

    # Check if already indexed
    existing = collection.get(
        where={"doc_id": doc_id}, limit=1
    )
    if existing["ids"]:
        return 0, True  # already indexed

    pages = extract_text_with_pages(pdf_file)
    all_chunks = []
    for page_num, page_text in pages:
        chunks = split_page_into_chunks(
            page_text, page_num, doc_id, filename
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0, False

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [
        f"{doc_id}_p{c['metadata']['page']}_c{i}"
        for i, c in enumerate(all_chunks)
    ]

    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = [e.tolist() for e in embeddings]

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )

    return len(all_chunks), False


def retrieve(query, model, collection,
             top_k=5, filter_doc=None, min_similarity=0.50):
    query_embedding = model.encode([query]).tolist()

    params = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }
    if filter_doc and filter_doc != "All documents":
        params["where"] = {"doc_id": filter_doc}

    results = collection.query(**params)

    retrieved = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = round(1 / (1 + distance), 4)
        if similarity >= min_similarity:
            retrieved.append({
                "text":       results["documents"][0][i],
                "source":     results["metadatas"][0][i]["source"],
                "doc_id":     results["metadatas"][0][i]["doc_id"],
                "page":       results["metadatas"][0][i]["page"],
                "similarity": similarity
            })

    return retrieved


def generate_answer(query, retrieved_chunks):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += (
            f"\n--- Source {i+1}: "
            f"{chunk['source']} (page {chunk['page']}) ---\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""You are a helpful assistant analyzing documents.
Answer the question using ONLY the context provided.
When referencing information mention which source it came from.
If the answer is not in the context say "I don't have enough 
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


# ---- Streamlit UI ----

st.set_page_config(
    page_title="Multi-Doc RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-Document RAG System")
st.caption("Project 3 — Query across multiple documents with source attribution")


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )


model = load_model()
collection = load_collection()

# ---- Layout: sidebar + main ----
with st.sidebar:
    st.header("📄 Document Library")

    # Show indexed documents
    total = collection.count()
    if total > 0:
        all_items = collection.get()
        from collections import Counter
        sources = Counter(
            m["source"] for m in all_items["metadatas"]
        )
        for source, count in sources.most_common():
            st.markdown(f"✅ **{source}** — {count} chunks")
    else:
        st.info("No documents indexed yet")

    st.divider()

    # Upload new PDFs
    st.header("➕ Add Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                count, already = ingest_pdf(
                    uploaded_file, collection,
                    model, uploaded_file.name
                )
            if already:
                st.info(f"↩️ {uploaded_file.name} already indexed")
            else:
                st.success(f"✅ {uploaded_file.name} — {count} chunks added")
        st.rerun()

    st.divider()

    # Search settings
    st.header("⚙️ Search Settings")

    # Get available doc_ids for filter
    doc_options = ["All documents"]
    if total > 0:
        doc_ids = list(set(
            m["doc_id"] for m in all_items["metadatas"]
        ))
        doc_options += sorted(doc_ids)

    filter_doc = st.selectbox("Search in:", doc_options)
    top_k = st.slider("Chunks to retrieve", 1, 8, 5)
    min_sim = st.slider("Min confidence", 0.0, 1.0, 0.50)
    show_sources = st.toggle("Show sources", value=True)


# ---- Main chat area ----
if total == 0:
    st.info("👈 Upload PDFs in the sidebar to get started")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            if show_sources and msg["sources"]:
                with st.expander("📚 Sources used"):
                    for chunk in msg["sources"]:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(
                                "Confidence",
                                f"{chunk['similarity']:.2f}"
                            )
                            st.caption(f"{chunk['source']}")
                            st.caption(f"Page {chunk['page']}")
                        with col2:
                            st.caption(chunk["text"][:300] + "...")
                        st.divider()

# Chat input
query = st.chat_input("Ask anything across your documents...")

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("assistant"):
        filter_param = (
            None if filter_doc == "All documents"
            else filter_doc
        )

        with st.spinner("Searching documents..."):
            retrieved = retrieve(
                query, model, collection,
                top_k, filter_param, min_sim
            )

        if not retrieved:
            answer = ("I couldn't find confident matches in "
                     "the documents. Try lowering the minimum "
                     "confidence or rephrasing your question.")
            st.warning(answer)
        else:
            # Show which docs are being searched
            sources_hit = defaultdict(int)
            for r in retrieved:
                sources_hit[r["source"]] += 1

            source_summary = " · ".join(
                f"{src} ({n})" for src, n in sources_hit.items()
            )
            st.caption(f"🔍 Searching: {source_summary}")

            with st.spinner("Generating answer..."):
                answer = generate_answer(query, retrieved)

            st.write(answer)

            if show_sources:
                with st.expander("📚 Sources used"):
                    for chunk in retrieved:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(
                                "Confidence",
                                f"{chunk['similarity']:.2f}"
                            )
                            st.caption(f"{chunk['source']}")
                            st.caption(f"Page {chunk['page']}")
                        with col2:
                            st.caption(
                                chunk["text"][:300] + "..."
                            )
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved
    })