"""
Ingestion script for Luna — CSE AI Tutor.
Extracts text from PDF notes in the /knowledge directory, breaks them into semantic chunks,
generates embeddings, and stores them in ChromaDB.
"""

import os
import chroma_compat  # Compatibility shim for gRPC telemetry
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# ---- Step 1: Set up the embedding function and database ----
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ---- Wipe any existing collection so we always rebuild fresh ----
try:
    chroma_client.delete_collection(name="dsa_notes")
    print("Cleared old database collection.\n")
except Exception:
    print("No existing database found, starting fresh.\n")

collection = chroma_client.get_or_create_collection(
    name="dsa_notes",
    embedding_function=embedding_fn
)

# ---- Step 2: Find every PDF inside the knowledge folder (and subfolders) ----
def find_pdfs(root_folder):
    pdf_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(dirpath, filename))
    return pdf_paths

# ---- Step 3: Extract text from a PDF ----
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ---- Step 4: Break long text into overlapping chunks ----
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---- Step 5: Process every PDF and store chunks in the database ----
pdf_files = find_pdfs("knowledge")
print(f"Found {len(pdf_files)} PDF files.\n")

chunk_id = 0
for pdf_path in pdf_files:
    print(f"Processing: {pdf_path}")
    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    for chunk in chunks:
        if chunk.strip():
            collection.add(
                documents=[chunk],
                ids=[f"chunk_{chunk_id}"],
                metadatas=[{"source": pdf_path}]
            )
            chunk_id += 1

print(f"\nDone. Stored {chunk_id} chunks in the database.")