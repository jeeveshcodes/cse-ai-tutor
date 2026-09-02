"""
Diagnostic utility to inspect ChromaDB vector storage and SQLite short-term conversation memory.
"""

import os
from collections import Counter
from dotenv import load_dotenv
import chroma_compat  # Compatibility shim for gRPC telemetry
import chromadb
from chromadb.utils import embedding_functions

from memory_db import get_memory_stats, get_recent_conversation_history

load_dotenv()


def check_vector_db():
    print("=" * 60)
    print(" 🔍 ChromaDB Vector Storage Status")
    print("=" * 60)
    
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        print(f"[!] ChromaDB directory '{db_path}' not found. Run 'python ingest.py' first.\n")
        return 0

    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(
            name="dsa_notes",
            embedding_function=embedding_fn
        )

        all_data = collection.get()
        metadatas = all_data.get("metadatas", [])
        
        if not metadatas:
            print("[!] Collection 'dsa_notes' exists but is empty. Run 'python ingest.py' to populate it.\n")
            return 0

        sources = [meta.get("source", "Unknown") for meta in metadatas if meta]
        counts = Counter(sources)

        print(f"✓ Total chunks indexed: {len(sources)}")
        print(f"✓ Total source PDF files: {len(counts)}\n")
        print("Chunks per source file:")
        for source, count in sorted(counts.items()):
            print(f"  • {count:3d} chunks — {source}")
        
        # Show a sample chunk preview
        docs = all_data.get("documents", [])
        if docs:
            print("\nSample chunk snippet:")
            snippet = docs[0][:180].replace("\n", " ")
            print(f"  \"{snippet}...\"\n")
            
        return len(sources)
    except Exception as e:
        print(f"[!] Error inspecting ChromaDB: {e}\n")
        return 0


def check_memory_database():
    print("=" * 60)
    print(" 🧠 SQLite Short-Term Memory Status (6-Hour Window)")
    print("=" * 60)
    try:
        stats = get_memory_stats(hours=6)
        history = get_recent_conversation_history(hours=6)
        print(f"✓ Active messages in 6-hr window: {stats['count']}")
        print(f"✓ Last activity timestamp: {stats['last_active'] or 'None'}")
        
        if history:
            print("\nRecent conversation turns:")
            for idx, msg in enumerate(history[-4:], 1):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")[:80].replace("\n", " ")
                print(f"  [{idx}] {role}: {content}...")
        else:
            print("  (No active conversations currently recorded)")
        print()
    except Exception as e:
        print(f"[!] Error checking SQLite memory: {e}\n")


def check_environment():
    print("=" * 60)
    print(" ⚙️ Environment Configuration")
    print("=" * 60)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"✓ GEMINI_API_KEY: Configured ({masked})")
    else:
        print("[!] GEMINI_API_KEY: Not found in environment or .env file.")
    print("=" * 60)


if __name__ == "__main__":
    check_environment()
    print()
    check_vector_db()
    check_memory_database()