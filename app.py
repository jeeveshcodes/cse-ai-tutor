import os
from typing import Generator, List, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
import chroma_compat  # Compatibility shim for gRPC telemetry
import chromadb
from chromadb.utils import embedding_functions

from memory_db import (
    clear_chat_history,
    format_history_for_prompt,
    get_memory_stats,
    get_recent_conversation_history,
    log_conversation,
)

# Load environment variables
load_dotenv()

# ---- Page Configuration ----
st.set_page_config(
    page_title="Luna — CSE AI Tutor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS for Modern UI ----
st.markdown(
    """
    <style>
    /* Global Font and Layout Styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 960px;
    }

    /* Header Styling */
    .luna-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
    }
    
    .luna-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .luna-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-top: -4px;
        margin-bottom: 20px;
    }

    /* Status Badges */
    .status-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.85rem;
        padding: 4px 0;
    }

    .status-badge {
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
    }
    
    .status-badge-ok {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-badge-warn {
        background-color: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    .status-badge-error {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Quick Action Chips */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 20px;
    }

    /* Sidebar Divider & Cards */
    .sidebar-topic-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }

    /* Streamlit Button Overrides */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- System Instruction ----
SYSTEM_INSTRUCTION = (
    "You are Luna, an expert and supportive CSE tutor for a college student preparing for exams, "
    "coding assessments, and technical placements.\n\n"
    "You have access to two contextual knowledge layers:\n"
    "1. RECENT CONVERSATION HISTORY: Past turns in the current 6-hour window. Use this to track multi-turn context, "
    "resolve pronouns (e.g. 'explain its space complexity'), and maintain continuity.\n"
    "2. RETRIEVED COURSE NOTES: Excerpts from the student's indexed DSA and OOP notes.\n\n"
    "CORE TUTORING RULES:\n"
    "- Ground your explanation primarily in the retrieved course notes when relevant.\n"
    "- If the notes do not contain the answer, explicitly mention that and answer accurately from your broad Computer Science knowledge.\n"
    "- STRICT FORMATTING & BREVITY RULE: Keep standard answers under 120 words. Format every response as 3 to 6 crisp bullet points, "
    "each bullet being a single clear sentence — not a bulky paragraph, no excessive headings, and no code blocks unless explicitly requested.\n"
    "- If the student specifically says 'explain in detail', 'go deep', 'give an example', or 'show code', you may expand beyond the bullet format with thorough code and commentary.\n"
    "- IMPORTANT NOTATION RULE: Never output raw LaTeX or delimiters like \\(O(n)\\) or $O(n)$. Always write time/space complexity in plain text like O(n log n), O(n^2), or O(1).\n"
    "- Keep your tone encouraging, pedagogically sharp, and exam-focused."
)

# ---- Resource Loading ----
@st.cache_resource
def load_resources():
    """Loads Google GenAI client and ChromaDB vector collection."""
    api_key = os.getenv("GEMINI_API_KEY")
    client = None
    if api_key:
        try:
            client = genai.Client()
        except Exception as e:
            print(f"[Luna Init] GenAI client init error: {e}")

    collection = None
    collection_count = 0
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(
            name="dsa_notes",
            embedding_function=embedding_fn
        )
        collection_count = collection.count()
    except Exception as e:
        print(f"[Luna Init] ChromaDB init error: {e}")

    return client, collection, collection_count


client, collection, collection_count = load_resources()


# ---- Helper Functions ----
def retrieve_context(query_text: str, n_results: int = 5) -> str:
    """Retrieves the top-k most relevant text chunks from ChromaDB."""
    if collection is None:
        return "No note vector database is available."
    try:
        count = collection.count()
        if count == 0:
            return "Note knowledge base is empty. Run 'python ingest.py' to populate it."
        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count)
        )
        docs = results.get("documents", [[]])
        if docs and docs[0]:
            return "\n\n---\n\n".join(docs[0])
        return "No relevant note context found."
    except Exception as e:
        return f"Error retrieving context: {e}"


def get_answer_stream(question: str, history_list: List[dict]) -> Generator[str, None, None]:
    """Generates a streaming response from Gemini fusing RAG notes and short-term memory."""
    if not client:
        yield "⚠️ **Gemini API Key Missing**: Please configure `GEMINI_API_KEY` in your `.env` file or environment variables to enable Luna."
        return

    # 1. Retrieve RAG Context
    rag_context = retrieve_context(question)

    # 2. Format past conversation history (last 6 turns within 6-hour window)
    history_str = format_history_for_prompt(history_list, max_turns=6)

    # 3. Construct unified prompt
    prompt_payload = (
        f"### RECENT CONVERSATION HISTORY (Last 6 Hours):\n{history_str}\n\n"
        f"### RETRIEVED COURSE NOTES (RAG Context):\n{rag_context}\n\n"
        f"### STUDENT QUESTION:\n{question}"
    )

    try:
        # Stream using Google GenAI SDK
        stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt_payload,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        # Fallback to interactions API or alternative model if applicable
        try:
            interaction_stream = client.interactions.create(
                model="gemini-3.5-flash",
                system_instruction=SYSTEM_INSTRUCTION,
                input=prompt_payload,
                stream=True,
            )
            for event in interaction_stream:
                if event.event_type == "step.delta" and event.delta:
                    if getattr(event.delta, "type", None) == "text" and getattr(event.delta, "text", None):
                        yield event.delta.text
        except Exception as fallback_error:
            yield f"⚠️ **Error generating response**: {fallback_error or e}"


# ---- Initialize Session State with Persistent SQLite Memory ----
if "messages" not in st.session_state:
    # Pre-populate session state from SQLite 6-hour short-term memory
    saved_history = get_recent_conversation_history(hours=6)
    st.session_state.messages = saved_history if saved_history else []

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

# ---- Sidebar Layout ----
with st.sidebar:
    st.markdown("## 🌙 Luna Tutor")
    st.caption("Personal AI CSE Placement & Exam Mentor")
    st.markdown("---")

    # System Status Card
    st.markdown("#### ⚡ System Status")
    
    # API Status
    has_api = bool(os.getenv("GEMINI_API_KEY"))
    api_badge = '<span class="status-badge status-badge-ok">Connected</span>' if has_api else '<span class="status-badge status-badge-error">Missing Key</span>'
    
    # ChromaDB Status
    total_chunks = collection.count() if collection else 0
    if total_chunks > 0:
        db_badge = f'<span class="status-badge status-badge-ok">{total_chunks} Chunks</span>'
    else:
        db_badge = '<span class="status-badge status-badge-warn">0 Chunks</span>'
    
    # Active Memory Stats
    mem_stats = get_memory_stats(hours=6)
    mem_count = mem_stats.get("count", 0)
    mem_badge = f'<span class="status-badge status-badge-ok">{mem_count} Active (6h)</span>'

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-item">
                <span>Gemini API</span>
                {api_badge}
            </div>
            <div class="status-item">
                <span>ChromaDB Vector RAG</span>
                {db_badge}
            </div>
            <div class="status-item">
                <span>SQLite Memory Window</span>
                {mem_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_api:
        st.warning("⚠️ Set `GEMINI_API_KEY` in `.env` to start asking questions.")

    if total_chunks == 0:
        st.info("💡 Run `python ingest.py` in your terminal to index your PDF course notes.")

    st.markdown("---")

    # Loaded Knowledge Base Topics
    st.markdown("#### 📚 Loaded Knowledge Topics")
    with st.expander("📦 Data Structures & Algorithms", expanded=True):
        st.markdown(
            """
            - Arrays & Dynamic Strings
            - Linked Lists (Singly / Doubly / Circular)
            - Stacks, Queues & Deques
            - Trees (Binary, BST, AVL, Traversals)
            - Heaps & Priority Queues
            - Graphs (BFS, DFS, Dijkstra, TopoSort)
            - Sorting & Searching Algorithms
            - Dynamic Programming & Recursion
            - Greedy Strategies & Bit Manipulation
            - Asymptotic Time & Space Complexity
            """
        )

    with st.expander("🧱 OOP & Software Design", expanded=False):
        st.markdown(
            """
            - Encapsulation, Abstraction, Inheritance, Polymorphism
            - Classes, Objects, Constructors & Destructors
            - SOLID Design Principles
            - Abstract Classes vs Interfaces
            """
        )

    st.markdown("---")

    # Clear Chat / Reset Action
    if st.button("🗑️ Reset Chat & Memory", use_container_width=True):
        clear_chat_history()
        st.session_state.messages = []
        st.session_state.quick_prompt = None
        st.toast("Conversation memory cleared successfully!", icon="🧹")
        st.rerun()

    st.markdown(
        """
        <div style="font-size:0.75rem; color:#64748b; text-align:center; margin-top:20px;">
            Dual Memory: SQLite 6h Window + ChromaDB RAG<br>
            Powered by Google Gemini
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---- Main Content Area ----
st.markdown(
    """
    <div class="luna-header">
        <h1 class="luna-title">🌙 Luna — CSE AI Tutor</h1>
    </div>
    <div class="luna-subtitle">
        Intelligent, concise exam & placement tutoring grounded in your verified DSA & OOP course notes.
    </div>
    """,
    unsafe_allow_html=True,
)

# Starter Suggestion Pills (if no messages yet)
if len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Questions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏱️ QuickSort time & space complexity", use_container_width=True):
            st.session_state.quick_prompt = "What is the average and worst-case time complexity of QuickSort, and why?"
            st.rerun()
        if st.button("🧱 Explain SOLID principles in OOP", use_container_width=True):
            st.session_state.quick_prompt = "Explain the 5 SOLID design principles with concise real-world examples."
            st.rerun()
    with col2:
        if st.button("🔄 Difference between Stack and Queue", use_container_width=True):
            st.session_state.quick_prompt = "What are the core differences and use cases between a Stack and a Queue?"
            st.rerun()
        if st.button("⚡ Dynamic Programming vs Greedy Method", use_container_width=True):
            st.session_state.quick_prompt = "When should I use Dynamic Programming over a Greedy strategy in algorithms?"
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    avatar = "👤" if role == "user" else "🌙"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

# Handle Input (via Quick Prompt or Chat Input)
chat_input_val = st.chat_input("Ask Luna a question about DSA, OOP, or placements...")
active_prompt = chat_input_val or st.session_state.quick_prompt

if active_prompt:
    # Reset quick prompt state
    st.session_state.quick_prompt = None

    # Append user prompt to session state
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_prompt)

    # Stream assistant response
    with st.chat_message("assistant", avatar="🌙"):
        # We pass the conversation history prior to the current question
        history_for_context = st.session_state.messages[:-1]
        response_generator = get_answer_stream(active_prompt, history_for_context)
        full_response = st.write_stream(response_generator)

    # Append and persist response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    log_conversation(active_prompt, full_response)