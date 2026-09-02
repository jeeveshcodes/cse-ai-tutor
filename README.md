# 🌙 Luna — CSE AI Tutor

A high-performance, RAG-powered (Retrieval-Augmented Generation) AI tutor designed specifically for Computer Science students preparing for college examinations, coding assessments, and technical placements. Luna combines semantic document retrieval with a rolling SQLite conversational memory to deliver concise, technically rigorous, and context-aware responses.

**Live demo:** [cse-ai-tutor-wnnptpyzkym3hm6eiavuaf.streamlit.app](https://cse-ai-tutor-wnnptpyzkym3hm6eiavuaf.streamlit.app)

---

## 🚀 Key Features

- **Dual-Layer Context Fusion:** Seamlessly fuses long-term course note retrieval (via ChromaDB vector search) with short-term conversational context (via SQLite with a 6-hour sliding window).
- **Exam-Oriented Pedagogy:** System instructions tuned for maximum retention — responses are delivered in concise, high-yield bullet points with plain-text complexity notations.
- **Modernized Streamlit Interface:** Sleek dark-mode aesthetic featuring live streaming responses, real-time system status indicators (API connectivity, vector collection count, active memory), topic drawers, and quick-start prompt chips.
- **Multi-Turn Continuity:** Remembers past discussion context so students can ask seamless follow-up questions (e.g., *"What is its space complexity?"* or *"Can you provide an example in Python?"*).
- **Graceful Error Handling:** Automated fallbacks for missing environment variables, unindexed vector databases, or API communication timeouts.

---

## 🛠️ Tech Stack

- **Large Language Model:** Google Gemini (`gemini-2.5-flash` / `gemini-3.5-flash`) via the Google GenAI SDK
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database (Long-Term Notes):** ChromaDB
- **Short-Term Memory (Conversational History):** SQLite3 with 6-hour sliding window eviction
- **PDF Extraction & Chunking:** `pypdf`
- **Web Frontend:** Streamlit
- **Environment Management:** `python-dotenv`

---

## 🧠 System Architecture

```text
               ┌───────────────────────────────┐
               │    Course Notes (PDFs in      │
               │         /knowledge)           │
               └───────────────┬───────────────┘
                               │ (pypdf & chunking)
                               ▼
               ┌───────────────────────────────┐
               │ Sentence Transformers Embeddings│
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │ ChromaDB Vector Storage (RAG) │
               └───────────────┬───────────────┘
                               │
Student Question ──────────────┼────────────────────────┐
       │                       │ (Top-k Chunks)         │
       ▼                       ▼                        ▼
┌──────────────┐      ┌─────────────────┐      ┌────────────────┐
│ Streamlit UI │ ───► │  Context Fusion │ ───► │  Google Gemini │ ──► Live Streamed Answer
└──────────────┘      │     Engine      │      │     Model      │
       │              └────────┬────────┘      └────────────────┘
       │                       ▲                        │
       │                       │ (Last 6h History)      │
       ▼                       │                        ▼
┌───────────────────────────────────────────────────────────────┐
│              SQLite Short-Term Memory (memory_db.py)          │
└───────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
├── app.py              # Streamlit web application with modern UI & live streaming
├── chatbot.py          # Terminal CLI interactive tutor with dual-memory integration
├── memory_db.py        # SQLite short-term conversation storage (6-hour window)
├── ingest.py           # Ingestion script to chunk PDFs and build ChromaDB vector store
├── check_db.py         # Diagnostic utility to inspect vector database & memory stats
├── knowledge/          # Curated PDF course notes (DSA, OOP, SOLID principles)
│   ├── DSA/            # Data structures notes (Arrays, Trees, Graphs, etc.)
│   ├── DSA-1/          # Advanced topics (DP, Greedy, Complexity, Strings/Bits)
│   ├── 13_oop_fundamentals.pdf
│   └── 14_solid_principles.pdf
├── requirements.txt    # Project dependencies
└── .env                # Local configuration & GEMINI_API_KEY (not committed)
```

---

## 💻 Getting Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Achintyasingh412/cse-ai-tutor.git
cd cse-ai-tutor
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Build the Vector Knowledge Base & Run
```bash
# Ingest PDF notes into ChromaDB
python ingest.py

# Inspect database health (optional)
python check_db.py

# Launch the Streamlit Web Application
streamlit run app.py
```

Or run the terminal chatbot:
```bash
python chatbot.py
```

---

## 👥 Contributors

### **Achintya Singh** ([@Achintyasingh412](https://github.com/Achintyasingh412))
- **Core AI Architecture & Prompt Engineering:** Designed the end-to-end AI system architecture, instructional guidelines, and pedagogical constraints for *Luna — CSE AI Tutor*.
- **Vector Database & RAG Pipeline:** Set up and integrated ChromaDB with Sentence Transformers (`all-MiniLM-L6-v2`) for semantic note retrieval and chunking.
- **Short-Term Memory System (`memory_db.py`):** Engineered the local SQLite conversation memory architecture with an automated 6-hour sliding window eviction mechanism.
- **Interactive Streamlit Web Application (`app.py`):** Modernized the user interface with distinct chat styling, dynamic status monitoring, topic drawers, and live response streaming.
- **Context Fusion Engine:** Integrated multi-turn conversation memory with retrieved RAG context for coherent, context-aware tutoring.

### **Jeevesh** ([@jeeveshcodes](https://github.com/jeeveshcodes))
- Initial prototype and course notes curation for CSE placement preparation.
- Conceived the original vision, core concept, and foundational idea for "Luna — CSE AI Tutor" to assist college students.

Established the foundational project architecture, outlining how RAG and vector retrieval can integrate with AI tutors.

Defined the primary scope and target capabilities focusing on core Computer Science Engineering subjects like DSA and OOP.

Proposed the implementation of a dual-layer memory system combining short-term chat persistence with long-term semantic notes.

Curated and structured the initial set of academic course materials and notes to populate the project's vector database.

Formulated Luna's persona, tone, and strict output guidelines, such as enforcing concise bullet points and avoiding special math formatting.

Guided the overall technical strategy and technology stack selection, choosing Python, Google Gemini, ChromaDB, and SQLite.

Designed the context-retrieval workflow ensuring student answers prioritize personal notes over generic LLM training data.

Outlined chunking and embedding strategies using Sentence Transformers to maximize document retrieval accuracy.

Led collaborative development efforts, task delegation, and code structuring across the repository.

Supervised the integration of the Streamlit web interface components with backend data and retrieval pipelines.

Maintained domain relevance and academic quality by aligning model outputs with standard college DSA and OOP syllabi.

Designed the project's open-source repository layout, setup guidelines, and dependency structure for easy deployment.

Pioneered the roadmap for advanced features like live status badges, expandable topic drawers, and chat reset capabilities.

Established the core documentation standards and usage guides to ensure maintainability and community collaboration.

---

## 🗺️ Roadmap

- [ ] Support for additional subjects (DBMS, Operating Systems, Computer Networks).
- [ ] Voice input & spoken response capabilities.
- [ ] Multi-turn quiz generation and interactive flashcards.
- [ ] Automated benchmark evaluation suite for retrieval accuracy.
