"""
Terminal CLI Chatbot for Luna — CSE AI Tutor.
Combines ChromaDB vector retrieval with SQLite short-term conversation memory.
"""

import os
import sys
from dotenv import load_dotenv
from google import genai
import chroma_compat  # Compatibility shim for gRPC telemetry
import chromadb
from chromadb.utils import embedding_functions

from memory_db import (
    clear_chat_history,
    format_history_for_prompt,
    get_recent_conversation_history,
    log_conversation,
)

load_dotenv()

# System instruction defining Luna's personality, formatting rules, and pedagogy
SYSTEM_INSTRUCTION = (
    "You are Luna, an expert and encouraging CSE tutor for college students preparing for exams, "
    "coding assessments, and technical placements.\n\n"
    "You have access to two sources of information:\n"
    "1. RECENT CONVERSATION HISTORY: Past discussion context within the current session. Use this to "
    "understand follow-up questions, pronouns, and references.\n"
    "2. RETRIEVED COURSE NOTES: High-priority knowledge retrieved from the student's own DSA and OOP syllabus notes.\n\n"
    "CORE INSTRUCTIONS:\n"
    "- Base your answers primarily on the retrieved course notes when relevant.\n"
    "- If the retrieved notes do not contain the answer, explicitly state that and answer accurately using your broader Computer Science knowledge.\n"
    "- Maintain context across questions using the conversation history.\n"
    "- STRICT LENGTH & FORMAT RULE: Keep answers concise (under 120 words) formatted as 3 to 6 crisp, single-sentence bullet points, "
    "unless the student explicitly asks to 'explain in detail', 'go deep', 'give an example', or 'show code'.\n"
    "- NO raw LaTeX or math delimiters (do not use \\(O(n)\\) or $O(n)$). Format mathematical complexities in plain text, e.g., O(n log n), O(n^2), or O(1).\n"
    "- Always be supportive, technically precise, and exam-focused."
)


def initialize_clients():
    """Initializes and returns the Gemini client and ChromaDB collection."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[!] Warning: GEMINI_API_KEY is not set in environment or .env file.")
        print("    Please set your GEMINI_API_KEY to interact with Luna.\n")

    client = genai.Client()

    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(
            name="dsa_notes",
            embedding_function=embedding_fn
        )
    except Exception as e:
        print(f"[!] Warning: Could not initialize ChromaDB ({e}).")
        print("    Make sure 'python ingest.py' has been run to index your notes.\n")
        collection = None

    return client, collection


def retrieve_context(collection, question: str, n_results: int = 5) -> str:
    """Safely retrieves relevant note chunks from ChromaDB."""
    if collection is None:
        return "No note vector database available."

    try:
        count = collection.count()
        if count == 0:
            return "Knowledge base is currently empty. Run 'python ingest.py' to index notes."
        
        results = collection.query(
            query_texts=[question],
            n_results=min(n_results, count)
        )
        documents = results.get("documents", [[]])
        if documents and documents[0]:
            return "\n\n---\n\n".join(documents[0])
        return "No relevant note passages found."
    except Exception as e:
        return f"Error retrieving context: {e}"


def run_chatbot():
    print("=" * 60)
    print(" 🌙 Luna — CSE AI Tutor (Terminal Edition)")
    print("=" * 60)
    print(" Commands: type 'quit' to exit | 'clear' to reset chat memory\n")

    client, collection = initialize_clients()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Luna. Happy studying!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! Best of luck with your CSE prep! 🚀")
            break

        if user_input.lower() in ("clear", "reset"):
            clear_chat_history()
            print("✓ Chat memory cleared. Starting fresh session.\n")
            continue

        # 1. Fetch short-term memory (last 6 hours)
        recent_history = get_recent_conversation_history()
        history_context = format_history_for_prompt(recent_history, max_turns=6)

        # 2. Retrieve long-term RAG notes
        rag_context = retrieve_context(collection, user_input)

        # 3. Construct unified prompt
        prompt_payload = (
            f"### RECENT CONVERSATION HISTORY (Last 6 Hours):\n{history_context}\n\n"
            f"### RETRIEVED COURSE NOTES (RAG Context):\n{rag_context}\n\n"
            f"### STUDENT QUESTION:\n{user_input}"
        )

        try:
            # We use gemini-2.5-flash or gemini-1.5-flash with fallback
            model_name = "gemini-2.5-flash"
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt_payload,
                    config=dict(system_instruction=SYSTEM_INSTRUCTION),
                )
                answer = response.text
            except Exception:
                # Fallback to interactions API or gemini-1.5-flash if needed
                interaction = client.interactions.create(
                    model="gemini-3.5-flash",
                    system_instruction=SYSTEM_INSTRUCTION,
                    input=prompt_payload,
                )
                answer = interaction.output_text

            print(f"\nLuna:\n{answer}\n")
            
            # 4. Save to short-term SQLite memory
            log_conversation(user_input, answer)

        except Exception as err:
            print(f"\n[!] Error generating response: {err}\n")


if __name__ == "__main__":
    run_chatbot()