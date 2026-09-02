"""
Memory Database Module for Luna — CSE AI Tutor.
Manages local SQLite short-term conversation storage with a 6-hour sliding memory window.
"""

from datetime import datetime, timedelta
import os
import sqlite3
from typing import Any, Dict, List, Optional

DB_NAME = "chat_memory.db"


def get_db_path() -> str:
    """Returns absolute path to the database file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)


def init_db() -> None:
    """Initializes the SQLite database table and indices for chat history."""
    db_path = get_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_timestamp 
                ON chat_history (timestamp)
            """)
            conn.commit()
    except sqlite3.Error as e:
        print(f"[MemoryDB Error] Failed to initialize database: {e}")


def save_message(role: str, content: str) -> None:
    """Saves a user or assistant message with an exact timestamp."""
    db_path = get_db_path()
    init_db()
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, datetime.now()),
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"[MemoryDB Error] Failed to save message: {e}")


def clean_old_messages(hours: int = 6) -> None:
    """Deletes any chat logs older than the specified hours (default 6 hours)."""
    db_path = get_db_path()
    purge_threshold = datetime.now() - timedelta(hours=hours)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM chat_history WHERE timestamp < ?", (purge_threshold,)
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"[MemoryDB Error] Failed to clean old messages: {e}")


def get_recent_conversation_history(hours: int = 6) -> List[Dict[str, str]]:
    """Cleans old data and returns messages from the last 6 hours."""
    init_db()
    clean_old_messages(hours)

    db_path = get_db_path()
    messages: List[Dict[str, str]] = []

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            time_threshold = datetime.now() - timedelta(hours=hours)
            cursor.execute(
                """
                SELECT role, content 
                FROM chat_history 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC
                """,
                (time_threshold,),
            )
            rows = cursor.fetchall()
            for role, content in rows:
                messages.append({"role": role, "content": content})
    except sqlite3.Error as e:
        print(f"[MemoryDB Error] Failed to retrieve history: {e}")

    return messages


def log_conversation(user_text: str, ai_text: str) -> None:
    """Saves both the user prompt and AI response in chronological order."""
    if user_text:
        save_message("user", user_text)
    if ai_text:
        save_message("assistant", ai_text)


def clear_chat_history() -> bool:
    """Wipes all conversation history from the SQLite database."""
    db_path = get_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history")
            conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"[MemoryDB Error] Failed to clear chat history: {e}")
        return False


def get_memory_stats(hours: int = 6) -> Dict[str, Any]:
    """Returns diagnostic statistics about current active memory."""
    init_db()
    clean_old_messages(hours)
    db_path = get_db_path()
    stats: Dict[str, Any] = {"count": 0, "last_active": None, "window_hours": hours}
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            time_threshold = datetime.now() - timedelta(hours=hours)
            cursor.execute(
                "SELECT COUNT(*), MAX(timestamp) FROM chat_history WHERE timestamp >= ?",
                (time_threshold,),
            )
            row = cursor.fetchone()
            if row:
                stats["count"] = row[0] or 0
                stats["last_active"] = row[1]
    except sqlite3.Error as e:
        print(f"[MemoryDB Error] Failed to get memory stats: {e}")
    return stats


def format_history_for_prompt(messages: Optional[List[Dict[str, str]]] = None, max_turns: int = 6) -> str:
    """Formats recent conversation history for inclusion in the LLM prompt."""
    if messages is None:
        messages = get_recent_conversation_history()
    
    if not messages:
        return "No prior conversation in current session."

    # Take the most recent turns (up to max_turns)
    recent = messages[-max_turns:]
    formatted = []
    for msg in recent:
        speaker = "Student" if msg.get("role") == "user" else "Luna"
        formatted.append(f"{speaker}: {msg.get('content', '')}")
    
    return "\n".join(formatted)


# Auto-initialize table on import
init_db()

if __name__ == "__main__":
    init_db()
    print("Chat memory database initialized successfully!")
    stats = get_memory_stats()
    print(f"Active messages in last 6 hours: {stats['count']}")