import sqlite3
from pathlib import Path


DATABASE_DIR = Path("database")
DATABASE_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "chat_history.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_tables():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
    """)

    connection.commit()
    connection.close()


def create_chat(title="New Chat"):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO chats (title)
        VALUES (?)
        """,
        (title,)
    )

    chat_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return chat_id


def add_message(chat_id, role, content):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO messages (chat_id, role, content)
        VALUES (?, ?, ?)
        """,
        (chat_id, role, content)
    )

    cursor.execute(
        """
        UPDATE chats
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (chat_id,)
    )

    connection.commit()
    connection.close()


def get_chats():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM chats
        ORDER BY updated_at DESC
        """
    )

    chats = cursor.fetchall()

    connection.close()

    return chats


def get_messages(chat_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, chat_id, role, content, created_at
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
        """,
        (chat_id,)
    )

    messages = cursor.fetchall()

    connection.close()

    return messages


def delete_chat(chat_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM chats
        WHERE id = ?
        """,
        (chat_id,)
    )

    connection.commit()
    connection.close()

create_tables()
