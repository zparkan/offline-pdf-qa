import sqlite3
from pathlib import Path
from contextlib import contextmanager
 
 
# ============================================================
# Database Path
# ============================================================
 
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(exist_ok=True)
DATABASE_PATH = DATABASE_DIR / "chat_history.db"
 
UPLOADS_DIR = BASE_DIR / "uploaded_files"
UPLOADS_DIR.mkdir(exist_ok=True)
 
DEFAULT_CHAT_COLOR = "#AEC6CF"  # آبی آسمانی پاستلی، پیش‌فرض چت‌های قدیمی
 
 
# ============================================================
# Database Connection
# ============================================================
 
@contextmanager
def get_connection():
    """
    Context manager برای اتصال به دیتابیس SQLite.
    commit/rollback و close به صورت خودکار مدیریت می‌شن.
    """
 
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
 
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
 
 
# ============================================================
# Create Tables
# ============================================================
 
def create_tables():
    """
    ایجاد جداول مورد نیاز در صورت عدم وجود.
    """
 
    with get_connection() as conn:
 
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                color TEXT NOT NULL DEFAULT '{DEFAULT_CHAT_COLOR}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
 
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
 
                FOREIGN KEY (chat_id)
                    REFERENCES chats(id)
                    ON DELETE CASCADE
            );
 
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
 
                FOREIGN KEY (chat_id)
                    REFERENCES chats(id)
                    ON DELETE CASCADE
            );
 
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id
            ON messages(chat_id);
 
            CREATE INDEX IF NOT EXISTS idx_files_chat_id
            ON files(chat_id);
        """)
 
 
# ============================================================
# Chat Operations
# ============================================================
 
def create_chat(title="New Chat", color=None):
    """
    ایجاد یک چت جدید.
 
    Args:
        title: اسم چت (نباید خالی باشه؛ این اعتبارسنجی سمت UI انجام می‌شه)
        color: کد رنگ هگز (مثلاً "#AEC6CF"). اگه داده نشه، رنگ پیش‌فرض
               استفاده می‌شه.
    """
    if color is None:
        color = DEFAULT_CHAT_COLOR
 
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chats (title, color) VALUES (?, ?)",
            (title, color)
        )
        return cursor.lastrowid
 
 
def get_chats():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *
            FROM chats
            ORDER BY updated_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
 
 
def get_chat_by_id(chat_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
 
 
def rename_chat(chat_id, new_title):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE chats
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_title, chat_id)
        )
 
 
def delete_chat(chat_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
 
 
# ============================================================
# Message Operations
# ============================================================
 
def add_message(chat_id, role, content):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (chat_id, role, content)
            VALUES (?, ?, ?)
            """,
            (chat_id, role, content)
        )
        conn.execute(
            """
            UPDATE chats
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (chat_id,)
        )
 
 
def get_messages(chat_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (chat_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
 
 
# ============================================================
# File Operations
# ============================================================
 
def add_file(chat_id, filename, stored_path, status="pending"):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO files (chat_id, filename, stored_path, status)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, filename, stored_path, status)
        )
        return cursor.lastrowid
 
 
def get_files(chat_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM files
            WHERE chat_id = ?
            ORDER BY created_at ASC
            """,
            (chat_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
 
 
def update_file_status(file_id, status):
    with get_connection() as conn:
        conn.execute(
            "UPDATE files SET status = ? WHERE id = ?",
            (status, file_id)
        )
 
 
def delete_file(file_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT stored_path FROM files WHERE id = ?", (file_id,))
        row = cursor.fetchone()
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        return row["stored_path"] if row else None
 
 
# ============================================================
# Initialize Database
# ============================================================
 
def initialize_database():
    create_tables()
 
 
if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully.")
    print(f"Database path: {DATABASE_PATH}")
