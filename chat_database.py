import sqlite3
from pathlib import Path


# ============================================================
# Database Path
# ============================================================

# مسیر اصلی فایل پروژه
BASE_DIR = Path(__file__).resolve().parent

# پوشه دیتابیس
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(exist_ok=True)

# فایل دیتابیس
DATABASE_PATH = DATABASE_DIR / "chat_history.db"


# ============================================================
# Database Connection
# ============================================================

def get_connection():
    """
    ایجاد اتصال به دیتابیس SQLite
    """

    connection = sqlite3.connect(DATABASE_PATH)

    # دسترسی به ستون‌ها به صورت dictionary-like
    connection.row_factory = sqlite3.Row

    # فعال کردن Foreign Key
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# ============================================================
# Create Tables
# ============================================================

def create_tables():
    """
    ایجاد جداول مورد نیاز در صورت عدم وجود
    """

    with get_connection() as conn:

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
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

            CREATE INDEX IF NOT EXISTS idx_messages_chat_id
            ON messages(chat_id);
        """)


# ============================================================
# Chat Operations
# ============================================================

def create_chat(title="New Chat"):
    """
    ایجاد یک چت جدید

    Returns:
        int: شناسه چت ایجاد شده
    """

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO chats (title) VALUES (?)",
            (title,)
        )

        return cursor.lastrowid


def get_chats():
    """
    دریافت تمام چت‌ها

    چت‌هایی که اخیراً آپدیت شده‌اند
    در ابتدای لیست قرار می‌گیرند.
    """

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM chats
            ORDER BY updated_at DESC
        """)

        return [dict(row) for row in cursor.fetchall()]


def delete_chat(chat_id):
    """
    حذف یک چت

    به دلیل ON DELETE CASCADE،
    تمام پیام‌های مربوط به آن چت نیز حذف می‌شوند.
    """

    with get_connection() as conn:

        conn.execute(
            "DELETE FROM chats WHERE id = ?",
            (chat_id,)
        )


# ============================================================
# Message Operations
# ============================================================

def add_message(chat_id, role, content):
    """
    اضافه کردن یک پیام به یک چت

    role می‌تواند مثلاً:
        user
        assistant
    باشد.
    """

    with get_connection() as conn:

        # اضافه کردن پیام
        conn.execute(
            """
            INSERT INTO messages
                (chat_id, role, content)
            VALUES
                (?, ?, ?)
            """,
            (chat_id, role, content)
        )

        # به‌روزرسانی زمان آخرین فعالیت چت
        conn.execute(
            """
            UPDATE chats
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (chat_id,)
        )


def get_messages(chat_id):
    """
    دریافت تمام پیام‌های یک چت
    """

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
# Initialize Database
# ============================================================

def initialize_database():
    """
    راه‌اندازی اولیه دیتابیس و ساخت جداول
    """

    create_tables()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    initialize_database()

    print("Database initialized successfully.")
    print(f"Database path: {DATABASE_PATH}")
