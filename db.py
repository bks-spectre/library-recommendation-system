import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        name TEXT,
        grade INTEGER,
        preference_genre TEXT,
        preferred_level TEXT,
        books_read INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedbacks (
        id SERIAL PRIMARY KEY,
        name TEXT,
        student_id TEXT,
        rating INTEGER,
        feedback TEXT,
        date TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS book_ratings (
        id SERIAL PRIMARY KEY,
        student_id TEXT,
        book_id TEXT,
        rating INTEGER,
        review TEXT,
        date_read TEXT
    )
    """)

    conn.commit()
    conn.close()
