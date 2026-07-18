import sqlite3
import hashlib

DB = "users.db"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_users_table():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def signup_user(name, email, password):
    email = email.strip().lower()
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE email=?",
        (email,),
    )

    if cur.fetchone():
        conn.close()
        return False, "Email already exists."

    cur.execute(
        "INSERT INTO users(name,email,password) VALUES(?,?,?)",
        (
            name,
            email,
            hash_password(password),
        ),
    )

    conn.commit()
    conn.close()

    return True, "Account created successfully."


def login_user(email, password):

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    email = email.strip().lower()

    # Keep lightweight logging (Streamlit will capture stdout)
    print("Trying Login:", email)

    cur.execute(
        "SELECT * FROM users WHERE LOWER(email)=? AND password= ?",
        (
            email,
            hash_password(password),
        ),
    )

    user = cur.fetchone()

    conn.close()

    return user


def reset_password(email, new_password):

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    email = email.strip().lower()

    cur.execute(
        "SELECT * FROM users WHERE LOWER(email)=?",
        (email,),
    )

    user = cur.fetchone()

    if not user:
        conn.close()
        return False

    cur.execute(
        "UPDATE users SET password=? WHERE LOWER(email)=?",
        (
            hash_password(new_password),
            email,
        ),
    )

    conn.commit()
    conn.close()

    return True


if __name__ == "__main__":
    # Create the users table when running this module directly for CLI setup/testing.
    create_users_table()
    print("Users table created (if it did not exist).")
