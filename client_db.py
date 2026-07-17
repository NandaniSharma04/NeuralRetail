import sqlite3

conn = sqlite3.connect("users.db")
cur = conn.cursor()

cur.execute("SELECT id, name, email, password FROM users")

rows = cur.fetchall()

for row in rows:
    print(row)

conn.close()