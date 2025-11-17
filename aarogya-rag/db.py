# import psycopg2
# import pandas as pd

# conn = psycopg2.connect(
#     dbname="aarogya",
#     user="user",
#     password="password",
#     host="localhost",
#     port="5432"
# )
# df = pd.read_sql("SELECT * FROM reports ORDER BY created_at DESC LIMIT 5;", conn)
# print(df)
# conn.close()
# list_tables.py
import psycopg2

conn = psycopg2.connect(dbname="aarogya", user="user", password="password", host="localhost", port="5432")
cur = conn.cursor()
cur.execute("""
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
    ORDER BY tablename;
""")
for row in cur.fetchall():
    print(row[0])
cur.close()
conn.close()
