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
