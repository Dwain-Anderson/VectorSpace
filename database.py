import sqlite3
import pandas as pd
from queue import Queue
from vector_model import TextProcessor


class ConnectionPool:
    def __init__(self, db_path, pool_size=5):
        self.pool = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            self.pool.put(conn)

    def get_connection(self):
        return self.pool.get()

    def release_connection(self, conn):
        self.pool.put(conn)

class Database:
    def __init__(self, db_path='VectorSpace.db'):
        self.pool = ConnectionPool(db_path)
        self._initialize_tables()
        self.txtp = TextProcessor

    def _initialize_tables(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                content TEXT
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS cached_queries (
                query TEXT PRIMARY KEY,
                result TEXT
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS pagerank (
                doc_id INTEGER PRIMARY KEY,
                score REAL,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
            ''')
            conn.commit()
        finally:
            self.pool.release_connection(conn)

    def save_document(self, url, document, conn):
        try:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO documents (url, content) VALUES (?, ?)''', (url, document))
            conn.commit()
            print(f"Document saved successfully. URL: {url}")
        except sqlite3.IntegrityError as e:
            print(f"Integrity error saving document: {e}")
        except sqlite3.OperationalError as e:
            print(f"Operational error saving document: {e}")

    def save_pagerank(self, pagerank):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            for doc_id, score in pagerank.items():
                cursor.execute('''INSERT OR REPLACE INTO pagerank (doc_id, score) VALUES (?, ?)''', (doc_id, score))
            conn.commit()
            print(f"PageRank scores saved successfully.")
        except sqlite3.IntegrityError as e:
            print(f"Integrity error saving PageRank: {e}")
        except sqlite3.OperationalError as e:
            print(f"Operational error saving PageRank: {e}")
        finally:
            self.pool.release_connection(conn)

    def save_query_result(self, query, result):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR REPLACE INTO cached_queries (query, result) VALUES (?, ?)''', (query, result))
            conn.commit()
            print(f"Query result saved successfully.")
        except sqlite3.OperationalError as e:
            print(f"Operational error saving query result: {e}")
        finally:
            self.pool.release_connection(conn)

    def get_all_cached_queries(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''SELECT query FROM cached_queries''')
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        finally:
            self.pool.release_connection(conn)

    def get_query_result(self, query):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''SELECT result FROM cached_queries WHERE query = ?''', (query,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            self.pool.release_connection(conn)

    def get_pagerank_scores(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''SELECT doc_id, score FROM pagerank''')
            df = pd.DataFrame(cursor.fetchall(), columns=['doc_id', 'score'])
            return df.set_index('doc_id')['score'].to_dict()
        finally:
            self.pool.release_connection(conn)

    def get_documents(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''SELECT id, url, content FROM documents''')
            df = pd.DataFrame(cursor.fetchall(), columns=['id', 'url', 'content'])
            return df.set_index('id')[['url', 'content']].to_dict(orient='index')
        finally:
            self.pool.release_connection(conn)

    def search(self, query):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''SELECT content FROM documents WHERE content LIKE ?''', ('%' + query + '%',))
            results = cursor.fetchall()
            return [row[0] for row in results]
        finally:
            self.pool.release_connection(conn)

    def prune_cache(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''DROP TABLE IF EXISTS cached_queries''')
            conn.commit()
            print("Cache cleared successfully.")
        except sqlite3.OperationalError:
            print("Operational error clearing cache.")
        finally:
            self.pool.release_connection(conn)

    def prune_documents(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''DROP TABLE IF EXISTS documents''')
            conn.commit()
            print("Documents cleared successfully.")
        except sqlite3.OperationalError:
            print("Operational error clearing cache.")
        finally:
            self.pool.release_connection(conn)

    def get_k_urls(self, k=None):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            if k is None:
                cursor.execute('''SELECT url FROM documents ORDER BY id DESC''')
            elif k == 1:
                cursor.execute('''SELECT url FROM documents WHERE id = (SELECT MAX(id) FROM documents)''')
            else:
                query = f'''SELECT url FROM documents ORDER BY id DESC LIMIT {k}'''
                cursor.execute(query)
            rows = cursor.fetchall()
            urls = [row for row in rows]
            return urls
        finally:
            self.pool.release_connection(conn)




    def get_text_content(self, url):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''SELECT content FROM documents WHERE url = ?''', (url,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            self.pool.release_connection(conn)
