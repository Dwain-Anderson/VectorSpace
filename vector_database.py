import sqlite3
import numpy as np
from scipy.spatial import distance
from database import Database
from vector_model import KMeans, Vectorizer, compute_total_bits
import threading


class VectorizedDatabase(Database):
    def __init__(self, n=25, db_path='VectorSpace.db'):
        super().__init__(db_path)
        self.n_dims = n
        self._initialize_vector_table()
        self.lock = threading.Lock()


    def _initialize_vector_table(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DROP TABLE IF EXISTS vectors')
            columns = ', '.join(f'v{i + 1} REAL' for i in range(self.n_dims))
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS vectors (
                doc_id INTEGER PRIMARY KEY,
                {columns},
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
            ''')
            conn.commit()
        finally:
            self.pool.release_connection(conn)

    # Instance of a write to database.
    def save_document_and_vector(self, url, document):
        with self.lock:
            conn = self.pool.get_connection()
            try:
                cursor = conn.cursor()
                self.save_document(url, document, conn)  # both release lock
                doc_id = cursor.lastrowid
                self.insert_vector(doc_id, document, conn)  # both release lock
            finally:
                self.pool.release_connection(conn)

    def insert_vector(self, doc_id, content, conn):
        try:
            v1 = Vectorizer(self.n_dims).generate_vector(content)
            norm = np.linalg.norm(v1)
            if norm > 0:
                v1 = v1 / norm
            else:
                v1 = v1
            if isinstance(v1, np.ndarray):
                vector = v1.tolist()
            else:
                vector = v1

            # Ensure doc_id is an integer
            if isinstance(doc_id, float):
                doc_id = int(doc_id)

            # Prepare column names and placeholders
            columns = ', '.join(f'v{i + 1}' for i in range(self.n_dims))
            placeholders = ', '.join('?' for _ in range(self.n_dims))

            # Prepare the SQL statement
            sql = f'''
                INSERT OR REPLACE INTO vectors (doc_id, {columns})
                VALUES (?, {placeholders})
            '''
            cursor = conn
            # Execute the SQL statement with the vector values
            cursor.execute(sql, (doc_id, *vector))
            conn.commit()

        except sqlite3.IntegrityError as e:
            print(f"Integrity error saving vector: {e}")
        except sqlite3.OperationalError as e:
            print(f"Operational error saving vector: {e}")


    def get_vector(self, doc_id):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM vectors WHERE doc_id = ?', (doc_id,))
            row = cursor.fetchone()
            if row:
                return np.array(row[1:])
            return None
        finally:
            self.pool.release_connection(conn)

    def prune_vectors(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DROP TABLE IF EXISTS vectors')
            conn.commit()
            print("Vectors cleared successfully.")
        except sqlite3.OperationalError as e:
            print(f"Operational error clearing cache: {e}")
        finally:
            self.pool.release_connection(conn)

    def k_nearest_neighbors(self, query_vector, k=5):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT doc_id, ' + ', '.join(f'v{i + 1}' for i in range(self.n_dims)) + ' FROM vectors')
            rows = cursor.fetchall()
            distances = [(row[0], distance.euclidean(query_vector, np.array(row[1:]))) for row in rows]
            distances.sort(key=lambda x: x[1])
            return [doc_id for doc_id, dist in distances[:k]]
        finally:
            self.pool.release_connection(conn)

    def k_means_clustering(self, k=5):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT ' + ', '.join(f'v{i + 1}' for i in range(self.n_dims)) + ' FROM vectors')
            vectors = np.array(cursor.fetchall())
            kmeans = KMeans(k=k)
            clusters = kmeans.fit(vectors)
            return clusters, kmeans.centroids
        finally:
            self.pool.release_connection(conn)

    def compute_total_content_bits(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            # Retrieve all text content from the documents table
            cursor.execute('SELECT content FROM documents')
            rows = cursor.fetchall()
            total_bits = compute_total_bits(rows)
            print(f"Total bits of information across all documents: {total_bits:.2f}")
            return total_bits

        except sqlite3.OperationalError as e:
            print(f"Operational error computing total bits: {e}")

        finally:
            self.pool.release_connection(conn)

    def translate_database(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            # Retrieve all documents with their content
            cursor.execute('SELECT id, url, content FROM documents')
            rows = cursor.fetchall()

            # Process each document and update its content and URL
            for idx, value in enumerate(rows):
                url = value[1]
                content = value[2]

                # Translate the current content to English and clean it
                text = self.txtp.clean_content(content)

                # Update the document content and URL using the new idx
                # Inline SQL statement
                sql = f"""
                    UPDATE documents 
                    SET url = ?, content = ? 
                    WHERE id = ?
                """
                cursor.execute(sql, (url, text, idx + 1))
                # Recalculate the vector for the translated text
                v1 = Vectorizer(self.n_dims).generate_vector(text)
                norm = np.linalg.norm(v1)
                v1 = v1 / norm if norm > 0 else v1
                vector = v1.tolist() if isinstance(v1, np.ndarray) else v1

                # Prepare column names and placeholders for the vector update
                columns = ', '.join(f'v{i + 1} = ?' for i in range(self.n_dims))

                # Update the vector in the vectors table using the new idx
                sql = f'''
                    UPDATE vectors
                    SET {columns}
                    WHERE doc_id = ?
                '''
                cursor.execute(sql, (*vector, idx + 1))

            # Commit the transaction
            conn.commit()
            print("Database translation and vector update completed successfully.")
        except sqlite3.OperationalError as e:
            print(f"Operational error during database translation: {e}")
        finally:
            self.pool.release_connection(conn)
