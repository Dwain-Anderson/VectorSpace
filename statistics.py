import sqlite3
import pandas as pd
import datetime
import time
from database import Database

class Statistics(Database):
    def __init__(self, db_file='VectorSpace.db'):
        super().__init__(db_file)
        self.create_statistics_table()
        self.current_stats = {}

    def create_statistics_table(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT,
                    end_time TEXT,
                    total_time TEXT, 
                    threads_used INTEGER,
                    pages_gathered INTEGER,
                    links_seen INTEGER,
                    status TEXT
                )
            ''')
            conn.commit()
        finally:
            self.pool.release_connection(conn)

    def start_new_session(self, threads_used):
        start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        self.current_stats = {
            'start_time': start_time,
            'end_time': '',
            'total_time': '',
            'threads_used': threads_used,  # Set the threads_used parameter here
            'pages_gathered': 0,
            'links_seen': 0,
            'status': 'In Progress'
        }

    def end_session(self, pages_gathered, links_seen):
        end_time = time.strftime('%Y-%m-%d %H:%M:%S')
        start_time = time.strptime(self.current_stats['start_time'], '%Y-%m-%d %H:%M:%S')
        end_time_struct = time.strptime(end_time, '%Y-%m-%d %H:%M:%S')

        # Calculate total time in seconds
        total_time_seconds = time.mktime(end_time_struct) - time.mktime(start_time)
        total_time_str = time.strftime('%H:%M:%S', time.gmtime(total_time_seconds))

        self.current_stats.update({
            'end_time': end_time,
            'total_time': total_time_str,
            'pages_gathered': pages_gathered,
            'links_seen': links_seen,
            'status': 'Completed'
        })

        self.save_statistics()

    def save_statistics(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO statistics (start_time, end_time, total_time, threads_used, pages_gathered, links_seen, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (self.current_stats.get('start_time'), self.current_stats.get('end_time'),
                  self.current_stats.get('total_time'), self.current_stats.get('threads_used'),
                  self.current_stats.get('pages_gathered'), self.current_stats.get('links_seen'),
                  self.current_stats.get('status')))
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"Operational error saving statistics: {e}")
        finally:
            self.pool.release_connection(conn)

    def print_statistics(self):
        conn = self.pool.get_connection()
        try:
            query = 'SELECT * FROM statistics'
            df = pd.read_sql_query(query, conn)
            print(df)
        finally:
            self.pool.release_connection(conn)

    def print_most_recent_statistics(self):
        conn = self.pool.get_connection()
        try:
            query = 'SELECT * FROM statistics ORDER BY id DESC LIMIT 1'
            df = pd.read_sql_query(query, conn)
            print(df)
        finally:
            self.pool.release_connection(conn)

    def prune_statistics(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''DROP TABLE IF EXISTS statistics''')
            conn.commit()
            print("Statistics cleared successfully.")
        except sqlite3.OperationalError:
            print("Operational error clearing statistics.")
        finally:
            self.pool.release_connection(conn)

    def current_expected_time(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            A = cursor.execute('''SELECT id, total_time FROM statistics''')
            conn.commit()
            print(f"Current expected time is: {self.calculate_expected_time(A)}")
        except sqlite3.OperationalError:
            print("Operational error clearing statistics.")
        finally:
            self.pool.release_connection(conn)

    def total_time_crawling(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            Ts = cursor.execute('''SELECT total_time FROM statistics''').fetchall()
            conn.commit()
            total_time_spent = sum(self.time_to_seconds(time_string[0]) for time_string in Ts)
            print(f"Total time spent crawling: {total_time_spent} seconds")
        except sqlite3.OperationalError as e:
            print(f"Operational error retrieving total time: {e}")
        finally:
            self.pool.release_connection(conn)

    def time_to_seconds(self, T):
        x = time.strptime(T.split(',')[0],'%H:%M:%S')
        return datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()

    def calculate_expected_time(self, A):
        accum = 0
        N = 0
        for data in A:
            accum += (self.time_to_seconds(data[1]))  # Assuming data[1] is the total_time string
            N += 1
        return accum / N if N != 0 else 0  # Prevent division by zero

    def update_ids(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()

            # Create a temporary table to store old IDs and their new sequential IDs
            cursor.execute('''
                CREATE TEMPORARY TABLE temp_ids AS
                SELECT id, ROW_NUMBER() OVER (ORDER BY id) - 1 AS new_id
                FROM statistics
            ''')

            # Update the statistics table with the new IDs
            cursor.execute('''
                UPDATE statistics
                SET id = (
                    SELECT new_id
                    FROM temp_ids
                    WHERE statistics.id = temp_ids.id
                )
            ''')

            conn.commit()
            print("IDs updated successfully.")

        except sqlite3.OperationalError as e:
            print(f"Operational error updating IDs: {e}")

        finally:
            self.pool.release_connection(conn)


    def fix_statistics_ids(self):
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            # Step 1: Create a temporary table with the correct id sequence
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS temp_statistics (
                    new_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT,
                    end_time TEXT,
                    total_time TEXT, 
                    threads_used INTEGER,
                    pages_gathered INTEGER,
                    links_seen INTEGER,
                    status TEXT
                )
            ''')

            # Step 2: Copy data from the original table to the temporary table with corrected ids
            cursor.execute('''
                INSERT INTO temp_statistics (start_time, end_time, total_time, threads_used, pages_gathered, links_seen, status)
                SELECT start_time, end_time, total_time, threads_used, pages_gathered, links_seen, status
                FROM statistics
            ''')

            # Step 3: Drop the original table
            cursor.execute('DROP TABLE statistics')

            # Step 4: Rename the temporary table to the original table name
            cursor.execute('ALTER TABLE temp_statistics RENAME TO statistics')

            conn.commit()
            print("Statistics table id values fixed successfully.")

        except sqlite3.OperationalError as e:
            print(f"Operational error fixing ids: {e}")
        finally:
            conn.close()



