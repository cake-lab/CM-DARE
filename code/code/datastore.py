import sqlite3

class StatusDAO(object):

	def __init__(self, db):
		self.db_conn = sqlite3.connect(db, check_same_thread=False)

	def add_entry(self, name, creation_time=None, termination_time=None):
		cursor = self.db_conn.cursor()

		try:
			query = '''INSERT INTO server_record (name, creation_time, termination_time) VALUES (?,?,?)'''
			cursor.execute(query, (name, creation_time, termination_time))
			self.db_conn.commit()
		except Exception as e:
			print "An error occured when adding server to DB:", e
			self.db_conn.rollback()
		finally:
			cursor.close()

	def update_creation_time(self, name, time):
		cursor = self.db_conn.cursor()

		try:
			query = '''UPDATE server_record SET creation_time=? WHERE name=?'''
			cursor.execute(query, (time, name))
			self.db_conn.commit()
		except Exception as e:
			print "An error occured when adding server to DB:", e
			self.db_conn.rollback()
		finally:
			cursor.close()
	

	def update_termination_time(self, name, time):
		cursor = self.db_conn.cursor()

		try:
			query = '''UPDATE server_record SET termination_time=? WHERE name=?'''
			cursor.execute(query, (time, name))
			self.db_conn.commit()
		except Exception as e:
			print "An error occured when adding server to DB:", e
			self.db_conn.rollback()
		finally:
			cursor.close()