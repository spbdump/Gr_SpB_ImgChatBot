import sqlite3
from datetime import datetime

import context as ctx
from index import Index, ImageData

index_table_name = 'indexes'

GENERAL_DB_NAME = 'bot.db'
CHAT_DB_NAME = 'chat.db'
PATH_TO_GENERAL_DB = './'

def create_image_table(prefix_path:str, name:str = CHAT_DB_NAME):
    path_to_db = prefix_path + name
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Create a table to store image data if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_data (
            index_id INTEGER,
            img_id INTEGER,
            t_msg_id INTEGER,
            img_name TEXT,
            created_at TEXT,
            PRIMARY KEY (index_id, img_id)
        )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def create_index_table(prefix_path:str, name:str = CHAT_DB_NAME):
    path_to_db = prefix_path + name
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Create the index table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS indexes (
            index_id INTEGER PRIMARY KEY,
            index_name TEXT,
            index_size INTEGER,
            max_size INTEGER,
            nfeatures INTEGER,
            desc_size INTEGER,
            desc_name TEXT
        )
    ''')

    conn.commit()
    conn.close()

def create_ctx_table(prefix_path: str = PATH_TO_GENERAL_DB,
                     name: str = GENERAL_DB_NAME):
    path_to_db = prefix_path + name
    try:
        # Connect to the database (it will be created if it doesn't exist)
        connection = sqlite3.connect(path_to_db)
        cursor = connection.cursor()

        # Define the SQL query to create the "contexts" table
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS context (
            nfeatures INTEGER,
            desc_size INTEGER,
            max_size INTEGER,
            chat_path TEXT,
            chat_id INTEGER PRIMARY KEY
        )
        '''

        # Execute the SQL query to create the table
        cursor.execute(create_table_query)

        # Commit the changes and close the database connection
        connection.commit()
        connection.close()

        print("Table 'context' created successfully.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")


def store_img_data(data, prefix_path:str, name:str = CHAT_DB_NAME):
    path_to_db = prefix_path + name
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()
    # Insert image data into the table
    for img_data in data:
        insert_query = '''
            INSERT INTO image_data 
            (index_id, img_id, t_msg_id, img_name, created_at) 
            VALUES (?, ?, ?, ?, ?)
        '''
        cursor.execute(insert_query, (
            img_data["index_id"],
            img_data["img_id"],
            img_data["t_msg_id"],
            img_data["img_name"],
            str(datetime.now())
        ))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def save_runtime_img_data(index_id:int, start_id:int, data, prefix_path:str, name:str = CHAT_DB_NAME):
    path_to_db = prefix_path + name
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()
    # Insert image data into the table
    for img_data in data:
        insert_query = '''
            INSERT INTO image_data 
            (index_id, img_id, t_msg_id, img_name, created_at) 
            VALUES (?, ?, ?, ?, ?)
        '''
        cursor.execute(insert_query, (
            index_id,
            start_id + img_data.img_id,
            img_data.t_msg_id,
            img_data.img_name,
            str(datetime.now())
        ))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def get_last_image_data(prefix_path:str, name:str = CHAT_DB_NAME):
    try:
        path_to_db = prefix_path + name
        conn = sqlite3.connect(path_to_db)
        cursor = conn.cursor()

        # Query to retrieve the last record
        select_query = '''
            SELECT * FROM image_data ORDER BY ROWID DESC LIMIT 1
        '''

        cursor.execute(select_query)
        last_record = cursor.fetchone()
        if last_record == None:
            return None

        img_data = ImageData(
            last_record[0],
            last_record[1], #in index and pos in desc file
            last_record[2],
            last_record[3],
        )

        conn.close()
        return img_data

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None  # Return None to indicate an error occurred

def find_msg_id(index_id: int, img_id: int, prefix_path: str, name:str = CHAT_DB_NAME):
    path_to_db = prefix_path + name
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Query to retrieve the t_msg_id based on index_id and img_id
    select_query = '''
        SELECT t_msg_id FROM image_data WHERE index_id=? AND img_id=?
    '''
    cursor.execute(select_query, (index_id, img_id))
    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]
    else:
        return None


def add_index_record(index_data, prefix_path: str, name:str = CHAT_DB_NAME):
    try:
        path_to_db = prefix_path + name
        conn = sqlite3.connect(path_to_db)
        cursor = conn.cursor()

        # Insert the record into the index table
        insert_query = '''
            INSERT INTO indexes 
            (index_id, index_name, index_size, max_size, nfeatures, desc_size, desc_name) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        cursor.execute(insert_query, (
            index_data.index_id,
            index_data.index_name,
            index_data.index_size,
            index_data.max_size,
            index_data.nfeatures,
            index_data.desc_size,
            index_data.desc_name,
        ))

        conn.commit()
        conn.close()
        
        return True
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False

def get_index_triplets(prefix_path: str, name:str = CHAT_DB_NAME):
    try:
        path_to_db = prefix_path + name
        conn = sqlite3.connect(path_to_db)
        cursor = conn.cursor()

        # Query to retrieve index_id, index_name, and desc_name triples
        select_query = '''
            SELECT index_id, index_name, desc_name FROM indexes
        '''

        cursor.execute(select_query)
        triplets = cursor.fetchall()

        conn.close()
        return triplets

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None  # Return None to indicate an error occurred

def update_index_size(index_id:int, add_value:int, prefix_path: str, name:str = CHAT_DB_NAME):
    path_to_db = prefix_path + name
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(path_to_db)
        cursor = conn.cursor()

        # Get the current value from the specified column
        select_query = f"SELECT index_size FROM {index_table_name} WHERE index_id = ?"
        cursor.execute(select_query, (index_id,))
        current_value = cursor.fetchone()[0]

        # Calculate the new value by adding the add_value
        new_value = current_value + add_value

        # Update the specified column with the new value
        update_query = f"UPDATE {index_table_name} SET index_size = ? WHERE index_id = ?"
        cursor.execute(update_query, (new_value, index_id))

        # Commit the changes
        conn.commit()
        conn.close()
        return True  # Success

    except sqlite3.Error as e:
        print("SQLite error:", e)
        return False  # Error

def get_last_index_data(prefix_path: str, name:str = CHAT_DB_NAME):
    path_to_db = prefix_path + name
    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(path_to_db)
        cursor = conn.cursor()

        # Get the data of the last index based on the index_id (assuming index_id is an auto-incremented primary key)
        select_query = "SELECT * FROM indexes ORDER BY index_id DESC LIMIT 1"
        cursor.execute(select_query)
        last_index_data = cursor.fetchone()

        if last_index_data:
            index = Index(
                index_id=last_index_data[0],
                index_size=last_index_data[2],
                max_size=last_index_data[3],
                nfeatures=last_index_data[4],
                desc_size=last_index_data[5],
                index_name=last_index_data[1],
                desc_name=last_index_data[6]
            )
            return index  # Returns an Index object
        else:
            return None  # No data found

    except sqlite3.Error as e:
        print("SQLite error:", e)
        return None  # Error or no data found
    finally:
        # Close the database connection
        if conn:
            conn.close()

def does_index_exist(index_id, prefix_path: str, name:str = CHAT_DB_NAME):
    path_to_db = prefix_path + name
    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(path_to_db)
        cursor = conn.cursor()

        # Execute a SELECT query to check if the record exists
        cursor.execute("SELECT 1 FROM indexes WHERE index_id = ?", (index_id,))

        # Fetch one row (if exists)
        record = cursor.fetchone()

        return record is not None

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    finally:
        # Close the database connection
        if conn:
            conn.close()

def read_image_data_batch(database_path, batch_size, offset=0):
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        # Execute a SQL query to fetch the desired fields for a batch of records with an offset
        cursor.execute(f"SELECT img_name, index_id, img_id FROM image_data LIMIT {batch_size} OFFSET {offset}")

        # Fetch the results
        results = cursor.fetchall()

        # Close the database connection
        connection.close()

        # Extract the data into separate lists
        image_names = [row[0] for row in results]
        index_ids = [row[1] for row in results]
        image_ids = [row[2] for row in results]

        return image_names, index_ids, image_ids

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None, None, None


def get_context_by_chat_id(chat_id, database_path:str = PATH_TO_GENERAL_DB + GENERAL_DB_NAME):
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        # Execute a SQL query to retrieve the context by chat_id
        cursor.execute(f"SELECT nfeatures, desc_size, max_size, chat_path, chat_id FROM context WHERE chat_id = ?", (chat_id,))

        # Fetch the result
        result = cursor.fetchone()

        # Close the database connection
        connection.close()

        # If the result is None, no context with the specified chat_id was found
        if result is None:
            return None

        # Create a Context object from the retrieved data
        context = ctx.Context(*result)
        return context

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    
def add_context_to_db(context: ctx.Context, database_path:str = PATH_TO_GENERAL_DB + GENERAL_DB_NAME):
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        # Insert the context record into the database
        cursor.execute("INSERT INTO context (nfeatures, desc_size, max_size, chat_path, chat_id) VALUES (?, ?, ?, ?, ?)",
                       (context.nfeatures, context.desc_size, context.max_size, context.chat_path, context.chat_id))

        # Commit the changes and close the database connection
        connection.commit()
        connection.close()

        print(f"Context with chat_id {context.chat_id} added successfully.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

def add_ctx_record(chat_id, nfeatures, desc_size, max_size, chat_path, database_path:str = PATH_TO_GENERAL_DB + GENERAL_DB_NAME):
    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Execute an INSERT query to add a new record to the "context" table with the specified chat_id
        cursor.execute(
            "INSERT INTO context (chat_id, nfeatures, desc_size, max_size, chat_path) VALUES (?, ?, ?, ?, ?)",
            (chat_id, nfeatures, desc_size, max_size, chat_path)
        )

        # Commit the transaction
        conn.commit()

        return chat_id

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    finally:
        # Close the database connection
        if conn:
            conn.close()


def get_list_chats(prefix_path: str, name:str = GENERAL_DB_NAME):
    path_to_db = prefix_path + name
    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(path_to_db)
        cursor = conn.cursor()

        # Execute a SELECT query to retrieve chat_path and chat_id records
        cursor.execute("SELECT chat_path, chat_id FROM context")

        # Fetch all records
        chat_records = cursor.fetchall()

        # Extract chat paths and chat IDs from the result
        chat_data = [(record[0], record[1]) for record in chat_records]

        return chat_data

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return []
    finally:
        # Close the database connection
        if conn:
            conn.close()

def get_last_index_data_for_all(prefix_path: str):
    all_index_data = []
    chat_folder_list =  get_list_chats(prefix_path)
    for chat_frolder, chat_id in chat_folder_list:
        index = get_last_index_data( chat_frolder )
        if index == None:
            ctx = get_context_by_chat_id(chat_id)
            if ctx == None:
                continue
            index = Index(max_size=ctx.max_size, nfeatures=ctx.nfeatures, desc_size=ctx.desc_size)

        all_index_data.append( (chat_id, index) )

    return all_index_data