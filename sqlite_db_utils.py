import sqlite3
from datetime import datetime

def create_iamge_table(prefix_path:str, name:str = 'chat.db'):
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

def create_index_table(prefix_path:str, name:str = 'chat.db'):
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


def store_img_data(data, prefix_path:str, name:str = 'chat.db'):
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

def get_last_image_data(prefix_path:str, name:str = 'chat.db'):
    path_to_db = prefix_path + name
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Query to retrieve the last record
    select_query = '''
        SELECT * FROM image_data ORDER BY ROWID DESC LIMIT 1
    '''

    cursor.execute(select_query)
    last_record = cursor.fetchone()

    conn.close()
    return last_record

def find_msg_id(index_id: int, img_id: int, prefix_path: str, name:str = 'chat.db'):
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
    

def add_index_record(index_data, prefix_path: str, name:str = 'chat.db'):
    path_to_db = prefix_path + name
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Insert the record into the index table
    insert_query = '''
        INSERT INTO index_table 
        (index_id, index_name, index_size, max_size, nfeatures, desc_size, desc_name) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    cursor.execute(insert_query, (
        index_data['index_id'],
        index_data['index_name'],
        index_data['index_size'],
        index_data['max_size'],
        index_data['nfeatures'],
        index_data['desc_size'],
        index_data['desc_name']
    ))

    conn.commit()
    conn.close()


def get_index_triplets(prefix_path: str, name:str = 'chat.db'):
    path_to_db = prefix_path + name
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Query to retrieve index_id, index_name, and desc_name triples
    select_query = '''
        SELECT index_id, index_name, desc_name FROM index_table
    '''
    cursor.execute(select_query)
    triplets = cursor.fetchall()

    conn.close()
    return triplets