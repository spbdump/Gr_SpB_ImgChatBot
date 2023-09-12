import sqlite3

def update_chat_path(database_path, chat_id, new_chat_path):
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        # Update the chat_path for a specific chat_id
        update_query = '''
            UPDATE context
            SET chat_path = ?
            WHERE chat_id = ?
        '''
        cursor.execute(update_query, (new_chat_path, chat_id))

        connection.commit()
        connection.close()

        print(f"Chat path updated successfully for chat_id {chat_id}.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

# Example usage:
database_path = 'bot.db'  # Replace with the path to your database file
chat_id_to_update = -1001685634092  # Replace with the chat_id you want to update
new_chat_path = './grbrt_spb/'  # Replace with the new chat_path

update_chat_path(database_path, chat_id_to_update, new_chat_path)
