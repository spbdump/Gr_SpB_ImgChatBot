from sqlitedict import SqliteDict

image_table = "img_data"

def store_img_data(db_name:str, data):
    db = SqliteDict(db_name, tablename=image_table)

    for key, v in data.items():
        db[key] = v
    
    db.commit()
    db.close()

def get_context(chat_id):
    db = SqliteDict('./context.db')
    ctx = db[chat_id]
    db.close()

    return ctx

def get_last_img_record(path_to_db):
    db = SqliteDict(path_to_db, tablename=image_table)
    last_record = db[ len(db) - 1 ]

    db.close()

    return last_record.key + 1

import re
import json

def parse_img_txt_data():
    with open('./grbrt_spb/imgs_data.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    parsed_data = []

    for line in lines:
        data = line.strip().split(', ')
        t_msg_id_match = re.search(r'-?\d+', data[0])
        if t_msg_id_match:
            t_msg_id = int(t_msg_id_match.group())
        else:
            t_msg_id = None
        img_name = data[1].split(': ')[-1].strip("'").replace('photos/', '')

        parsed_data.append({'t_msg_id': t_msg_id, 'img_name': img_name})

    # Write parsed data to a JSON file
    output_file = './grbrt_spb/parsed_data.json'
    with open(output_file, 'w') as json_file:
        json.dump(parsed_data, json_file, indent=4)
