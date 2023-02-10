import requests

BOT_TOKEN = 'YOUR_BOT_TOKEN'
CHAT_ID = 'YOUR_CHAT_ID'
LAST_UPDATE_ID = 0

def get_file_path(file_id):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}'
    response = requests.get(url)
    file_path = response.json()['result']['file_path']
    return file_path

def download_image(file_path):
    url = f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}'
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

def get_new_messages(last_update_id):
    global LAST_UPDATE_ID
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates'
    params = {'offset': last_update_id + 1, 'limit': 100, 'timeout': 0}
    response = requests.get(url, params=params)
    messages = response.json()['result']
    for message in messages:
        update_id = message['update_id']
        LAST_UPDATE_ID = max(LAST_UPDATE_ID, update_id)
        if message['message']['chat']['id'] == CHAT_ID:
            if 'photo' in message['message']:
                photo = message['message']['photo'][-1]
                file_id = photo['file_id']
                file_path = get_file_path(file_id)
                download_image(file_path)

# while True:
#     get_new_messages(LAST_UPDATE_ID)
