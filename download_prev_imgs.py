import requests
import tg_chat_utils

prefix_path = './archive_imgs/'

def get_file_path(file_id):
    url = f'https://api.telegram.org/bot{tg_chat_utils.BOT_TOKEN}/getFile?file_id={file_id}'
    response = requests.get(url)
    file_path = response.json()['result']['file_path']
    return file_path

def download_image(file_path):
    url = f'https://api.telegram.org/file/bot{tg_chat_utils.BOT_TOKEN}/{file_path}'
    response = requests.get(url)
    if not response.ok :
        print(url)
        print(response.content)
        print("Can't get image")
        return

    with open(file_path, 'wb') as f:
        f.write(response.content)

def get_images_from_chat(chat_id):
    url = f'https://api.telegram.org/bot{tg_chat_utils.BOT_TOKEN}/getUpdates'
    params = {'offset': -1000, 'limit': 2000, 'timeout': 5}
    response = requests.get(url, params=params)
    print(response.text)
    messages = response.json()['result']
    images = []
    for message in messages:
        # print(message)
        if message['message']['chat']['id'] == chat_id:
            if 'photo' in message['message']:
                photo = message['message']['photo'][-1]
                file_id = photo['file_id']
                file_path = get_file_path(file_id)
                print(file_path)
                download_image(file_path)
                images.append(file_path)
    return images
