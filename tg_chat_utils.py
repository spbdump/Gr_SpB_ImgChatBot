import requests

BOT_TOKEN = 'YOUR_BOT_TOKEN'
CHAT_ID = 'YOUR_CHAT_ID'

def send_message(text):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    params = {'chat_id': CHAT_ID, 'text': text}
    requests.post(url, params=params)

# send_message('Hello, Telegram!')


def send_reply(chat_id, message_id, text):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    params = {'chat_id': chat_id, 'text': text, 'reply_to_message_id': message_id}
    requests.post(url, params=params)

# send_reply(CHAT_ID, MESSAGE_ID, 'Hello, Telegram!')