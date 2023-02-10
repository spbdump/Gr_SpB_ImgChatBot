import requests
import telegram
import os

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = '-1001685634092'

async def init_chat_id():
    if BOT_TOKEN == None :
        return

    bot = telegram.Bot(BOT_TOKEN)

    updates = await bot.get_updates()

    # Get the last update
    if updates == None :
        return

    last_update = updates[-1]
    last_message = last_update.message
    chat_id = last_message.chat.id
    global CHAT_ID
    CHAT_ID = chat_id

    # Print the chat ID
    print(chat_id)

def send_message(text):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    global CHAT_ID
    params = {'chat_id': CHAT_ID, 'text': text}
    requests.post(url, params=params)

def send_reply(chat_id, message_id, text):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    params = {'chat_id': chat_id, 'text': text, 'reply_to_message_id': message_id}
    requests.post(url, params=params)

# send_reply(CHAT_ID, MESSAGE_ID, 'Hello, Telegram!')