import logging
import telegram
from telegram.error import NetworkError, TelegramError
from time import sleep
import os
import asyncio

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

# Define the Bot API token
BOT_TOKEN = os.environ.get("BOT_TOKEN")

async def get_chat_chat_id():
    if BOT_TOKEN == None :
        return

    bot = telegram.Bot(BOT_TOKEN)

    updates = await bot.get_updates()

    # Get the last update
    if updates == None :
        print("FUCk UP")
        return

    last_update = updates[-1]

    # Get the last message
    last_message = last_update.message

    chat_id = last_message.chat.id

    # Print the chat ID
    print(chat_id)


def main():
    """Run the bot."""
    print("TOKEN ID :", BOT_TOKEN)
    if BOT_TOKEN == None :
        return

    asyncio.run(get_chat_chat_id())

if __name__ == '__main__':
    main()
