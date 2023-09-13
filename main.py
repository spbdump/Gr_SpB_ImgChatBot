import os

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

import commands
import handlers
import index as rn_index

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():

    BOT_TOKEN = os.environ.get('BOT_TOKEN')

    if BOT_TOKEN == None:
        logger.error("Can't read bot token env")
        return
    
    rn_index.init_runtime_chat_indexes()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("tits", commands.tits))
    app.add_handler(MessageHandler(filters.PHOTO, handlers.receive_tits_or_cats_v2))

    app.run_polling()


if __name__ == "__main__":
    main()
