import tg_chat_utils


from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

import commands
import handlers
import HNSW_index

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

def main():

    app = Application.builder().token(tg_chat_utils.BOT_TOKEN).build()

    # load index on each bot start
    HNSW_index.load_hnsw_indexies()

    app.add_handler(CommandHandler("tits", commands.tits))
    app.add_handler(MessageHandler(filters.PHOTO, handlers.receive_tits_or_cats))

    app.run_polling()


if __name__ == "__main__":
    main()
