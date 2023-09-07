
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

import commands
import handlers

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

def main():

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("tits", commands.tits))
    app.add_handler(MessageHandler(filters.PHOTO, handlers.receive_tits_or_cats_v2))

    app.run_polling()


if __name__ == "__main__":
    main()
