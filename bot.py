import os

from telegram.ext import (
    Application,
    CommandHandler,
    ChatMemberHandler,
    MessageHandler,
    filters,
)

import commands
import handlers
import bot_impl as bi

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():

    BOT_TOKEN = os.environ.get('BOT_TOKEN')
    v_path = os.environ.get('VOLUME_PATH')

    if BOT_TOKEN == None:
        logger.error("Can't read bot token env")
        return

    if v_path != None:
        handlers.update_VOLUME_PATH(v_path)

    bi.init()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("tits", commands.tits))
    app.add_handler(CommandHandler("ban", commands.ban))

    app.add_handler(MessageHandler(filters.PHOTO, handlers.receive_tits_or_cats_v2))
    app.add_handler(ChatMemberHandler(handlers.track_chats, ChatMemberHandler.MY_CHAT_MEMBER))
    app.add_handler(MessageHandler(filters.ALL, handlers.track_stickers))

    app.run_polling()


if __name__ == "__main__":
    main()
