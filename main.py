import img_proccessing
import tg_chat_utils
import os
import cv2
import asyncio

import get_imgs_from_new_posts
import download_prev_imgs

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

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

def main():

    

    # asyncio.run(tg_chat_utils.init_chat_id())
    # download_prev_imgs.get_images_from_chat(tg_chat_utils.CHAT_ID)

    app = Application.builder().token(tg_chat_utils.BOT_TOKEN).build()

    app.add_handler(CommandHandler("tits", commands.tits))
    # pic_filter = filters.PHOTO | 
    app.add_handler(MessageHandler(filters.PHOTO, handlers.receive_tits_or_cats))

    app.run_polling()

    # tg_chat_utils.send_message("Всем предупреждение!")


if __name__ == "__main__":
    main()
