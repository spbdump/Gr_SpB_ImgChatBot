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

    # print("Check base functionality")
    # path = './test_images/'
    # img2 = cv2.imread(path+'photo_2023-02-09_02-17-55.jpg')
    # img1 = cv2.imread(path+'photo_2023-02-09_02-17-55_copy.jpg')
    # img3 = cv2.imread(path+'photo_2023-02-08_19-24-07.jpg ')
    
    # # Check if the image was loaded successfully
    # if img2 is None:
    #     print("Error: Could not load the image.")
    #     return

    # print("Is same imgs: ", img_proccessing.compare_images_sift(img1, img2) )
    # print("Is same imgs: ", img_proccessing.compare_images_sift(img1, img3) )

    # tg_chat_utils.send_message("Всем предупреждение!")


if __name__ == "__main__":
    main()
