import logging
import telegram
from telegram.error import NetworkError, TelegramError
from time import sleep
import os
import asyncio
from pymongo import MongoClient

import image_d
import cv2
import img_proccessing
import db_utils
import build_index

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

# Define the Bot API token
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DB_NAME = os.environ.get("DB_NAME")
DB_ADDRES = os.environ.get("DB_ADDRES")


def main():
    """Run the bot."""

    logger.info("Test logger!")

    image_data = img_proccessing.get_image_data("./photos/photo_1@04-03-2022_01-26-02_thumb.jpg")
    # image_data2 = img_proccessing.get_image_data("./photos/photo_1@04-03-2022_01-26-02.jpg")
    # db.create_collection("")
    # db_utils.save_img_data([image_data, image_data2])

    # step 1
    # build_index.build_index("./photos")

    print("db has same image: ", img_proccessing.poces_similar_sift_descriprors(image_data.descriptor) )



if __name__ == '__main__':
    main()
