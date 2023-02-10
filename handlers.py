from telegram import Update
from telegram.ext import CallbackContext
import requests
import logging

import img_proccessing
import db_utils


async def receive_tits_or_cats(update: Update, context: CallbackContext) -> None:

    url = update.message.

    response = requests.get(url)
    if not response.ok :
        print(url)
        print(response.content)
        print("Can't get image")
        return

    with open(path_to_img, 'wb') as f:
        f.write(response.content)
    
    img_data = img_proccessing.get_image_data(path_to_img)

    res = img_proccessing.poces_similar_sift_descriprors(img_data.descriptor)

    if len(res) == 0:
        db_utils.save_img_data([img_data])
        logging.info("New image was stored to database")
        return

    if len(res) == 1:
        await update.message.reply_text(text="Предупреждение!\nYou got -rep!")
    
    if len(res) >= 2:
        await update.message.reply_text(text="I got unpredictable result!\nВозможное предупредупреждение!")