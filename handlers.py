from telegram import Update
from telegram.ext import CallbackContext
import requests
import logging

import img_proccessing
import db_utils
# import tg_chat_utils

async def receive_tits_or_cats(update: Update, context: CallbackContext) -> None:

    url = update.message.

    # I'm not shure if it is required
    # check if this message is replay or forward frome same chat 

    # if update.message.chat.id == tg_chat_utils.CHAT_ID
    #     logging.info("This message just replay or forward")
    #     return

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
        message_id = db_utils.get_addtional_data_about_image(img_data.descriptor)
        # add link to existed post
        await update.message.reply_text(text="Предупреждение!\nYou got -rep!")
        return
    
    if len(res) >= 2:
        await update.message.reply_text(text="I got unpredictable result!\nВозможное предупредупреждение!")