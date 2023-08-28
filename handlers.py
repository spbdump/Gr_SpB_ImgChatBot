from telegram import Update, Bot
from telegram.ext import CallbackContext

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import img_proccessing
import db_utils
import tg_chat_utils


import bot_general

async def receive_tits_or_cats(update: Update, context: CallbackContext) -> None:

    # check if this message is a reply or forward from the same chat 
    if update.message.chat.id == tg_chat_utils.CHAT_ID:
        logger.info("This message is just a reply or forward")
        return

    photo_list = update.message.photo

    # Get the last item in the list (the largest photo size)
    photo = photo_list[-1]
    # Get the file ID of the photo
    file_id = photo.file_id
    # Use the file ID to download the photo
    path_to_img = './tmp_photos/img_photo.jpg'

    file = await context.bot.get_file(file_id)
    await file.download_to_drive(path_to_img)

    # update.message.reply_text('Thanks for the photo!')
    
    img_data = img_proccessing.get_image_data(path_to_img)

    res = img_proccessing.poces_similar_sift_descriprors(img_data.descriptor)

    if len(res) == 0:
        db_utils.save_img_data([img_data])
        logging.info("New image was saved to database")
        return

    if len(res) == 1:
        message_id = db_utils.get_addtional_data_about_image(res[0]["_id"])
        # add link to existed post
        await update.message.reply_text(text="Предупреждение!\nYou got -rep!")
        return
    
    if len(res) >= 2:
        await update.message.reply_text(text="I got unpredictable result!\nВозможное предупредупреждение!")


async def receive_tits_or_cats_v2(update: Update, context: CallbackContext) -> None:

    chat_id = update.message.chat.id
    # check if this message is a reply or forward from the same chat 
    if chat_id == tg_chat_utils.CHAT_ID:
        logger.info("This message is just a reply or forward")
        return

    ctx = bot_general.get_chat_ctx(chat_id)

    photo_list = update.message.photo
    # Get the last item in the list (the largest photo size)
    photo = photo_list[-1]
    # Get the file ID of the photo
    # Use the file ID to download the photo
    file_id = photo.file_id

    date_time = update.message.date
    img_id = bot_general.get_next_img_id()

    # should has frormat "photo_{img_id}@{date}-{time}.jpg"
    img_name = f'photo_{img_id}@{date_time}.jpg'
    path_to_img = './photos/' + img_name

    file = await context.bot.get_file(file_id)
    await file.download_to_drive(path_to_img)

    res, img_desc = bot_general.find_image_in_indexes(path_to_img)

    if len(res) == 0:
        message_id = update.message.id
        img_data = {
            "msg_id": message_id,
            "img_id": img_id,
            "img_name": img_name,
            "index_id": 0,
        }

        bot_general.save_img_data(ctx.db_name, img_data)
        bot_general.update_index(ctx.meta_file, img_desc)

        logging.info("New image was saved to database")
        return

    if len(res) == 1:
        img_id = res[0]
        message_id = bot_general.get_message_id(img_id)

        # add link to existed post
        await update.message.reply_text(text="Предупреждение!\nYou got -rep!")
        if message_id != None:
            await Bot.sendMessage(chat_id=chat_id, text="Исходный пост", reply_to_msg_id=message_id)

        return

    if len(res) >= 2:
        await update.message.reply_text(text="I got unpredictable result!\nВозможное предупредупреждение!")