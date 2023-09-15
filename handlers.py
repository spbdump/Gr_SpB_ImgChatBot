from telegram import Update, Bot, ChatMemberUpdated, Chat, ChatMember
from telegram.ext import (
    CallbackContext,
    ContextTypes,
)

from typing import Optional, Tuple

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import bot_general
import random_name


VOLUME_PATH = ""

def update_VOLUME_PATH(path):
    global VOLUME_PATH
    VOLUME_PATH = path
    bot_general.update_DBPATH(path)

async def receive_tits_or_cats_v2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    chat = update.effective_chat
    m_chat_id = chat.id
    ctx = bot_general.get_chat_ctx(m_chat_id)

    if ctx == None:
        logger.error("Bad state: can't get context for chat_id", m_chat_id)
        return
    
    chat_path = VOLUME_PATH + ctx.chat_path
    nfeatures = ctx.nfeatures
    #c_chat_id = ctx.chat_id
    # check if this message is a reply or forward from the same chat 
    # if m_chat_id == c_chat_id:
    #     logger.info("This message is just a reply or forward")
    #     return

    ef_message = update.effective_message
    photo_list = ef_message.photo
    # Get the last item in the list (the largest photo size)
    photo = photo_list[-1]
    # Get the file ID of the photo
    file_id = photo.file_id

    date_time = ef_message.date
    img_id = bot_general.generate_next_img_id( chat_path )

    # should has frormat "photo_{img_id}@{date}-{time}.jpg"
    img_name = f'photo_{img_id}@{date_time}.jpg'
    path_to_img = chat_path + '/tmp/' + img_name

    # Use the file ID to download the photo
    file = await context.bot.get_file(file_id)
    await file.download_to_drive(path_to_img)

    res, img_desc = bot_general.find_image_in_indexes(path_to_img, chat_path, m_chat_id, nfeatures)

    if img_desc.shape[0] < nfeatures:
        logger.info("Image has %d features, should be %d", img_desc.shape[0], nfeatures)
        await ef_message.reply_text(text="Can't calculate enough features. Image wasn't indexed!\n")
        return

    if len(res) == 0:
        message_id = ef_message.id
        bot_general.update_index( ctx, chat_path, img_desc, img_name, message_id )
        logging.info("New image was saved to database")
        #await ef_message.reply_text(text="Image was indexed!\n")
        return

    if len(res) == 1:
        index_id, img_id = res[0]
        img_id = img_id[0]
        message_id = bot_general.get_message_id(chat_path, m_chat_id, img_id, index_id)

        # add link to existed post
        await ef_message.reply_text(text="Предупреждение!\nYou got -rep!")

        if message_id != None:
            await context.bot.send_message(chat_id=m_chat_id, text="Исходный пост", reply_to_message_id=message_id)

        return

    if len(res) >= 2:
        await ef_message.reply_text(text="I got unpredictable result!\nВозможное предупредупреждение!")


def extract_status_change(chat_member_update: ChatMemberUpdated) -> Optional[Tuple[bool, bool]]:
    """Takes a ChatMemberUpdated instance and extracts whether the 'old_chat_member' was a member
    of the chat and whether the 'new_chat_member' is a member of the chat. Returns None, if
    the status didn't change.
    """
    status_change = chat_member_update.difference().get("status")
    old_is_member, new_is_member = chat_member_update.difference().get("is_member", (None, None))

    if status_change is None:
        return None

    old_status, new_status = status_change
    was_member = old_status in [
        ChatMember.MEMBER,
        ChatMember.OWNER,
        ChatMember.ADMINISTRATOR,
    ] or (old_status == ChatMember.RESTRICTED and old_is_member is True)
    is_member = new_status in [
        ChatMember.MEMBER,
        ChatMember.OWNER,
        ChatMember.ADMINISTRATOR,
    ] or (new_status == ChatMember.RESTRICTED and new_is_member is True)

    return was_member, is_member


async def track_chats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tracks the chats the bot is in."""
    if update.my_chat_member == None:
        return

    result = extract_status_change(update.my_chat_member)
    if result is None:
        return

    was_member, is_member = result

    # Let's check who is responsible for the change
    cause_name = update.effective_user.full_name

    # Handle chat types differently:
    chat = update.effective_chat
    if chat.type == Chat.PRIVATE:

        # this is not iterested case for me !!!
        # if not was_member and is_member:
        #     # This may not be really needed in practice because most clients will automatically
        #     # send a /start command after the user unblocks the bot, and start_private_chat()
        #     # will add the user to "user_ids".
        #     # We're including this here for the sake of the example.
        #     logger.info("%s unblocked the bot", cause_name)
        #     context.bot_data.setdefault("user_ids", set()).add(chat.id)
        # elif was_member and not is_member:
        #     logger.info("%s blocked the bot", cause_name)
        #     context.bot_data.setdefault("user_ids", set()).discard(chat.id)

        pass

    elif chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        if not was_member and is_member:
            logger.info("%s added the bot to the group %s", cause_name, chat.title)
            context.bot_data.setdefault("group_ids", set()).add(chat.id)

            bot_general.on_add_bot(chat.id, VOLUME_PATH, random_name.get_random_name())

        elif was_member and not is_member:
            logger.info("%s removed the bot from the group %s", cause_name, chat.title)
            context.bot_data.setdefault("group_ids", set()).discard(chat.id)

            bot_general.on_remove_bot(chat.id, VOLUME_PATH)

    elif not was_member and is_member:
        logger.info("%s added the bot to the channel %s", cause_name, chat.title)
        context.bot_data.setdefault("channel_ids", set()).add(chat.id)

        bot_general.on_add_bot(chat.id, VOLUME_PATH, random_name.get_random_name())

    elif was_member and not is_member:
        logger.info("%s removed the bot from the channel %s", cause_name, chat.title)
        context.bot_data.setdefault("channel_ids", set()).discard(chat.id)

        bot_general.on_remove_bot(chat.id, VOLUME_PATH)