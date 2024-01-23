from telegram import Update, Bot, ChatPermissions
from telegram.ext import ContextTypes
from datetime import datetime, timedelta

async def tits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user

    if user.name == "vmalkov":
        await update.message.reply_text(text="GAY")

    await update.message.reply_text(text="MEOW")

async def cats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user

    await update.message.reply_text(text="BOOPS")

async def ban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_message.chat_id
    # user_ids = []
    user_id =  context.args[0]
    permissions = ChatPermissions(
        can_send_messages=False,
        can_send_media_messages=False,
        can_send_other_messages=False,
    )
    new_datetime = datetime.now() + timedelta(minutes=5)
    await Bot.restrictChatMember(chat_id, user_id, permissions, until_date=new_datetime)