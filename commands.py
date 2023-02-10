from telegram import Update
from telegram.ext import ContextTypes
import tg_chat_utils
import asyncio

async def tits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user

    if user.name == "vmalkov":
        await update.message.reply_text(text="GAY")

    await update.message.reply_text(text="MEOW")
