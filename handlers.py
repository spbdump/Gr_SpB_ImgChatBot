from telegram import Update
from telegram.ext import CallbackContext

async def receive_tits_or_cats(update: Update, context: CallbackContext) -> None:

    user = update.effective_user
    if user.name == "vmalkov":
        await update.message.reply_text(text="GAY")

    await update.message.reply_text(text="MEOW")