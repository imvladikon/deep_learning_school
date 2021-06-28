import os
import logging
import gc
import io
import django
from aiogram import Bot, Dispatcher, executor, types
from const import ROOT_DIR
from gan.net2 import *
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton, ParseMode
import json

from gan.utils import apply_style_transfer

logging.basicConfig(level=logging.INFO)


API_TOKEN = json.load(open(ROOT_DIR/"config"/"main.json"))["API_TOKEN"]

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

IS_BEGINNING = True

weight_folder = ROOT_DIR / "gan" / "weights"

style_model = Net(ngf=128)
style_model.load_state_dict(torch.load(weight_folder / "pretrained2.pt"), False)



@dp.message_handler(content_types=['new_chat_members'])
async def starting_message(message: types.Message):
    username = message.new_chat_members[0]['username']

    start_msg = f"""Привет, {username}! я бот по переносу стиля из одного изображения на другое.
                 Отправь мне две фотографии, сначала первую, а потом вторую
                 и я продемонстрирую свою мощь.
                 Доступные команды /help"""

    await bot.send_message(message.chat.id, start_msg, parse_mode=ParseMode.HTML)


@dp.message_handler(commands=['help'])
async def help_message(message: types.Message):
    await message.answer(text="Привет! я простой бот, мне дают парочку изображений, а я переношу стиль с одного на другое."
                              "отправь мне парочку изображений(последовательно)")


@dp.message_handler(content_types=["photo", "file", "document"])
async def photo_processing(message):

    global IS_BEGINNING

    file_info = None
    if message.content_type=="document":
        file_info = message.document
    elif message.content_type == "file":
        file_info = message.file
    else:
        file_info = message.photo

    if isinstance(file_info, list):
        file_info = file_info[-1]

    if IS_BEGINNING:
        await file_info.download('content.jpg')
        await message.answer(text='Получено первое изображение, пришли мне, пожалуйста еще одно.')
        IS_BEGINNING = False
    else:
        IS_BEGINNING = True
        await file_info.download('style.jpg')
        await message.answer(text='Получено второе изображение. Начата нейронная магия.',
                             reply_markup=types.ReplyKeyboardRemove())

        with io.BytesIO() as byte_io:
            img = apply_style_transfer(style_model, 'content.jpg', 'style.jpg', im_size=300)
            img.save(byte_io, format="JPEG")
            jpg_buffer = byte_io.getvalue()
            await message.answer_photo(jpg_buffer, caption='Voila!')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)