## Simple Telegram bot style transfer

### Описание

Простой телеграм бот
основная архитектура модели взята https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer
есть две модели (файлы net.py и net2.py, с небольшими модификациями),
веса были также взяты https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer
(тренировачный скрипт в оригинальном репо, но надо иметь ввиду, как оказалось это достаточно ресурсоемкая задача.)

Боту @phtstyler_bot нужно последовательно отправить 2 фото 

###

для запуска бота нужно в папку config , положить main.json, с ключом API_TOKEN и значением токена
затем установить все зависимости и запустить main.py