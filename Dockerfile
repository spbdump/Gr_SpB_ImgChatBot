FROM deb12-py-opencv:latest

RUN apt update
RUN apt install sqlite3

#install other requirements
COPY requirements.docker.py-3.10.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /bot/sources

COPY bot.py handlers.py commands.py \
     random_name.py bot_impl.py ./
COPY core/ core/
COPY model/ model/

CMD [ "python", "/bot/sources/bot.py" ]