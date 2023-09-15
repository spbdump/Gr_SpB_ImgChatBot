FROM python:3.10-alpine

# install cmake and compiler gcc and headers
RUN apk --no-cache add cmake ccache build-base libc-dev musl-dev linux-headers g++

#install requirements
COPY requirements.docker.txt requirements.txt
RUN pip install -r requirements.txt

# build opencv
# endable contrib modules
RUN apk --no-cache add git
RUN git clone --recursive https://github.com/skvark/opencv-python.git

ENV CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON"
RUN python /opencv-python/setup.py bdist_wheel
RUN pip install /opencv-python/dist/opencv_python-4.8.0.76-cp310-cp310-linux_x86_64.whl

WORKDIR /bot

COPY bot.py handlers.py commands.py \
     context.py HNSW_index.py index.py \
     file_descriptor_utils.py image_d.py \
     img_proccessing.py sqlite_db_utils.py \
     random_name.py bot_general.py /bot/sources/

CMD [ "python", "/bot/sources/bot.py" ]