FROM python:3.10-slim-bookworm

RUN apt update

# install build tools
RUN apt install -y cmake ccache \
    build-essential gcc g++

# open-cv dependencies
RUN apt install -y \
    python3-dev python3-numpy

RUN apt install -y libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev

RUN apt install -y libtbb-dev libeigen3-dev
ENV ENABLE_TBB=" -D WITH_TBB=ON"
ENV ENABLE_EIGEN=" -D WITH_EIGEN=ON"

RUN apt install -y git
RUN git clone --recursive https://github.com/opencv/opencv.git
# endable contrib modules
ENV ENABLE_NONFREE=" -D OPENCV_ENABLE_NONFREE=ON"
ENV DISABLE_TESTS ="-D BUILD_TESTS=OFF"
ENV CMAKE_ARGS="${ENABLE_NONFREE}${ENABLE_TBB}${ENABLE_EIGEN}${DISABLE_TESTS}"

RUN pip install numpy

#setup build type and path
#-D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local

# build opencv
RUN  mkdir -p /opencv/build
WORKDIR /opencv/build
RUN cmake "${CMAKE_ARGS}" ../
RUN make -j8 && make install

