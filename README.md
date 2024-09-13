# Gr_SpB_ImgChatBot


Store in db - 
 * desc - descriptor of image
 * id - unique identifer
 * post_id - id of post in telegramm chat (to put it in a chat if match is founded)
 * 

 $Env:BOT_TOKEN = 'TOKEN' - for powershell

## Tests

to start tests : python -m pytest tests/

## OpenCV

git clone --recursive https://github.com/skvark/opencv-python.git
cd opencv-python
export CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON"
python setup.py bdist_wheel

## Hardvare requirements:
RAM 1gb or 512mb - tveakable option
solid memory ~ 1gb per 3000 images
proc intel-seleron


# Research
jupyter lab

TODO: switch to FLAAN computor!!

Helpful techniques:
    Index Pruning
    Cluster-Based Indexing


# Steps
1. build docker manylinux with open-cv
2. copy build from builder to my image


# build image
docker build -t opencv-test-image -f .\Dockerfile.opencv.builder .
# run containter
docker run -it --name test-py-opencv opencv-test-image bash
# go to container
docker ps
docker exec -it <CONTAINER_ID_OR_NAME> bash



docker build -t deb12-py-opencv -f .\Dockerfile.local.dev .
docker build -t spb-img-bot -f .\Dockerfile.dev .
docker run --env-file .env -it --name spb-img-bot spb-img-bot bash

