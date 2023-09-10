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