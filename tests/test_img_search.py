import bot_general

def test_find_img():
    path_to_img = './grbrt_spb/photos/'+'photo_1099@14-08-2022_14-06-36.jpg'
    path_prefix = './'
    nfeatures = 800
    res, img_desc = bot_general.find_image_in_indexes(path_to_img, path_prefix, nfeatures)

    print( res )