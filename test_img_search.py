import bot_general

def main():
    path_to_img = './grbrt_spb/photos/photo_25@04-03-2022_02-15-28.jpg'
    path_prefix = './grbrt_spb/'
    nfeatures = 700
    res, img_desc = bot_general.find_image_in_indexes(path_to_img, path_prefix, nfeatures)

    print( res )


if __name__ == "__main__":
    main()