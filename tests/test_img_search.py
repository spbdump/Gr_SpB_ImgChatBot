import bot_general

def main():
    path_to_img = './grbrt_spb/photos/'+'photo_1099@14-08-2022_14-06-36.jpg'
    path_prefix = './grbrt_spb/'
    nfeatures = 1000
    res, img_desc = bot_general.find_image_in_indexes(path_to_img, path_prefix, nfeatures)

    print( res )


if __name__ == "__main__":
    main()