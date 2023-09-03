import bot_general

def main():
    path_to_img = './grbrt_spb/photos/'+'photo_3342@22-05-2023_10-26-34.jpg'
    path_prefix = './grbrt_spb/'
    nfeatures = 1000
    res, img_desc = bot_general.find_image_in_indexes(path_to_img, path_prefix, nfeatures)

    print( res )


if __name__ == "__main__":
    main()