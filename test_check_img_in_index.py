import bot_general


def main():
    path_prefix = './tests/'
    prefix_img_path = './grbrt_spb/photos/'
    nfeatures = 800

    img_list = [
        "photo_1187@08-09-2022_10-26-24.jpg",
    ]

    for img_name in img_list:
        path_to_img = prefix_img_path + img_name
        res, img_desc = bot_general.find_image_in_indexes(path_to_img, path_prefix, nfeatures)
        print( res )

if __name__ == "__main__":
    main()