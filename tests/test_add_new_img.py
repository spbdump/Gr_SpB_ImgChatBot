import bot_general
import model.context as context

import utils

PATH_TO_IMGS = './grbrt_spb/photos/'
N_RAND_IMGS = 25

import helper.debug_utils as debug_utils

def main():
    nfeatures = 800
    chat_path = './tests/'
    prefix_img_path = PATH_TO_IMGS
    list_imgs = utils.get_image_files( prefix_img_path )
    rand_imgs_list = utils.get_random_images(list_imgs, 10)

    for img_name in rand_imgs_list:
        path_to_img = prefix_img_path + img_name
        
        ctx = context.Context(nfeatures, 128, 20, chat_path, -1)
        img_data = bot_general.get_image_data(path_to_img, nfeatures)
        img_desc = img_data.descriptor

        if img_desc.shape[0] < nfeatures:
            continue

        # move this check to get_imag_data
        if img_desc.shape[0] > nfeatures:
            img_desc = img_desc[:nfeatures]

        # should be before saving img data to db
        # because can create new index

        # DEBUG
        desc_data_to_file = img_desc.reshape(-1)[:30]
        debug_utils.append_row_to_txt(desc_data_to_file)
        # DEBUG

        # b_updated = bot_general.update_index( img_desc, ctx )
        # if not b_updated:
        #     continue
    
        # message_id = -1
        # bot_general.save_img_data(chat_path, img_name, message_id)

        print("Image to add: ", img_name)

    return

if __name__ == "__main__":
    main()