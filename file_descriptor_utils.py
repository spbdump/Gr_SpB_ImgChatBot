import numpy as np
import img_proccessing

import os

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

dtype = np.float32

def write_array_to_file_by_chunks(array, filename, chunk_size):
    """
    Writes a NumPy array to a file in chunks, appending to previous data.
    
    Parameters:
    - array: NumPy array to be written.
    - filename: Name of the file to write to.
    - chunk_size: Size of each chunk to be written.
    """
    with open(filename, 'ab') as file:
        num_elements = len(array)
        for i in range(0, num_elements, chunk_size):
            chunk = array[i:i+chunk_size]
            chunk.tofile(file)


def read_array_from_file_by_chunks(filename: str, chunk_size: int, nfeatures=1, dtype=np.float32):
    """
    Reads a NumPy array from a file in chunks.
    
    Parameters:
    - filename: Name of the file to read from.
    - chunk_size: Size of each chunk to be read.
    - dtype: Data type of the array.
    - nfeatures: Number of features in each chunk.
    
    Yields:
    - Chunks of the NumPy array read from the file.
    """
    desc_lenght = 128
    chunk_width = desc_lenght * nfeatures
    total_chunk_elements = chunk_size * chunk_width

    with open(filename, 'rb') as file:
        while True:
            chunk = np.fromfile(file, dtype=dtype, count=total_chunk_elements)

            if chunk.size == 0:
                break
            elif chunk.size != total_chunk_elements:
                last_chunk_size = int( chunk.size / chunk_width )
                chunk = chunk.reshape(last_chunk_size, chunk_width)
            else:
                chunk = chunk.reshape(chunk_size, chunk_width)

            yield chunk


def append_array_with_same_width(file_path, new_array: np.ndarray):
    """
    Appends a new NumPy array with the same width to an existing file.
    
    Parameters:
    - file_path: Path to the existing file to which you want to append.
    - new_array: NumPy array to append.
    """
    logger.info("%d images append to file %s", len(new_array), file_path)
    # Append the new array to the file
    with open(file_path, 'ab') as file:
        new_array.tofile(file)

    logger.info("Images data was saved to file")


def get_desc_batch(batch_offset:int, batch_size:int, raw_data:bool = False):

    descriptors = read_array_from_file_by_chunks()
    if len(descriptors) == 0:
        return []

    first_nfeatures = NFEATURES
    if not raw_data:
        np_descriptors = np.empty((0, first_nfeatures*128))
        for desc in descriptors:
            np_desc_nf = np.array(desc["descriptor"], dtype=np.float32)[:first_nfeatures]
            padded_matrix = np_desc_nf

            if np_desc_nf.shape[0] < first_nfeatures:
                n_rows = first_nfeatures - np_desc_nf.shape[0]
                padded_matrix = np.pad(np_desc_nf, pad_width=((0, n_rows), (0, 0)), mode='constant')

            np_desc = padded_matrix.reshape(1, -1)
            np_descriptors = np.concatenate((np_descriptors, np_desc), axis=0)

        logger.info("Got converted descriptors data from 'image_data_collection'")
        return np_descriptors

    logger.info("Got raw image data from 'image_data_collection'")
    return descriptors

def write_images_list_to_file(file_path, array_of_strings):
    with open(file_path, "w") as file:
        for index, element in enumerate(array_of_strings):
            file.write(f"{index} {element}\n")

def get_image_files(directory: str):
    image_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_files.append(filename)
    return image_files


def fullfill_desc_file(path_to_imgs_dir: str, desc_file: str, 
                       max_imgs_cnt:int, chunk_size:int,
                       path_to_imgs_data="./descriptors/imgs_data.txt"):

    list_imgs = get_image_files(path_to_imgs_dir)
    if max_imgs_cnt > 0 and max_imgs_cnt < len(list_imgs) :
        list_imgs = list_imgs[:max_imgs_cnt]

    # debug
    # write_images_list_to_file(path_to_imgs_data, list_imgs)

    # if chunk_size too big may be need divide external loop !!! because take a lot of RAM
    first_nfeatures = 1000
    bad_img_counter = 0
    curr_idx = 0
    last_idx = 0
    logger.info("Proccessing %d images", len(list_imgs))

    while curr_idx < len(list_imgs):
        if curr_idx + chunk_size < len(list_imgs):
           last_idx = curr_idx + chunk_size 
        else:
            last_idx = len(list_imgs)

        imgs_data = np.array([], dtype=np.float32).reshape(0, 128*first_nfeatures)
        for img_path in list_imgs[curr_idx:last_idx]:
            img_data = img_proccessing.get_image_data(path_to_imgs_dir + '/' + img_path)

            if img_data.descriptor.shape[0] < first_nfeatures:
                logger.info("Bad image feature detection: %s", img_path)
                logger.info("descriptor features len: %s", img_data.descriptor.shape[0])
                bad_img_counter = bad_img_counter + 1
                continue

            desc = img_data.descriptor[:first_nfeatures].reshape(1,-1)
            imgs_data = np.vstack((imgs_data, desc), dtype=np.float32)

        append_array_with_same_width(desc_file, imgs_data)
        np.savetxt("./descriptors/test_descriptors.txt", imgs_data, fmt="%.2f")

        curr_idx += chunk_size

    logger.info("Batch fullfilled. Processed %d images", curr_idx)
    logger.info("Bad images %d", bad_img_counter)
    logger.info("Index has %.2f%% of banch images", float(100.0 - (bad_img_counter*100)/len(list_imgs)) )


def read_specific_rows_from_file(file_path : str, row_indices, num_columns:int, chunk_size=10000, dtype=np.float32):
    """
    Reads specific rows from a NumPy array in a file using memory-mapped arrays.
    
    Parameters:
    - file_path: Path to the file containing the NumPy array.
    - row_indices: List of row indices to read.
    - dtype: Data type of the array (default is np.float32).
    - chunk_size: Size of each chunk to be read (default is 10000).
    - num_columns: Number of columns in the original array (default is 128).
    
    Returns:
    - NumPy array containing the specified rows.
    """
    result = []
    with open(file_path, 'rb') as file:
        for start_idx in range(0, len(row_indices), chunk_size):
            end_idx = min(start_idx + chunk_size, len(row_indices))
            chunk_indices = row_indices[start_idx:end_idx]
            
            # Create a memory-mapped array for the chunk
            mmap_array = np.memmap(file, dtype=dtype, mode='r', shape=(len(chunk_indices), num_columns))
            
            # Retrieve the rows using the indices
            chunk = mmap_array
            result.extend(chunk)
            
    return np.array(result, dtype=dtype)
