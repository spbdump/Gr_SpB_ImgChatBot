import numpy as np
import img_proccessing

import os

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


def read_array_from_file_by_chunks(filename, chunk_size, dtype):
    """
    Reads a NumPy array from a file in chunks.
    
    Parameters:
    - filename: Name of the file to read from.
    - chunk_size: Size of each chunk to be read.
    - dtype: Data type of the array.
    
    Returns:
    - NumPy array containing the data read from the file.
    """
    result = []
    with open(filename, 'rb') as file:
        while True:
            chunk = np.fromfile(file, dtype=dtype, count=chunk_size)
            if not chunk.size:
                break
            result.extend(chunk)
    return np.array(result, dtype=dtype)

def append_array_with_same_width(file_path, new_array):
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


# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_image_files(directory: str):
    image_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_files.append(filename)
    return image_files


def fullfill_desc_file(path_to_imgs_dir: str, desc_file: str):


    list_imgs = get_image_files(path_to_imgs_dir)
    max_iter_size = HNSW_index.BATCH_SIZE # should be same with size of index

    # if max_iter_size too big may be need divide external loop !!! because take a lot of RAM

    curr_idx = 0
    last_idx = 0
    logger.info("Proccessing %d images", len(list_imgs))

    curr_batch_idx = 0
    curr_batch_idx_in = 0

    while curr_idx < len(list_imgs):
        if curr_idx + max_iter_size < len(list_imgs):
           last_idx = curr_idx + max_iter_size 
        else:
            last_idx = len(list_imgs) - 1

        imgs_data = []
        for img_path in list_imgs[curr_idx:last_idx]:
            img_data = img_proccessing.get_image_data(path_to_imgs_dir + '/' + img_path)
            img_data.batch_id = curr_batch_idx
            img_data.batch_id_in = curr_batch_idx_in
            imgs_data.append(img_data)
            curr_batch_idx_in += 1

            if len(imgs_data) > 300:
                append_array_with_same_width(desc_file, imgs_data)
                imgs_data = []

        append_array_with_same_width(desc_file, imgs_data)


        curr_idx += max_iter_size

        if curr_batch_idx_in == HNSW_index.BATCH_SIZE:
            logger.info("Batch fullfilled. Current id: %d", curr_batch_idx)
            curr_batch_idx += 1
            curr_batch_idx_in = 0

def read_specific_rows_from_file(file_path, row_indices, chunk_size=10000):
    """
    Reads specific rows from a NumPy array in a file using memory-mapped arrays.
    
    Parameters:
    - file_path: Path to the file containing the NumPy array.
    - row_indices: List of row indices to read.
    - dtype: Data type of the array.
    - chunk_size: Size of each chunk to be read (default is 1000).
    
    Returns:
    - NumPy array containing the specified rows.
    """
    result = []
    with open(file_path, 'rb') as file:
        for start_idx in range(0, len(row_indices), chunk_size):
            end_idx = start_idx + chunk_size
            chunk_indices = row_indices[start_idx:end_idx]
            
            # Create a memory-mapped array for the chunk
            mmap_array = np.memmap(file, dtype=dtype, mode='r', shape=(len(chunk_indices),))
            
            # Retrieve the rows using the indices
            chunk = mmap_array[chunk_indices - min(chunk_indices)]
            result.extend(chunk)
            
            # Delete the memory-mapped array
            del mmap_array
            
    return np.array(result, dtype=dtype)
