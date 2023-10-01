import numpy as np
import core.img_proccessing as img_proccessing

import json
import re
import os

from core.sqlite_db_utils import store_img_data

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


def read_array_from_file_by_chunks(filename: str, chunk_size: int, cnt_chunks: int, nfeatures: int, offset: int = 0, dtype=np.float32):
    """
    Reads a NumPy array from a file in chunks.
    
    Parameters:
    - filename: Name of the file to read from.
    - chunk_size: Size of each chunk to be read.
    - cnt_chunks: Total number of chunks to read.
    - nfeatures: Number of features in each chunk.
    - offset: Offset in rows to start reading from (default is 0).
    - dtype: Data type of the array.
    
    Yields:
    - Chunks of the NumPy array read from the file.
    """
    desc_length = 128
    chunk_width = desc_length * nfeatures
    total_chunk_elements = chunk_size * chunk_width
    total_elements_to_skip = offset * chunk_width

    with open(filename, 'rb') as file:
        element_size = np.dtype(dtype).itemsize
        bytes_to_skip = total_elements_to_skip * element_size

        for _ in range(cnt_chunks):
            chunk = np.fromfile(file, dtype=dtype, count=total_chunk_elements, offset=bytes_to_skip)

            if chunk.size == 0:
                break
            elif chunk.size != total_chunk_elements:
                last_chunk_size = int(chunk.size / chunk_width)
                chunk = chunk.reshape(last_chunk_size, chunk_width)
            else:
                chunk = chunk.reshape(chunk_size, chunk_width)

            yield chunk


def append_array_with_same_width(file_path: str, new_array: np.ndarray):
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


def extract_image_id(img_name):
    # Extract the image ID using regular expression
    match = re.search(r'photo_(\d+)', img_name)
    if match:
        image_id = int(match.group(1))
        return image_id
    else:
        return None

def fullfill_desc_file(path_to_dir: str, list_imgs, desc_file: str,
                       index_id: int,
                       first_nfeatures: int,
                       chunk_size:int):

    path_to_imgs_dir = path_to_dir + 'photos/'

    # debug
    # write_images_list_to_file(path_to_imgs_data, list_imgs)

    # if chunk_size too big may be need divide external loop !!! because take a lot of RAM
    indexed_images = 0
    bad_img_counter = 0
    curr_idx = 0
    last_idx = 0
    logger.info("Proccessing %d images", len(list_imgs))

    txt_bad_data = open(path_to_dir+'/bad_images_data.txt', 'a')

    with open( path_to_dir+'parsed_data.json', 'r') as json_file:
        data = json.load(json_file)
    # Create a mapping between img_name and t_msg_id
    img_name_to_id_map = {entry['img_name']: entry['t_msg_id'] for entry in data}

    while curr_idx < len(list_imgs):
        if curr_idx + chunk_size < len(list_imgs):
           last_idx = curr_idx + chunk_size 
        else:
            last_idx = len(list_imgs)

        imgs_kv_data = []
        imgs_data = np.array([], dtype=np.float32).reshape(0, 128*first_nfeatures)
        for img_name in list_imgs[curr_idx:last_idx]:
            desc, _ = img_proccessing.get_image_data(path_to_imgs_dir + img_name, first_nfeatures)

            if desc.shape[0] < first_nfeatures:
                #debug
                txt_bad_data.write(f'Bad image feature detection: {img_name}' + '\n')

                logger.info("Bad image feature detection: %s", img_name)
                logger.info("descriptor features len: %s", desc.shape[0])
                bad_img_counter = bad_img_counter + 1
                continue

            img_id =  indexed_images # extract_image_id(img_path)
            t_msg_id = img_name_to_id_map.get(img_name, None)

            #debug
            if t_msg_id == None:
                txt_bad_data.write(f'Bad telegram message id: {img_name}' + '\n')
                logger.info("Can't find telegram message id: %s", img_name)

            imgs_kv_data.append( {
                "index_id": index_id,
                "img_id": img_id, #in index and pos in desc file
                "t_msg_id": t_msg_id,
                "img_name": img_name,
            } )

            desc = desc[:first_nfeatures].reshape(1,-1)
            imgs_data = np.vstack((imgs_data, desc), dtype=np.float32)
            indexed_images = indexed_images + 1

        append_array_with_same_width(desc_file, imgs_data)
        store_img_data(imgs_kv_data, path_to_dir)

        # debug
        # np.savetxt(desc_file+"_.txt", imgs_data, fmt="%.2f")
        with open(desc_file+"_.txt", 'a') as file:
            # Iterate over the rows of the NumPy array and write each row as a new line
            for row in imgs_data:
                # Convert the row to a string with tabs as delimiters and write it to the file
                file.write('\t'.join(map(str, row)) + '\n')

        curr_idx += chunk_size

    logger.info("Batch fullfilled. Processed %d images", curr_idx)
    logger.info("Bad images %d", bad_img_counter)

    if bad_img_counter != 0:
        logger.info("Index has %.2f%% of banch images", float(100.0 - (bad_img_counter*100)/len(list_imgs)) )

    return indexed_images


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
    mapped_indices = []

    # with open(file_path, 'rb') as file:
    #     for start_idx in range(0, len(row_indices), chunk_size):
    #         end_idx = min(start_idx + chunk_size, len(row_indices))
    #         chunk_indices = row_indices[start_idx:end_idx]
            
    #         # Create a memory-mapped array for the chunk
    #         mmap_array = np.memmap(file, dtype=dtype, mode='r', shape=(len(chunk_indices), num_columns))
            
    #         # Append the chunk indices to the list of mapped indices
    #         mapped_indices.extend(chunk_indices)

    #         # Retrieve the rows using the indices
    #         chunk = mmap_array
    #         result.extend(chunk)

    # Calculate the number of elements to read
    num_elements = len(row_indices) * num_columns
    # Initialize an empty array to store the selected rows
    selected_rows = np.empty((len(row_indices), num_columns), dtype=dtype)

    # Read the specified rows from the binary file
    with open(file_path, 'rb') as file:
        for i, row_idx in enumerate(row_indices):
            file.seek(row_idx * num_columns * dtype.size)  # Assuming 4 bytes per element (float32)
            print("huy: ", i)
            selected_rows[i] = np.fromfile(file, dtype=dtype, count=num_columns)

    return selected_rows

    return np.array(mapped_indices), np.array(result, dtype=dtype)

def read_specific_rows_from_binfile(file_path, rows_to_read, num_columns:int, dtype=np.float32):
    # Calculate the number of elements to read
    num_elements = len(rows_to_read) * num_columns

    # Initialize an empty array to store the selected rows
    selected_rows = np.empty((len(rows_to_read), num_columns), dtype=dtype)

    # Read the specified rows from the binary file
    with open(file_path, 'rb') as file:
        for i, row_idx in enumerate(rows_to_read):
            bytes_to_skip = row_idx * num_columns * (np.dtype(dtype).itemsize)
            file.seek(bytes_to_skip)  # Assuming 4 bytes per element (float32)
            selected_rows[i] = np.fromfile(file, dtype=dtype, count=num_columns)

    return selected_rows


def get_array_rows_count(filename, row_size: int, dtype=np.float32):
    file_size = os.path.getsize(filename)
    element_size = np.dtype(dtype).itemsize
    return file_size // (row_size * element_size)