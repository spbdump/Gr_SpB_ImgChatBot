import numpy as np

def append_row_to_txt(row_data, file_path='./debug_desc.txt'):
    """
    Appends a single row from a NumPy array to a text file.

    Args:
    file_path (str): The path to the text file.
    row_data (numpy.ndarray): The NumPy array containing the row to be appended.

    Returns:
    None
    """
    try:
        with open(file_path, 'a') as file:
            # Convert the row to a string and write it to the file
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")