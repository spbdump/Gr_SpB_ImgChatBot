
import random
import os

def get_random_images(image_list, n:int) -> list[str]:
    """
    Select n random images from the image_list.

    Args:
        image_list (list): A list of image filenames or paths.
        n (int): The number of random images to select.

    Returns:
        list: A list containing n randomly selected images from image_list.
    """
    if n >= len(image_list):
        return image_list  # Return the whole list if n is greater or equal to the list size

    random_images = random.sample(image_list, n)
    return random_images

def get_image_files(directory: str):
    image_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_files.append(filename)
    return image_files