from PIL import Image
import functools
from tqdm import tqdm
import numpy as np



def load_reduce(file_list, reducer):
    """Loads and reduces a list of image files using a reduction function.

    Args:
        file_list (list of str): List of paths to image files.
        reducer (function): Function to reduce the images.

    Returns:
        array: Reduced image.
    """
    reduced_image = functools.reduce(reducer, map(np.asarray, map(Image.open, tqdm(file_list))))
    return reduced_image

def load_map(file_list, mapper):
    """Loads and maps a list of image files using a mapping function.

    Args:
        file_list (list of str): List of paths to image files.
        mapper (function): Function to map the images.

    Returns:
       List of array: Mapped images.
    """
    mapped_images = map(mapper, map(np.asarray, map(Image.open, tqdm(file_list))))
    return mapped_images

def load_max_image(file_list):
    """Loads a list of image files and computes the element-wise maximum.

    Args:
        file_list (list of str): List of paths to image files.

    Returns:
        array: Image with the element-wise maximum values.
    """
    max_image = load_reduce(file_list, np.maximum)
    return max_image