import numpy as np

def marshal_arrays(arrays):
    """
    Flatten a list of numpy arrays and store their shapes.

    Parameters:
    arrays (list of np.ndarray): List of numpy arrays to be marshalled.

    Returns:
    tuple: A tuple containing:
        - flat (np.ndarray): A single concatenated numpy array of all elements.
        - shapes (list of tuple): A list of shapes of the original arrays.
    """
    flattened = list(map(lambda a : np.reshape(a,-1),arrays))
    shapes = list(map(np.shape,arrays))
    flat = np.concatenate(flattened)
    return flat, shapes

def unmarshal_arrays(flat,shapes):
    """
    Rebuild the original list of numpy arrays from the flattened array and shapes.

    Parameters:
    flat (np.ndarray): The single concatenated numpy array of all elements.
    shapes (list of tuple): A list of shapes of the original arrays.

    Returns:
    list of np.ndarray: The list of original numpy arrays.
    """
    sizes = list(map(np.prod,shapes))
    splits = np.cumsum(np.asarray(sizes,dtype=int))[:-1]
    flattened = np.split(flat,splits)
    arrays = list(map(lambda t : np.reshape(t[0],t[1]),zip(flattened,shapes)))
    return arrays
