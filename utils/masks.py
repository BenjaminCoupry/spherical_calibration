import numpy as np
import scipy.ndimage as ndimage

def get_greatest_components(mask, n):
    """
    Extract the n largest connected components from a binary mask.

    Parameters:
        mask (Array ...): The binary mask.
        n (int): The number of largest connected components to extract.

    Returns:
        Array n,...: A boolean array of the n largest connected components
    """
    labeled, _ = ndimage.label(mask)
    unique, counts = np.unique(labeled, return_counts=True)
    greatest_labels = unique[unique != 0][np.argsort(counts[unique != 0])[-n:]]
    greatest_components = labeled[np.newaxis,...] == np.expand_dims(greatest_labels,axis=tuple(range(1,1+mask.ndim)))
    return greatest_components

def get_mask_border(mask):
    """
    Extract the border from a binary mask.

    Parameters:
    mask (Array ...): The binary mask.

    Returns:
    Array ...: A boolean array mask of the border
    """
    inverted_mask = np.logical_not(mask)
    dilated = ndimage.binary_dilation(inverted_mask)
    border = np.logical_and(mask,dilated)
    return border

def select_binary_mask(mask,metric):
    """Selects the side of a binary mask that optimizes the given metric.

    Args:
        mask (Array bool ...): Initial binary mask.
        metric (function): Function to evaluate the quality of the mask.

    Returns:
        Array bool ...: Selected binary mask that maximizes the metric.
    """
    inverted = np.logical_not(mask)
    result = mask if metric(mask)>metric(inverted) else inverted
    return result
