import numpy as np


def norm_vector(v):
    """computes the norm and direction of vectors

    Args:
        v (Array ..., dim): vectors to compute the norm and direction for

    Returns:
        Array ...: norms of the vectors
        Array ..., dim: unit direction vectors
    """
    norm = np.linalg.norm(v,axis=-1)
    direction = v/norm[...,np.newaxis]
    return norm,direction

def dot_product(v1,v2):
    """Computes the dot product between two arrays of vectors.

    Args:
        v1 (Array ..., ndim): First array of vectors.
        v2 (Array ..., ndim): Second array of vectors.

    Returns:
        Array ...: Dot product between v1 and v2.
    """
    result = np.einsum('...i,...i->...',v1,v2)
    return result

def cross_to_skew_matrix(v):
    """converts a vector cross product to a skew-symmetric matrix multiplication

    Args:
        v (Array ..., 3): vectors to convert

    Returns:
        Array ..., 3, 3: matrices corresponding to the input vectors
    """
    indices = np.asarray([[-1,2,1],[2,-1,0],[1,0,-1]])
    signs = np.asarray([[0,-1,1],[1,0,-1],[-1,1,0]])
    skew_matrix = v[...,indices]*signs
    return skew_matrix

def to_homogeneous(v):
    """converts vectors to homogeneous coordinates

    Args:
        v (Array ..., dim): input vectors

    Returns:
        Array ..., dim+1: homogeneous coordinates of the input vectors
    """
    append_term = np.ones(np.shape(v)[:-1]+(1,))
    homogeneous = np.append(v,append_term,axis=-1)
    return homogeneous

def one_hot(i,imax):
    """Converts indices to one-hot encoded vectors.

    Args:
        i (Array ...): Array of indices.
        imax (int): Number of classes.

    Returns:
        Array ..., imax: One-hot encoded vectors.
    """
    result = np.arange(imax)==np.expand_dims(i,axis=-1)
    return result
