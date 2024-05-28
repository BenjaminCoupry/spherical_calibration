import numpy as np
import utils.vector_utils as vector_utils

def deproject_ellipse_to_sphere(M, radius):
    """finds the deprojection of an ellipse to a sphere

    Args:
        M (Array 3,3): Ellipse quadratic form
        radius (float): radius of the researched sphere

    Returns:
        Array 3: solution of sphere centre location
    """
    H = 0.5*(np.swapaxes(M,-1,-2)+M)
    eigval, eigvec = np.linalg.eigh(H)
    i_unique = np.argmax(np.abs(np.median(eigval,axis=-1,keepdims=True)-eigval),axis=-1)
    unique_eigval = np.take_along_axis(eigval,i_unique[...,None],-1)[...,0]
    unique_eigvec = np.take_along_axis(eigvec,i_unique[...,None,None],-1)[...,0]
    double_eigval = 0.5*(np.sum(eigval,axis=-1)-unique_eigval)
    z_sign = np.sign(unique_eigvec[...,-1])
    dist = np.sqrt(1-double_eigval/unique_eigval)
    C = np.real(radius*(dist*z_sign)[...,None]*vector_utils.norm_vector(unique_eigvec)[1])
    return C
