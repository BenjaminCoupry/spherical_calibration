import numpy as np
import utils.vector_utils as vector_utils

def build_K_matrix(focal_length, u0, v0):
    """
    Build the camera intrinsic matrix.

    Parameters:
    focal_length (float): Focal length of the camera.
    u0 (float): First coordinate of the principal point.
    v0 (float): Seccond coordinate of the principal point.

    Returns:
    numpy.ndarray: Camera intrinsic matrix (3x3).
    """
    K = np.asarray([[focal_length, 0, u0],
                    [0, focal_length, v0],
                    [0, 0, 1]])
    return K

def get_camera_rays(points,K):
    """Computes the camera rays for a set of points given the camera matrix K.

    Args:
        points (Array ..., 2): Points in the image plane.
        K (Array 3, 3): Camera intrinsic matrix.

    Returns:
        Array ..., 3: Camera rays corresponding to the input points.
    """
    homogeneous = vector_utils.to_homogeneous(points)
    inv_K = np.linalg.inv(K)
    rays = np.einsum('ij,...j->...i',inv_K,homogeneous)
    return rays