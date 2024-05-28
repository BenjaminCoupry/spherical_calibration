import numpy as np
import utils.kernels as kernels
import utils.vector_utils as vector_utils
import utils.quadratic_forms as quadratic_forms
import utils.kernels as kernels

def sphere_parameters_from_points(points):
    """evaluates sphere parameters from a set of points

    Args:
        points (Array ... npoints ndim): points used to fit the sphere, homogeneous coordinates

    Returns:
        Array ... ndim: coordinates of the center of the sphere
        Array ...: values of radius of the sphere
    """
    homogeneous = vector_utils.to_homogeneous(points)
    Q = quadratic_forms.fit_quadratic_form(homogeneous)
    scale = np.mean(np.diagonal(Q[...,:-1,:-1],axis1=-2,axis2=-1))
    scaled_Q = Q*np.expand_dims(np.reciprocal(scale),axis=(-1,-2))
    center = -(scaled_Q[...,-1,:-1]+scaled_Q[...,:-1,-1])/2
    centered_norm = vector_utils.norm_vector(center)[0]
    radius = np.sqrt(np.square(centered_norm)-scaled_Q[...,-1,-1])
    return center,radius

def plane_parameters_from_points(points):
    """Computes the parameters of a plane from a set of points.

    Args:
        points (Array ..., dim): Coordinates of the points used to define the plane.

    Returns:
        Array ..., dim: Normal vector to the plane.
        Array ...: Plane constant alpha.
    """
    homogeneous = vector_utils.to_homogeneous(points)
    E = np.einsum('...ki,...kj->...ij',homogeneous,homogeneous)
    L = kernels.matrix_kernel(E)
    n,alpha = L[...,:-1],L[...,-1]
    return n, alpha