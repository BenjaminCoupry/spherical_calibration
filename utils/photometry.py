import numpy as np
import utils.kernels as kernels
import utils.vector_utils as vector_utils

def estimate_light(normals,grey_levels, treshold = (0,1)):
    """Estimates the light directions using the given normals, grey levels, and mask.

    Args:
        normals (Array ..., n, dim): Normal vectors.
        grey_levels (Array ..., n): Grey levels corresponding to the normals.
        threshold (tuple, optional): Intensity threshold for valid grey levels. Defaults to (0, 1).

    Returns:
        Array ..., dim: Estimated light directions.
    """
    validity_mask = np.logical_and(grey_levels>treshold[0],grey_levels<treshold[1])
    lights = kernels.weighted_least_squares(normals,grey_levels,validity_mask)
    return lights

def geometric_shading_parameters(light_point, principal_directions, points):
    """Computes geometric parameters for shading based on light source and points.

    Args:
        light_point (Array ..., dim): Coordinates of the light source.
        principal_directions (Array ..., dim): Principal directions of each light.
        points (Array ..., dim): Coordinates of the points on the surface.

    Returns:
        Array ..., dim: Distances from each point to the light source.
        Array ...: Computed light direction at each point.
        Array ...: Angular factors based on the principal directions and light direction.
    """
    distance, light_direction = vector_utils.norm_vector(light_point-points)
    angular_factor = np.maximum(vector_utils.dot_product(principal_directions,-light_direction),0)
    return distance, light_direction, angular_factor

def estimate_anisotropy(light_point, principal_directions, points,normals, grey_levels, min_grey_level = 0.1, min_dot_product = 0.2):
    """Estimates anisotropy parameters based on geometric shading and grey levels.

    Args:
        light_point (Array ..., dim): Coordinates of the light source.
        principal_directions (Array ..., dim): Principal directions of each light.
        points (Array ..., dim): Coordinates of the points on the surface.
        normals (Array ..., dim): Normal vectors at each point.
        grey_levels (Array ...): Observed grey levels at each point.
        min_grey_level (float, optional): Minimum valid grey level. Defaults to 0.1.
        min_dot_product (float, optional): Minimum valid dot product for shading and angular factors. Defaults to 0.2.

    Returns:
        Array ..., 1: Estimated anisotropy parameter mu.
        Array ..., 1: Estimated flux parameter.
    """
    distance, light_direction, angular_factor = geometric_shading_parameters(light_point, principal_directions, points)
    computed_shading = np.maximum(vector_utils.dot_product(normals,light_direction),0)
    validity_mask = np.logical_and(grey_levels>min_grey_level,np.logical_and(computed_shading>min_dot_product,angular_factor>min_dot_product))
    log_flux = np.log(np.maximum(grey_levels,min_grey_level)*np.square(distance)*np.reciprocal(np.maximum(computed_shading,min_dot_product)))
    log_factor = vector_utils.to_homogeneous(np.expand_dims(np.log(np.maximum(angular_factor,min_dot_product)),axis=-1))
    eta = kernels.weighted_least_squares(log_factor,log_flux,validity_mask)
    mu,log_phi = eta[...,0], eta[...,1]
    estimated_flux = np.exp(log_phi)
    return mu,estimated_flux

def light_conditions(light_point, principal_directions, points, mu, flux):
    distance, light_direction, angular_factor = geometric_shading_parameters(light_point, principal_directions, points)
    light_conditions = light_direction*(np.reciprocal(np.square(distance))*np.power(angular_factor,mu)*flux)[...,np.newaxis]
    return light_conditions