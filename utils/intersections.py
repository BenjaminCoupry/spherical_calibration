import numpy as np
import utils.vector_utils as vector_utils
import utils.kernels as kernels


def lines_intersections_system(points,directions):
    """computes the system of equations for intersections of lines, Ax=b
    where x is the instersection

    Args:
        points (Array ..., npoints, ndim): points through which the lines pass
        directions (Array ..., npoints, ndim): direction vectors of the lines

    Returns:
        Array ..., 3*npoints, ndim: coefficient matrix A for the system of equations
        Array ..., 3*npoints: right-hand side vector b for the system of equations
    """
    n = vector_utils.norm_vector(directions)[1]
    skew = np.swapaxes(vector_utils.cross_to_skew_matrix(n),-1,-2)
    root = np.einsum('...uij,...uj->...ui',skew,points)
    A = np.concatenate(np.moveaxis(skew,-3,0),axis=-2)
    b = np.concatenate(np.moveaxis(root,-2,0),axis=-1)
    return A,b

def lines_intersections(points,directions):
    """computes the intersections of lines

    Args:
        points (Array ..., npoints, ndim): points through which the lines pass
        directions (Array ..., npoints, ndim): direction vectors of the lines

    Returns:
        Array ..., ndim: intersection
    """
    A,b = lines_intersections_system(points,directions)
    x = kernels.iteratively_reweighted_least_squares(A,b)
    return x

def line_sphere_intersection_determinant(center,radius,directions):
    """computes the determinant for the intersection of a line and a sphere,

    Args:
        center (Array ..., dim): center of the sphere
        radius (Array ...): radius of the sphere
        directions (Array ..., dim): direction of the line

    Returns:
        Array ...:intersection determinant
    """
    directions_norm_2 = np.square(vector_utils.norm_vector(directions)[0])
    center_norm_2 = np.square(vector_utils.norm_vector(center)[0])
    dot_product_2 = np.square(vector_utils.dot_product(center,directions))
    delta = dot_product_2-directions_norm_2*(center_norm_2-np.square(radius))
    return delta

def line_plane_intersection(normal,alpha,directions):
    """Computes the intersection points between a line and a plane.

    Args:
        normal (Array ..., ndim): Normal vector to the plane.
        alpha (Array ...): Plane constant alpha.
        directions (Array ..., dim): direction of the line

    Returns:
        Array ..., ndim: Intersection points between the line and the sphere.
    """
    t = -alpha*np.reciprocal(vector_utils.dot_product(directions,normal))
    intersection = directions*t[...,np.newaxis]
    return intersection

def line_sphere_intersection(center,radius,directions):
    """Computes the intersection points between a line and a sphere.

    Args:
        center (Array ..., ndim): Center of the sphere.
        radius (Array ...): Radius of the sphere.
        directions (Array ..., ndim): Direction vectors of the line.

    Returns:
        Array ..., ndim: Intersection points between the line and the sphere.
        Array bool ...: Mask of intersection points
    """
    delta = line_sphere_intersection_determinant(center,radius,directions)
    mask = delta>0
    dot_product = vector_utils.dot_product(center,directions)
    directions_norm_2 = np.square(vector_utils.norm_vector(directions)[0])
    distances = (dot_product-np.sqrt(np.maximum(0,delta)))*np.reciprocal(directions_norm_2)
    intersection = np.expand_dims(distances,axis=-1)*directions
    return intersection,mask

def sphere_intersection_normal(center,point):
    """Computes the normal vector at the intersection point on a sphere.

    Args:
        center (Array ..., dim): Coordinates of the sphere center.
        point (Array ..., dim): Coordinates of the intersection point.

    Returns:
        Array ..., dim: Normal normal vector at the intersection point.
    """
    vector = point-center
    normal = vector_utils.norm_vector(vector)[1]
    return normal