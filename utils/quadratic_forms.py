import numpy as np
import utils.kernels as kernels
import utils.vector_utils as vector_utils


def evaluate_bilinear_form(Q,left,right):
    """evaluates bilinear forms at several points

    Args:
        Q (Array ...,ldim,rdim): bilinear form to evaluate
        left (Array ...,ldim): points where the bilinear form is evaluated to the left
        right (Array ...,rdim): points where the bilinear form is evaluated to the right
    Returns:
        Array ... bilinear forms evaluated
    """
    result = np.einsum('...ij,...i,...j->...',Q,left,right)
    return result

def evaluate_quadratic_form(Q,points):
    """evaluates quadratic forms at several points

    Args:
        Q (Array ...,dim,dim): quadratic form to evaluate
        points (Array ...,dim): points where the quadratic form is evaluated
    Returns:
        Array ... quadratic forms evaluated
    """
    result = evaluate_bilinear_form(Q,points,points)
    return result

def merge_quadratic_to_homogeneous(Q,b,c):
    """merges quadratic form, linear term, and constant term into a homogeneous matrix

    Args:
        Q (Array ..., dim, dim): quadratic form matrix
        b (Array ..., dim): linear term vector
        c (Array ...): constant term

    Returns:
        Array ..., dim+1, dim+1: homogeneous matrix representing the quadratic form
    """
    dim_points = Q.shape[-1]
    stack_shape = np.broadcast_shapes(np.shape(Q)[:-2],np.shape(b)[:-1],np.shape(c))
    Q_b = np.broadcast_to(Q,stack_shape+(dim_points,dim_points))
    b_b = np.broadcast_to(np.expand_dims(b,-1),stack_shape+(dim_points,1))
    c_b = np.broadcast_to(np.expand_dims(c,(-1,-2)),stack_shape+(1,1))
    H = np.block([[Q_b,0.5*b_b],[0.5*np.swapaxes(b_b,-1,-2),c_b]])
    return H

def quadratic_to_dot_product(points):
    """computes the matrix W such that
    x.T@Ax = W(x).T*A[ui,uj]

    Args:
        points ( Array ...,ndim): points of dimension ndim

    Returns:
        Array ...,ni: dot product matrix (W)
        Array ni: i indices of central matrix
        Array ni: j indices of central matrix
    """
    dim_points = points.shape[-1]
    ui,uj = np.triu_indices(dim_points)
    W = points[...,ui]*points[...,uj]
    return W,ui,uj

def fit_quadratic_form(points):
    """Fits a quadratic form to the given zeroes.

    Args:
        points (Array ..., n, dim): Input points.

    Returns:
        Array ..., dim, dim: Fitted quadratic form matrix.
    """
    dim_points = points.shape[-1]
    normed_points = vector_utils.norm_vector(points)[1]
    W,ui,uj = quadratic_to_dot_product(normed_points)
    H = np.einsum('...ki,...kj->...ij',W,W)
    V0 = kernels.matrix_kernel(H)
    Q = np.zeros(V0.shape[:-1]+(dim_points,dim_points))
    Q[...,ui,uj]=V0
    return Q
    
# import matplotlib.pyplot as plt

# Q = np.random.randn(3,3)

# x0, y0 = np.linspace(-1,1,300),np.linspace(-1,1,300)
# x,y = np.meshgrid(x0,y0,indexing='ij')
# points = vector_utils.to_homogeneous(np.stack([x,y],axis=-1))
# f = evaluate_quadratic_form(Q,points)
# mask = np.abs(f)<0.01
# u,v = np.where(mask)
# zeros = vector_utils.to_homogeneous(np.stack([x0[u],y0[v]],axis=-1))+np.random.randn(5,u.shape[0],3)*0.1
# Qc = fit_quadratic_form(zeros)
# fchap = evaluate_quadratic_form(Qc,points[...,None,:])
# print()
