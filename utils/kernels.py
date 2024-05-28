import numpy as np
import utils.vector_utils as vector_utils


def weighted_least_squares(A,y,weights):
    """Computes the weighted least squares solution of Ax=y.

    Args:
        A (Array ...,u,v): Design matrix.
        y (Array ...,u): Target values.
        weights (Array ...,u): Weights for each equation.

    Returns:
        Array ...,v : Weighted least squares solution.
    """
    pinv = np.linalg.pinv(A*weights[...,np.newaxis])
    result = np.einsum('...uv,...v->...u',pinv,y*weights)
    return result

def least_squares(A,y):
    """Computes the least squares solution of Ax=y.

    Args:
        A (Array ...,u,v): Design matrix.
        y (Array ...,u): Target values.

    Returns:
        Array ...,v : Least squares solution.
    """
    result = weighted_least_squares(A,y,np.ones(A.shape[0]))
    return result

def iteratively_reweighted_least_squares(A,y, epsilon=1e-5, it=20):
    """Computes the iteratively reweighted least squares solution. of Ax=y

    Args:
        A (Array ..., u, v): Design matrix.
        y (Array ..., u): Target values.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-5.
        it (int, optional): Number of iterations. Defaults to 20.

    Returns:
        Array ..., v: Iteratively reweighted least squares solution.
    """
    weights = np.ones(y.shape)
    for _ in range(it):
        result = weighted_least_squares(A,y,weights)
        ychap = np.einsum('...uv,...v->...u',A,result)
        delta = np.abs(ychap-y)
        weights = np.reciprocal(np.maximum(epsilon,np.sqrt(delta)))
    return result


def matrix_kernel(A):
    """Computes the eigenvector corresponding to the smallest eigenvalue of the matrix A.

    Args:
        A (Array ..., n, n): Input square matrix.

    Returns:
        Array ..., n: Eigenvector corresponding to the smallest eigenvalue.
    """
    eigval, eigvec = np.linalg.eig(A)
    min_index = np.argmin(np.abs(eigval),axis=-1)
    min_eigvec = np.take_along_axis(eigvec,min_index[...,None,None],-1)[...,0]
    normed_eigvec = vector_utils.norm_vector(min_eigvec)[1]
    return normed_eigvec

def masked_least_squares(A,y,mask):
    """Computes the least squares solution of Ax = y for masked data.

    Args:
        A (Array ..., n, p): Design matrix.
        y (Array ..., n): Target values.
        mask (Array ..., n, bool): Mask to select valid data points.

    Returns:
        Array ..., p: Least squares solution for the masked data.
    """
    masked_solver = lambda A,y,mask :  least_squares(A[mask,:],y[mask])
    vectorized = np.vectorize(masked_solver,signature='(n,p),(n),(n)->(p)')
    result = vectorized(A,y,mask)
    return result