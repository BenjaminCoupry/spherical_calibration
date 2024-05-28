import numpy as np
import utils.quadratic_forms as quadratic_forms


def gaussian_pdf(mu,sigma,x):
    """Computes the PDF of a multivariate Gaussian distribution.

    Args:
        mu (Array ...,k): Mean vector.
        sigma (Array ...,k,k): Covariance matrix.
        x (Array ...,k): Input vector.

    Returns:
        Array ...: Value of the PDF.
    """
    k = np.shape(x)[-1]
    Q = np.linalg.inv(sigma)
    normalization = np.reciprocal(np.sqrt(np.linalg.det(sigma)*np.power(2.0*np.pi,k)))
    quadratic = quadratic_forms.evaluate_quadratic_form(Q,x-mu)
    result = np.exp(-0.5*quadratic)*normalization
    return result

def gaussian_estimation(x,weights):
    """Estimates the mean and covariance matrix of a Gaussian distribution.

    Args:
        x (Array ...,n,dim): Data points.
        weights (Array ...,n): Weights for each data point.

    Returns:
        Array ...,dim: Estimated mean vector.
        Array ...,dim,dim: Estimated covariance matrix.
    """
    weights_sum = np.sum(weights,axis=-1)
    mu = np.sum(x*np.expand_dims(weights,axis=-1),axis=-2)/np.expand_dims(weights_sum,axis=-1)
    centered_x = x-np.expand_dims(mu,axis=-2)
    sigma = np.einsum('...s,...si,...sj->...ij',weights,centered_x,centered_x)/np.expand_dims(weights_sum,axis=(-1,-2))
    return mu,sigma

def gaussian_mixture_estimation(x,init_params,it=100):
    """Estimates the parameters of a k Gaussian mixture model using the EM algorithm.

    Args:
        x (Array ..., n, dim): Data points.
        init_params (tuple): Initial parameters (pi, sigma, mu).
            pi (Array ..., k): Initial mixture weights.
            sigma (Array ..., k, dim, dim): Initial covariance matrices.
            mu (Array ..., k, dim): Initial means.
        it (int, optional): Number of iterations. Defaults to 100.

    Returns:
        Tuple[(Array ..., k), (Array ..., k, dim, dim), (Array ..., k, dim)]: 
            Estimated mixture weights,covariance matrices, means.
    """
    pi,sigma,mu = init_params
    for _ in range(it):
        pdf = gaussian_pdf(np.expand_dims(mu,axis=-2),
                           np.expand_dims(sigma,axis=-3),
                           np.expand_dims(x,axis=-3))*np.expand_dims(pi,axis=-1)
        weights = pdf/np.sum(pdf,axis=-2,keepdims=True)
        pi=np.mean(weights,axis=-1)
        mu,sigma = gaussian_estimation(x,weights)
    return pi,sigma,mu

def maximum_likelihood(x,params):
    """Selects the best gaussian model for a point

    Args:
        x (Array ..., dim): Data points.
        params (tuple): Gaussians parameters (pi, sigma, mu).
            pi (Array ..., k): Mixture weights.
            sigma (Array ..., k, dim, dim): Covariance matrices.
            mu (Array ..., k, dim): Means.

    Returns:
        Array ...: integer in [0,k-1] giving the maximum likelihood model
    """
    pi,sigma,mu = params
    pdf = gaussian_pdf(mu,sigma,np.expand_dims(x,axis=-2))*pi
    result = np.argmax(pdf,axis=-1)
    return result

