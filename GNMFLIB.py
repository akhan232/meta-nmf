import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF
from scipy.spatial.distance import pdist, squareform
import warnings


random_state = np.random.RandomState(0) # Set Random State

def check_non_negative(X, whom):
    """
    Checks and raises Error if there are non negative values in data X
    
    Parameters
    -------------------------------------------------------
    X : ndarray - Matrix to check non negativity
    whom : String - Name of Function calling this.
    ---------------------------------------------------------------------------
    ValueError
        Negative values in data passed to 'whom'.

    Returns
    ---------------------------------------------------------------------------
    None.

    """
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)


def nndsvd_initialization(X, n_components, init=None):
    """
    Parameters
    ---------------------------------------------------------------------------
    X : ndarray - dataset
    n_components : int - rank (r)
    init : string, optional - Initialization to be used ['nndsvd', 'random']. The default is None.

    Raises
    ---------------------------------------------------------------------------
    ValueError
        Raises Error if init is anything other than [None, 'nndsvd', 'random']

    Returns
    ---------------------------------------------------------------------------
    W : ndarray - Basis Matrix - X.shape[1] x n_components
    H : ndarray - Coefficients Matrix - X.shape[0] x n_components
    
    """
    n_samples, n_features = X.shape
    if init is None:
        if n_components < n_features:
            init = 'nndsvd'
        else:
            init = 'random'

    if init == 'nndsvd':
        W, H = _initialize_nmf(X, n_components)
    elif init == "random":
        rng = check_random_state(random_state)
        W = rng.randn(n_samples, n_components)
        np.abs(W, W)
        H = rng.randn(n_components, n_features)
        np.abs(H, H)
    else:
        raise ValueError(
                'Invalid init parameter: got %r instead of one of %r' %
                (init, (None, 'nndsvd', 'random')))
    return W, H


def _initialize_nmf(X, n_components, variant=None, eps=1e-6,
                    random_state=None):
    """NNDSVD algorithm for NMF initialization.

    Computes a good initial guess for the non-negative
    rank k matrix approximation for X: X = WH

    Parameters
    ---------------------------------------------------------------------------

    X : array, [n_samples, n_features]
        The data matrix to be decomposed.

    n_components : array, [n_components, n_features]
        The number of components desired in the approximation.

    variant : None | 'a' | 'ar'
        The variant of the NNDSVD algorithm.
        Accepts None, 'a', 'ar'
        None: leaves the zero entries as zero
        'a': Fills the zero entries with the average of X
        'ar': Fills the zero entries with standard normal random variates.
        Default: None

    eps: float
        Truncate all values less then this in output to zero.

    random_state : numpy.RandomState | int, optional
        The generator used to fill in the zeros, when using variant='ar'
        Default: numpy.random

    Returns
    ---------------------------------------------------------------------------

    (W, H) :
        Initial guesses for solving X ~= WH such that
        the number of columns in W is n_components.

    Remarks
    ---------------------------------------------------------------------------

    This implements the algorithm described in
    C. Boutsidis, E. Gallopoulos: SVD based
    initialization: A head start for nonnegative
    matrix factorization - Pattern Recognition, 2008

    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "_initialize_nmf in GNMFLIB.py")
    if variant not in (None, 'a', 'ar'):
        raise ValueError("Invalid variant name")

    U, S, V = randomized_svd(X, n_components)
    # dtype modification
    W, H = np.zeros(U.shape, dtype=np.float32), np.zeros(V.shape,
                                                         dtype=np.float32)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = LA.norm(x_p), LA.norm(y_p)
        x_n_nrm, y_n_nrm = LA.norm(x_n), LA.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if variant == "a":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif variant == "ar":
        random_state = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * random_state.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * random_state.randn(len(H[H == 0])) / 100)

    return W, H

def rbf(X, sigma=0.5):
    """
    Calculates the Radial Basis Function (RBF) kernel.

    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix of shape (n_samples, n_features).
    sigma : float, optional
        Width parameter of the RBF kernel. The default is 0.5.

    Returns
    -------
    A : numpy.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    """

    pairwise_dists = squareform(pdist(X, 'euclidean'))
    A = np.exp(-pairwise_dists ** 2 / sigma ** 2)
    return A


def gnmf(X, W, H, nndsvd_init=False, lambd=0.3, n_components=None, tol_nmf=1e-3, 
         max_iter=100, verbose=False):
    """Performs Graph Non-negative Matrix Factorization (GNMF).

    Args:
        X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
        W (numpy.ndarray): Initial value for the matrix W.
        H (numpy.ndarray): Initial value for the matrix H.
        nndsvd_init (bool, optional): Whether to use nndsvd initialization. Defaults to False.
        lambd (float, optional): Regularization parameter. Defaults to 0.3.
        n_components (int, optional): Number of components (rank of factorization). 
                                     Defaults to min(n_samples, n_features).
        tol_nmf (float, optional): Tolerance for convergence. Defaults to 1e-3.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        tuple: A tuple containing:
            W (numpy.ndarray): The learned matrix W.
            H (numpy.ndarray): The learned matrix H.
            list_reconstruction_err_ (list): List of reconstruction errors at each iteration.
    """

    A = rbf(X.T)  # RBF kernel

    X = check_array(X)
    check_non_negative(X, "gnmf in GNMFLIB.py")
    n_samples, n_features = X.shape

    n_components = n_components or min(n_samples, n_features) #Simplified n_components setting

    if nndsvd_init:
        W, H = nndsvd_initialization(X, n_components, init='nndsvd')  # Assuming NBS_init is defined
    
    list_reconstruction_err_ = []
    reconstruction_err_ = LA.norm(X - np.dot(W, H))
    list_reconstruction_err_.append(reconstruction_err_)

    eps = np.spacing(1)
    Lp = np.diag(np.asarray(A).sum(axis=0))  # Degree matrix
    Lm = A  # Laplacian matrix

    for n_iter in range(1, max_iter + 1):

        if verbose:
            print(f"Iteration ={n_iter:4d} / {max_iter:d} - - - - — Error = {reconstruction_err_:.2f} - - - - — Tolerance = {tol_nmf:f}")

        H = H * (lambd * np.dot(H, Lm) + np.dot(W.T, (X + eps) / (np.dot(W, H) + eps))) / (lambd * np.dot(H, Lp) + np.dot(W.T, np.ones(X.shape)) + eps) #Combined update rule and added eps
        H[H <= 0] = eps
        H[np.isnan(H)] = eps

        W = W * (np.dot((X + eps) / (np.dot(W, H) + eps), H.T)) / (np.dot(np.ones(X.shape), H.T) + eps) #Combined update rule and added eps
        W[W <= 0] = eps
        W[np.isnan(W)] = eps

        current_reconstruction_err = LA.norm(X - np.dot(W, H))

        if reconstruction_err_ > current_reconstruction_err:
            H = (1 - eps) * H + eps * np.random.normal(0, 1, (n_components, n_features))**2
            W = (1 - eps) * W + eps * np.random.normal(0, 1, (n_samples, n_components))**2

        reconstruction_err_ = current_reconstruction_err
        list_reconstruction_err_.append(reconstruction_err_)

        if reconstruction_err_ < tol_nmf:
            warnings.warn("Tolerance error reached during fit")
            break

        if np.isnan(W).any() or np.isnan(H).any():
            warnings.warn(f"NaN values at {n_iter} Error={reconstruction_err_}")
            break

    return np.squeeze(np.asarray(W)), np.squeeze(np.asarray(H)), list_reconstruction_err_


def nmf(X, n_components, tol_nmf, max_iter=100):
    """Performs Non-negative Matrix Factorization (NMF) using scikit-learn.

    Args:
        X (numpy.ndarray or sparse matrix): Input data matrix.
        n_components (int): Number of components.
        tol_nmf (float): Tolerance for convergence.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple: A tuple containing:
            W (numpy.ndarray): The learned matrix W.
            H (numpy.ndarray): The learned matrix H.
    """
    check_non_negative(X, "nmf in GNMFLIB.py")
    n_samples, n_features = X.shape

    n_components = n_components or min(n_samples, n_features) #Simplified n_components setting

    model = NMF(n_components, tol=tol_nmf, max_iter=max_iter, init='nndsvd')
    W = model.fit_transform(X)
    H = model.components_

    return W, H