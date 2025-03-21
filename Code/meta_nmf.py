from utils import generate_tasks
from NMF import nmf
import numpy as np

def fit(X,y, k=2, nb_tasks=5, task_size= 100, maxiter_outer=500, maxiter_inner=100, init='random'):
    """
    Meta Non-negative Matrix Factorization (NMF) using multiplicative update rules.

    Args:
    ---------------------------------------------------------------------------
        X: numpy.ndarray, shape (n_samples, n_features). The main (target) dataset.
        y: numpy.ndarray, shape (n_samples,). Labels associated with the main dataset (not used in the NMF process, but passed to generate tasks).
        k: int. The desired rank of the factorized matrices (number of components).
        nb_tasks: int. The number of small sub-tasks to generate.
        task_size: int. The number of samples in each small sub-task.
        maxiter_outer: int. The maximum number of outer loop iterations. (for Meta)
        maxiter_inner: int. The maximum number of inner loop iterations (for NMF).
        init: str. The initialization method for NMF ('random' or 'nndsvd').

    Returns:
    ---------------------------------------------------------------------------
        Ht: numpy.ndarray, shape (k, n_samples). The factorized matrix H for the target task.
        Wt: numpy.ndarray, shape (n_features, k). The factorized matrix W for the target task.
    """
    
    # Initialization
    m = np.shape(X)[1]
    n = np.shape(X)[0]
    
    nb_tasks = 5
    task_s_size = 100   # _s_ means small tasks

    # Ws and Hs are factorized mat of small tasks
    Ws = np.random.rand(m, k, nb_tasks)
    Hs = np.random.rand(k, task_s_size, nb_tasks)
    
    # Wt and Ht are facorized mat of target tasks
    Wt = np.zeros((m,k))
    Ht = np.zeros((k,n))
    
    sum_W = np.zeros((m,k))
    
    # generate the related small tasks
    Tasks = generate_tasks(X,y, nb_tasks=nb_tasks, task_size = task_s_size)
       
    # Extract the target task
    target_task = Tasks['task_t']
    Xt = target_task[0] # features
    
    iter = 0
    while iter < maxiter_outer: #Wt_diff >= 1.5e-3 or Ht_diff >= 1.5e-3:
       
        # for each task
        C = np.zeros(nb_tasks) # List of Costs
        for i in range(nb_tasks):
            # get a task (features only)
            task = Tasks['task'+str(i)]
            Xs = task[0]
            # update W and H and compute the cost of each
            W = Wt+Ws[:,:,i]
            H = W.T @ Xs.T
            Hs[:,:,i], Ws[:,:,i], C[i] = nmf(Xs.T, k, thresh=0.001, L= maxiter_inner, W=W, H=H, norm=True, report=False, init=init)

           
            # accumulate the basis factorize matrices Ws
            sum_W = sum_W +  1 *  Ws[:,:,i]
          
        # Initialize Wt and Ht (target task)
        Wt = Wt + sum_W/nb_tasks
        Ht = Wt.T @ Xt.T
              
        # update Wt and Ht
        Ht, Wt, Ct = nmf(Xt.T, k, thresh=0.0, L=maxiter_inner, W=Wt, H=Ht, norm=True, report=False)
    
        iter = iter + 1
        
    return Ht, Wt

def predict(Xts, Wt, k=2, maxiter = 100, init='random'):
    """
    Predicts the coefficient matrix (Hts) for new test data (Xts) using a pre-trained basis matrix (Wt).
    
    Args:
    ---------------------------------------------------------------------------
    Xts : numpy.ndarray, shape (n_test_samples, n_features)
        Matrix of test features. Each row represents a sample, and each column represents a feature.
    Wt : numpy.ndarray, shape (n_features, k)
        Pre-trained basis matrix obtained from a training phase. It represents the learned basis vectors.
    k : int, optional
        The rank of the factorization. Defaults to 2.
    maxiter : int, optional, Maximum number of iterations for the NMF algorithm. Defaults to 100.
    init : str, optional, Initialization method for NMF. Defaults to 'random'.

    Returns
    ---------------------------------------------------------------------------
    Hts : numpy.ndarray, shape (k, n_test_samples)
        Coefficient matrix representing the test data in the lower-dimensional space defined by Wt.
    Wts : numpy.ndarray, shape (n_features, k), The basis matrix returned from NMF.
    """
    
    # perform NMF on the test data initialized by Meta NMF factorized matrix Wt
    Xts = Xts.T
    Wts = Wt
    Hts = Wts.T @ Xts
  
    Hts, Wts, _ = nmf(Xts, k, thresh=0.0, L=maxiter, W=Wts, H=Hts, norm=True, report=False, init=init)
    return Hts, Wts