from utils import generate_tasks
import GNMFLIB
import numpy as np

def fit(X,y, nndsvd_init=False, k=2, nb_tasks=5, task_size= 100, maxiter_outer=500, maxiter_inner=100):
    """
    Meta GNMF using multiplicative update rules.

    Args:
    ---------------------------------------------------------------------------
        X: numpy.ndarray, shape (n_samples, n_features). The main (target) dataset.
        y: numpy.ndarray, shape (n_samples,). Labels associated with the main dataset (not used in the NMF process, but passed to generate tasks).
        nndsvd_init: bool. Whether to use nndsvd init or not. Defaults to False.
        k: int. The desired rank of the factorized matrices (number of components).
        nb_tasks: int. The number of small sub-tasks to generate.
        task_size: int. The number of samples in each small sub-task.
        maxiter_outer: int. The maximum number of outer loop iterations. (for Meta)
        maxiter_inner: int. The maximum number of inner loop iterations (for NMF).

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
    while iter < maxiter_outer:
       
        # for each task
        for i in range(nb_tasks):
            # get a task (features only)
            task = Tasks['task'+str(i)]
            Xs = task[0]
            # update W and H and compute the cost of each
            W = Wt+Ws[:,:,i]
            H = W.T @ Xs.T
            if(nndsvd_init == True & iter == 0):
                Ws[:,:,i], Hs[:,:,i], _ = GNMFLIB.gnmf(Xs.T, W, H, nndsvd_init=True, n_components=k, max_iter=maxiter_inner)
            else:
                Ws[:,:,i], Hs[:,:,i], _ = GNMFLIB.gnmf(Xs.T, W, H, n_components=k, max_iter=maxiter_inner)
           
            # accumulate the basis factorize matrices Ws
            sum_W = sum_W +  1 *  Ws[:,:,i]
          
            
        # Initialize Wt and Ht (target task)
        Wt = Wt + sum_W/nb_tasks
        Ht = Wt.T @ Xt.T
              
        # update Wt and Ht
        Wt, Ht, Ct = GNMFLIB.gnmf(Xt.T, Wt, Ht, n_components=k, max_iter=maxiter_inner)
   
        iter = iter + 1
        
    return Ht, Wt

def predict(Xts, Wt, k=2, maxiter = 100):
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
    
    Wts, Hts, _ = GNMFLIB.gnmf(Xts, Wts, Hts, n_components=k, max_iter=maxiter)
    
    return Hts, Wts