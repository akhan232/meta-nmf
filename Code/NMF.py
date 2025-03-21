import numpy as np
from sklearn.decomposition import NMF as sklearnNMF
from sklearn.cluster import KMeans

def nmf(V, R, thresh=0.001, L=1000, W=None, H=None, init='random', norm=True, report=False):
    
    """NMF algorithm with Euclidean distance

    Args:
    ---------------------------------------------------------------------------
        V (np.ndarray): Nonnegative matrix of size K x N
        R (int): Rank parameter
        thresh (float): Threshold used as stop criterion (Default value = 0.001)
        L (int): Maximal number of iteration (Default value = 1000)
        W (np.ndarray): Nonnegative matrix of size K x R used for initialization (Default value = None)
        H (np.ndarray): Nonnegative matrix of size R x N used for initialization (Default value = None)
        init (string): {‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’}, default='random'
        norm (bool): Applies max-normalization of columns of final W (Default value = False)
        report (bool): Reports errors during runtime (Default value = False)

    Returns:
    ---------------------------------------------------------------------------
        W (np.ndarray): Nonnegative matrix of size K x R
        H (np.ndarray): Nonnegative matrix of size R x N
        V_approx (np.ndarray): Nonnegative matrix W*H of size K x N
        V_approx_err (float): Error between V and V_approx
        H_W_error (np.ndarray): History of errors of subsequent H and W matrices
    
    """

    K = V.shape[0]
    N = V.shape[1]
    
    if init=='random':
        if W is None:
            W = np.random.rand(K, R)
        if H is None:
            H = np.random.rand(R, N)
            
    elif init=='nndsvd':
        temp_model = sklearnNMF(n_components=R, init='nndsvd', random_state=0, solver='mu', max_iter=1)
        W = temp_model.fit_transform(V) 
        H = temp_model.components_
        
    elif init=='nndsvda':
        temp_model = sklearnNMF(n_components=R, init='nndsvda', random_state=0, solver='mu', max_iter=1)
        W = temp_model.fit_transform(V) 
        H = temp_model.components_
        
    elif init=='nndsvdar':
        temp_model = sklearnNMF(n_components=R, init='nndsvdar', random_state=0, solver='mu', max_iter=1)
        W = temp_model.fit_transform(V) 
        H = temp_model.components_
    
    elif init=='kmeans':
        kmeans = KMeans(n_clusters=R, random_state=42) 
        kmeans.fit(V)
        H_init = kmeans.cluster_centers_
        H_init = np.maximum(H_init, 0)
        temp_model = sklearnNMF(n_components=R, init='custom', random_state=0, solver='mu', max_iter=1)
        W_init = np.random.rand(K, R)
        W_init = np.maximum(W_init, 0)
        W = temp_model.fit_transform(V, W=W_init, H=H_init)
        H = temp_model.components_
    
    else:
        print('Invalid Initialization')
        
    V = V.astype(np.float64)
    W = W.astype(np.float64)
    H = H.astype(np.float64)
    H_W_error = np.zeros((2, L))
    ell = 1
    below_thresh = False
    eps_machine = np.finfo(np.float32).eps
    while not below_thresh and ell <= L:
        H_ell = H
        W_ell = W
        #print('--->',W.shape, H.shape, V.shape)
        H = H * (W.transpose().dot(V) / (W.transpose().dot(W).dot(H) + eps_machine))
        W = W * (V.dot(H.transpose()) / (W.dot(H).dot(H.transpose()) + eps_machine))

        #W = W / np.sum(W, axis=0, keepdims=True)  

        H_error = np.linalg.norm(H-H_ell, ord=2)
        W_error = np.linalg.norm(W - W_ell, ord=2)
        H_W_error[:, ell-1] = [H_error, W_error]
        if report:
            print('Iteration: ', ell, ', H_error: ', H_error, ', W_error: ', W_error)
        if H_error < thresh and W_error < thresh:
            below_thresh = True
            H_W_error = H_W_error[:, 0:ell]
        ell += 1
    if norm:
        for r in range(R):
            v_max = np.max(W[:, r])
            if v_max > 0:
                W[:, r] = W[:, r] / v_max
                H[r, :] = H[r, :] * v_max
    V_approx = W.dot(H)
    V_approx_err = np.linalg.norm(V-V_approx, ord=2)
    return H, W, V_approx_err #H_W_error#V_approx, V_approx_err,