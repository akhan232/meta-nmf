import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt  # Import matplotlib at the top level
import pandas as pd
import seaborn as sns

def add_noise(X, noise_level=0.1):
    """Adds Gaussian noise to a matrix.

    Args:
    ---------------------------------------------------------------------------
        X (numpy.ndarray): The input matrix.
        noise_level (float, optional): The standard deviation of the noise. Defaults to 0.1.

    Returns:
    ---------------------------------------------------------------------------
        numpy.ndarray: The noisy matrix.
        
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def my_nmf(X, n_components=2, init='random', random_state=0): # Renamed to avoid shadowing sklearn's NMF
    """Performs Non-negative Matrix Factorization.

    Args:
    ---------------------------------------------------------------------------
        X (numpy.ndarray): The input matrix.
        n_components (int, optional): The number of components or rank (r). Defaults to 2.
        init (str, optional): Initialization method ('random' or 'nndsvd', 'nndsvda', 'nndsvdar'). Defaults to 'random'.
        random_state (int, optional): Random state for reproducibility. Defaults to 0.

    Returns:
    ---------------------------------------------------------------------------
        tuple: A tuple containing W and H.T.  (Note the transpose)
            W (numpy.ndarray): The basis matrix.
            H.T (numpy.ndarray): The transposed coefficient matrix.
    
    """
    model = NMF(n_components=n_components, init=init, random_state=random_state) # Allows init and random_state to be specified
    W = model.fit_transform(X)
    H = model.components_
    return W, H.T


def safe_divide(numerator, denominator):  # More descriptive name
    """Divides numerator by denominator, handling potential division by zero.

    Args:
    ---------------------------------------------------------------------------
        numerator (numpy.ndarray): The numerator.
        denominator (numpy.ndarray): The denominator.

    Returns:
    ---------------------------------------------------------------------------
        numpy.ndarray: The result of the division, with 0 where denominator is 0.
        
    """
    out = np.copy(numerator) #copy is important to not change the original matrix
    return np.divide(out, denominator, out=out, where=denominator != 0) # more efficient way


def normalize(matrix, axis=0):  # Added default axis and more flexible
    """Normalizes a matrix along a specified axis (L2 norm).

    Args:
    ---------------------------------------------------------------------------
        matrix (numpy.ndarray): The input matrix.
        axis (int, optional): The axis to normalize along (0 for columns, 1 for rows). Defaults to 0.

    Returns:
    ---------------------------------------------------------------------------
        numpy.ndarray: The normalized matrix.
        
    """
    row_norms = np.linalg.norm(matrix, axis=axis, ord=2, keepdims=True) #keepdims is important
    return safe_divide(matrix, row_norms)  # Use safe_divide


def get_random_task(data, labels, num_samples=2):
    """Selects a random subset of data and labels for a task.

    Args:
    ---------------------------------------------------------------------------
        data (numpy.ndarray): The data.
        labels (numpy.ndarray): The labels.
        num_samples (int, optional): The number of samples to select. Defaults to 2.

    Returns:
    ---------------------------------------------------------------------------
        tuple: A tuple containing the task samples and task labels.
        
    """
    random_indices = np.random.choice(len(data), size=num_samples, replace=False)
    task_samples = data[random_indices]
    task_labels = labels[random_indices]
    task = (task_samples, task_labels)
    return task

def generate_tasks(data, labels, nb_tasks=3, task_size=3):
    """Generates a set of tasks, including a target task.

    Args:
    ---------------------------------------------------------------------------
        data (numpy.ndarray): The data.
        labels (numpy.ndarray): The labels.
        nb_tasks (int, optional): The number of additional tasks. Defaults to 3.
        task_size (int, optional): The size of each additional task. Defaults to 3.

    Returns:
    ---------------------------------------------------------------------------
        dict: A dictionary containing the tasks, including 'task_t' for the target task.
    
    """
    Tasks = {}
    Tasks.clear()
    # Target task
    all = data.shape[0]
    #print('all',all)
    task_t = get_random_task(data,labels, num_samples = all)
    Tasks['task_t'] = task_t
    # other task
    for i in range(nb_tasks):
        task = get_random_task(data,labels, num_samples = task_size)
        Tasks['task'+str(i)] = task
    return Tasks

def my_plot(X, y):
    """Plots data points with labels using seaborn.

    Args:
    ---------------------------------------------------------------------------
        X (numpy.ndarray): The data.
        y (numpy.ndarray): The labels.
        
    """
    df = pd.DataFrame(data=X, columns=['x1', 'x2'])  # Assuming 2D data
    df['label'] = y
    sns.lmplot(x="x1", y="x2", data=df, hue='label', fit_reg=False)
    plt.show()


def bases(W):
    """Reshapes the columns of W into a set of basis matrices.

    Args:
    ---------------------------------------------------------------------------
        W (numpy.ndarray): The matrix W.

    Returns:
    ---------------------------------------------------------------------------
        numpy.ndarray: An array of basis matrices.
        
    """
    d, k = W.shape
    m = int(np.sqrt(d))
    bases_W = W.reshape(m, m, k).transpose(2, 0, 1)
    return bases_W


def show_bases(W):
    """Plot a set of basis matrices.

    Args:
    ---------------------------------------------------------------------------
        W (numpy.ndarray): An array of basis matrices.
        
    """
    nb_bases = W.shape[0]
    nb_columns = min(nb_bases, 4)  # Limit to a reasonable number of columns
    nb_rows = int(np.ceil(nb_bases / nb_columns)) # Use ceil to ensure all bases are shown

    fig, axes = plt.subplots(nb_rows, nb_columns, figsize=(nb_columns * 3, nb_rows * 3)) #Adjusted figure size
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    for i in range(nb_bases):
        ax = axes[i]
        basis = W[i]
        ax.imshow(basis, interpolation='spline16') #Consistent interpolation
        ax.axis('off')  # Hide axis ticks and labels

    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    plt.tight_layout()
    plt.show()