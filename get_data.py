import numpy as np
import os
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_olivetti_faces
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as mms
import cv2 as cv

# Assuming 'utils.py' is in the same directory or a location in your PYTHONPATH
from utils import add_noise  # Make sure utils.py exists and contains the add_noise function

def blobs(n=2000, d=50, c=2, std=25, noise=0.0):
    """Generates synthetic blob data.

    Args:
    ---------------------------------------------------------------------------
        n (int, optional): Number of samples. Defaults to 2000.
        d (int, optional): Number of features. Defaults to 50.
        c (int, optional): Number of clusters. Defaults to 2.
        std (int, optional): Standard deviation of clusters. Defaults to 25.
        noise (float, optional): Noise level to add. Defaults to 0.0

    Returns:
    ---------------------------------------------------------------------------
        tuple: A tuple containing X_train, y_train, X_test, y_test, and the number of classes.
        
    """
    X, y = make_blobs(n_samples=n, centers=c, cluster_std=std,n_features=d, random_state=0)
    scaler = mms()
    X = scaler.fit_transform(X)
    X = add_noise(X, noise_level=noise)
    X = np.abs(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    num_classes = len(np.unique(y))
    
    return X_train, y_train, X_test, y_test, num_classes

def digits(noise=0.0):
    """Loads and preprocesses the digits dataset.

    Args:
    ---------------------------------------------------------------------------
        noise (float, optional): Noise level to add. Defaults to 0.0

    Returns:
    ---------------------------------------------------------------------------
        tuple: A tuple containing X_train, y_train, X_test, y_test, and the number of classes.
        
    """
    mnist = load_digits()
    X = mnist.data
    X = X/X.max()
    
    scaler = mms()
    X = scaler.fit_transform(X)
    X = add_noise(X, noise_level=noise)
    X = np.abs(X)
    y = mnist.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    num_classes = len(np.unique(y))
    
    return X_train, y_train, X_test, y_test, num_classes

def faces(noise=0.0):
    """Loads and preprocesses the Olivetti faces dataset.

    Args:
    ---------------------------------------------------------------------------
        noise (float, optional): Noise level to add (currently unused). Defaults to 0.

    Returns:
    ---------------------------------------------------------------------------
        tuple: A tuple containing X_train, y_train, X_test, y_test, and the number of classes.
    
    """
    X, y = fetch_olivetti_faces(return_X_y=True)
    X2 = []
    for itr in range(X.shape[0]):
        img = X[itr]
        resized_img = cv.resize(img, (32, 32))
        resized_img = img
        flattened_img = resized_img.flatten()
        X2.append(flattened_img)
    X = np.array(X2)
    
    scaler = mms()
    X = scaler.fit_transform(X)
    #X = add_noise(X, noise_level=noise)
    X = np.abs(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    num_classes = len(np.unique(y))
    
    return X_train, y_train, X_test, y_test, num_classes

def f_mnist(size_of_data=0.05, noise=0.0):
    """Loads and preprocesses the Fashion-MNIST dataset.

    Args:
    ---------------------------------------------------------------------------
        size_of_data (float, optional): Fraction of data to use. Defaults to 0.05.
        noise (float, optional): Noise level to add. Defaults to 0.

    Returns:
    ---------------------------------------------------------------------------
        tuple: A tuple containing X_train, y_train, X_test, y_test, and the number of classes.
        
    """
    (X, y), (_, _) = fashion_mnist.load_data()
    X = X.reshape(X.shape[0], -1)
    X=X/X.max()
    
    t_size = 1 - size_of_data
    X0, X1, y0, y = train_test_split(X,y , test_size=t_size, random_state=42)
    
    scaler = mms()
    X0 = scaler.fit_transform(X0)
    X0 = add_noise(X0, noise_level=noise)
    X0 = np.abs(X0)
    
    X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.5, random_state=42)
    num_classes = len(np.unique(y))
    
    return X_train, y_train, X_test, y_test, num_classes

def coil20(noise=0.0):
    """Loads and preprocesses the COIL20 dataset.

    Args:
    ---------------------------------------------------------------------------
        noise (float, optional): Noise level to add. Defaults to 0.0

    Returns:
    ---------------------------------------------------------------------------
        tuple: A tuple containing X_train, y_train, X_test, y_test, and the number of classes.
    
    """
    
    # Check if the file exists, if not, run prepare_coil20.py and store the files in the same directory
    if not os.path.exists("coil20_X_small.npy") or not os.path.exists("coil20_Y.npy"):
        raise FileNotFoundError("COIL20 data files (coil20_X_small.npy, coil20_Y.npy) not found. Please provide them.")
    
    coil20_X = np.load("coil20_X_small.npy")
    coil20_Y = np.load("coil20_Y.npy")
    coil20_X = np.asanyarray(coil20_X)
    coil20_Y = np.asanyarray(coil20_Y)
    
    scaler = mms()
    X0 = scaler.fit_transform(coil20_X)
    
    X0 = add_noise(X0, noise_level=noise)
    X0 = np.abs(X0)
    
    X_train, X_test, y_train, y_test = train_test_split(X0, coil20_Y, test_size=0.5, random_state=42)
    num_classes = len(np.unique(coil20_Y))
    
    return X_train, y_train, X_test, y_test, num_classes

def load_data(data_name, noise_level=0.0):
    """Loads and preprocesses a specified dataset.

    Args:
    ---------------------------------------------------------------------------
        data_name (str): Name of the dataset ('blobs', 'faces', 'digits', 'fashion_mnist', 'coil20').
        noise_level (float, optional): Noise level to add. Defaults to 0.

    Returns:
    ---------------------------------------------------------------------------
        tuple: A tuple containing X_train, y_train, X_test, y_test, and the number of classes.

    Raises:
        ValueError: If an invalid data_name is provided.
    """
    if data_name == 'blobs':
        X_train, y_train, X_test, y_test, num_classes = blobs(n=10000, d=500, c=5, noise=noise_level)
    elif data_name == 'faces':
        X_train, y_train, X_test, y_test, num_classes = faces(noise=noise_level)
    elif data_name == 'digits':
        X_train, y_train, X_test, y_test, num_classes = digits(noise=noise_level)
    elif data_name == 'fashion_mnist':
        X_train, y_train, X_test, y_test, num_classes = f_mnist(size_of_data=0.10, noise=noise_level)
    elif data_name == 'coil20':
        X_train, y_train, X_test, y_test, num_classes = coil20(noise=noise_level)
    else:
        raise ValueError(f"Invalid data_name: {data_name}.  Choose from 'blobs', 'faces', 'digits', 'fashion_mnist', 'coil20'.")

    return X_train, y_train, X_test, y_test, num_classes