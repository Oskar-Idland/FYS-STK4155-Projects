import numpy as np


def MSE(pred, targets):
    return np.mean((pred - targets) ** 2)

def MSE_derivative(pred, targets):
    return 2 * (pred - targets) / len(pred)

def R2(pred, targets):    
    return 1 - np.sum((targets - pred) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

def create_X(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Creates the design matrix X.

    ## Parameters:
    x (np.ndarray): Independent variable.
    y (np.ndarray): Independent variable.
    n (int): The degree of the polynomial features.

    ## Returns:
    np.ndarray: The design matrix X.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = int(len(x)*len(y))            # Number of rows in the design matrix
    l = int((n+1)*(n+2)/2)            # Number of columns in the design matrix
    X = np.ones((N, l))
    
    xx, yy = np.meshgrid(x, y)        # Make a meshgrid to get all possible combinations of x and y values
    xx = xx.flatten()
    yy = yy.flatten()

    idx = 1
    for i in range(1, n+1):
        for j in range(i+1):
            X[:, idx] = xx**(i-j) * yy**j
            idx += 1

    return X

def FrankeFunction(x: float | np.ndarray,y: float | np.ndarray) -> float | np.ndarray:
    """ 
    Generates a surface plot of the Franke function.
    # Parameters:
    x (float | np.ndarray): The x-value(s).
    y (float | np.ndarray): The y-value(s).
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4