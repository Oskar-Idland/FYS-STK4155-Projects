import numpy as np


def MSE(pred, targets):
    return np.mean((pred - targets) ** 2)

def MSE_derivative(pred, targets):
    return 2 * (pred - targets) / len(pred)

def R2(pred, targets):    
    return 1 - np.sum((targets - pred) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

def CostLogReg(pred, targets):
    """
    Compute the logistic regression cost (binary cross-entropy loss).
    
    Parameters:
        pred (np.ndarray): Model predictions, should be between 0 and 1
        targets (np.ndarray): True binary labels (0 or 1)
    
    Returns:
        float: Average binary cross-entropy loss
    """
    eps = 1e-15  # Small constant to prevent log(0)
    # Clip predictions to avoid numerical instability
    pred = np.clip(pred, eps, 1 - eps)
    
    # Binary cross-entropy formula: -[y*log(p) + (1-y)*log(1-p)]
    cost = -np.mean(targets * np.log(pred) + (1 - targets) * np.log(1 - pred))
    return cost

def CostLogReg_derivative(pred, targets):
    """
    Compute the derivative of logistic regression cost function.
    
    Parameters:
        pred (np.ndarray): Model predictions, should be between 0 and 1
        targets (np.ndarray): True binary labels (0 or 1)
    
    Returns:
        np.ndarray: Gradient of the cost with respect to predictions
    """
    eps = 1e-15  # Same small constant for numerical stability
    pred = np.clip(pred, eps, 1 - eps)
    
    # Derivative of binary cross-entropy: (p-y)/(p(1-p))
    derivative = (pred - targets) / (pred * (1 - pred))
    return derivative

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
    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y**k)
    return X

def optimal_parameters(matrix: np.ndarray, x: np.ndarray, y: np.ndarray, max_or_min: str = 'min') -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the indices of the minimum value in a matrix.

    ## Parameters:
    matrix (np.ndarray): The matrix to search.
    x (np.ndarray): The x-values.
    y (np.ndarray): The y-values.
    max_or_min (str ['min' | 'max']): Whether to find the maximum or minimum value. Default is 'min'.

    ## Returns:
    tuple[np.ndarray, np.ndarray]: The indices of the minimum value.
    """
    if max_or_min == 'max':
        idx = np.unravel_index(np.nanargmax(matrix), matrix.shape, )
    elif max_or_min == 'min':
        idx = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    else:
        raise ValueError("max_or_min must be either 'max' or 'min'.")
    return x[idx[1]], y[idx[0]]

# Activation functions
def identity(X):
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