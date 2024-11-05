import autograd.numpy as np

def CostOLS(target):
    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


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