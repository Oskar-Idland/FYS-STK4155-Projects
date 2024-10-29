import autograd.numpy as np
from autograd import elementwise_grad

# Activation functions
def identity(X):
    return X

def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))

def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)

def ReLU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))

def leakyReLU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)

def derivate(func):
    if func.__name__ == "ReLU":
        def func(X):
            return np.where(X > 0, 1, 0)
        return func

    elif func.__name__ == "leakyReLU":
        def func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)
        return func

    else:
        return elementwise_grad(func)

# # Activation functions
# def sigmoid(z):
#     #TODO docstring
#     return 1 / (1 + np.exp(-z))

# def sigmoid_der(z):
#     #TODO docstring
#     return sigmoid(z) * (1 - sigmoid(z))

# def ReLU(z):
#     #TODO docstring
#     return np.where(z > 0, z, 0)

# def ReLU_der(z):
#     #TODO docstring
#     return np.where(z > 0, 1, 0)

# def linear(z):
#     #TODO docstring
#     return z  # For regression, the final output layer should be linear

# def linear_der(z):
#     #TODO docstring
#     return np.ones_like(z)

# Score functions
def MSE(y: np.ndarray, y_pred: np.ndarray):
    """
    Calculates the Mean Squared Error (MSE) between the true and predicted values.

    ## Parameters:
    y (np.ndarray): The actual data values.
    y_pred (np.ndarray): The predicted data values from the model.

    ## Returns:
    float: The Mean Squared Error.
    """
    return np.mean((y_pred - y) ** 2)

def MSE_der(y: np.ndarray, y_pred: np.ndarray):
    """
    Calculates the derivative of the MSE with respect to the predicted values.

    ## Parameters:
    y (np.ndarray): The actual data values.
    y_pred (np.ndarray): The predicted data values from the model.

    ## Returns:
    float: The derivative of the Mean Squared Error.
    """
    return 2 * (y_pred - y) / len(y)

def R2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R2 score of the model.

    ## Parameters:
    y (np.ndarray): The actual data values.
    y_pred (np.ndarray): The predicted data values from the model.

    ## Returns:
    float: The R2 score.
    """
    return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

# Cost functions
def CostOLS(target):    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)
    return func

def CostLogReg(target):
    def func(X):   
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )
    return func

def CostCrossEntropy(target):  
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))
    return func


# Other functions
def f(x, coeffs):
    #TODO docstring
    y = 0
    for i, coeff in enumerate(coeffs):
        y += coeff * x**(i)
    return y

