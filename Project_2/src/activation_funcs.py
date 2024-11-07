import autograd.numpy as np
from autograd import elementwise_grad


def identity(X):
    '''The `identity` function simply returns the input `X` without any modifications.
    
    Parameters
    ----------
    X
        The parameter `X` in the `identity` function represents any input value that is passed to the
    function. The function simply returns the same value that is passed to it, hence it is called an
    "identity" function.
    
    Returns
    -------
        The function `identity` returns the input `X` without any modifications.
    
    '''
    return X


def sigmoid(X):
    '''The function `sigmoid` calculates the sigmoid function for the input array `X` handling potential
    FloatingPointError.
    
    Parameters
    ----------
    X
        The `sigmoid` function calculates the sigmoid activation function for the input `X`.

    Returns
    -------
        The function `sigmoid` returns the sigmoid function applied to the input `X`. If there is a
    `FloatingPointError` during the calculation, it returns an array where each element is set to 1 if
    the corresponding element in `X` is greater than 0, and 0 otherwise.
    
    '''
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))


def softmax(X):
    '''
    The softmax function calculates the softmax function for an input array X.
    
    Parameters
    ----------
    X : numpy.ndarray
        The input array for which to compute the softmax activation function.
    
    Returns
    -------
    numpy.ndarray
        The softmax values of the input array X. The softmax function is applied to each row of the input array X,
        and it normalizes the values in each row to be between 0 and 1, such that the sum of the values in each row is 1.
    '''
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X):
    '''
    The RELU function calculates the Rectified Linear Unit (ReLU) activation function for an input array X.
    
    Parameters
    ----------
    X : numpy.ndarray
        The input array for which to compute the ReLU activation function.
    
    Returns
    -------
    numpy.ndarray
        The ReLU values of the input array X. The ReLU function returns the input value if it is greater than 0,
        otherwise, it returns 0.
    '''
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X):
    '''
    The LRELU function calculates the Leaky Rectified Linear Unit (Leaky ReLU) activation function for an input array X.
    
    Parameters
    ----------
    X : numpy.ndarray
        The input array for which to compute the Leaky ReLU activation function.
    
    Returns
    -------
    numpy.ndarray
        The Leaky ReLU values of the input array X. The Leaky ReLU function returns the input value if it is greater than 0,
        otherwise, it returns a small fraction (delta) of the input value.
    '''
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


def derivate(func):
    '''
    The derivate function returns the derivative of the given activation function.
    
    Parameters
    ----------
    func : function
        The activation function for which to compute the derivative. Supported functions are RELU and LRELU.
    
    Returns
    -------
    function
        The derivative function of the given activation function.
    '''
    if func.__name__ == "RELU":
        def func(X):
            return np.where(X > 0, 1, 0)
        return func

    elif func.__name__ == "LRELU":
        def func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)
        return func

    else:
        return elementwise_grad(func)