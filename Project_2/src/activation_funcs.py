import autograd.numpy as np
from autograd import elementwise_grad

def identity(X):
    return X

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLU(z):
    return np.where(z > 0, z, 0.01 * z)

def leaky_ReLU_derivative(z):
    return np.where(z > 0, 1, 0.01)
