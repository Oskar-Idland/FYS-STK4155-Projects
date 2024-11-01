import autograd.numpy as np
from autograd import elementwise_grad

def identity(z):
	return z

def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def ReLU(z):
	return np.maximum(0, z)


def leaky_ReLU(z):
	return np.where(z > 0, z, 0.01 * z)

def derivate(func):
	if func.__name__ == "sigmoid":
		
		def func(z):
			return sigmoid(z) * (1 - sigmoid(z))

		return func
	
	elif func.__name__ == "RELU":

		def func(z):
			return np.where(z > 0, 1, 0)

		return func

	elif func.__name__ == "LRELU":

		def func(z):
			delta = 10e-4
			return np.where(z > 0, 1, delta)

		return func
	
	else:
		return elementwise_grad(func)